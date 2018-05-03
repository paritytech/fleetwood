#![cfg_attr(not(feature = "std"), no_std)]
#![feature(never_type, const_fn, specialization, overlapping_marker_traits)]
#![allow(dead_code)]

extern crate core;
extern crate either;
extern crate serde;
extern crate tiny_keccak;

pub mod shim {
    pub struct U256(());
    pub struct H256(());

    impl U256 {
        pub const fn new() -> Self {
            U256(())
        }
    }

    impl From<U256> for H256 {
        fn from(_other: U256) -> Self {
            unimplemented!()
        }
    }

    impl From<usize> for U256 {
        fn from(_other: usize) -> Self {
            unimplemented!()
        }
    }

    pub fn write(_key: &H256, _val: &[u8; 32]) {
        unimplemented!()
    }

    pub fn read(_key: &H256) -> [u8; 32] {
        unimplemented!()
    }

    pub fn write_state(_len: usize, _ptr: *const u8) {
        unimplemented!()
    }

    pub fn read_state(_len: usize, _ptr: *mut u8) {
        unimplemented!()
    }
}

use shim as pwasm_ethereum;

mod pwasm {
    use core::hash::Hash;
    use core::iter;
    use core::marker::PhantomData;
    use core::result::Result as StdResult;

    use pwasm_ethereum::{self, *};

    use serde::{Deserialize, Serialize};

    // Replacement for `HashMap` that doesn't require serializing/deserializing the
    // full map every time you attempt to run a handler.
    pub struct Database<K, V> {
        seed: u64,
        _marker: PhantomData<(K, V)>,
    }

    impl<K: Hash, V: Serialize + for<'a> Deserialize<'a>> Database<K, V> {
        fn insert(&mut self, _key: &K, _val: V) {
            unimplemented!()
        }

        fn get(&self, _key: &K) -> V {
            unimplemented!()
        }
    }

    #[derive(Debug, Copy, Clone, Default)]
    pub struct NoMethodError;

    pub struct TxInfo(());

    impl TxInfo {
        #[cfg(test)]
        pub fn new() -> Self {
            TxInfo(())
        }
    }

    pub type Result<T> = StdResult<T, NoMethodError>;

    pub struct Request {
        pub function_selector: [u8; 4],
        // TODO: Work out what to do here. We don't need to optimize this by avoiding serialization costs
        //       since the "optimized" route is for testing only.
        pub value: Box<Any>,
    }

    fn make_selector<'a, I: IntoIterator<Item = &'a str>>(iter: I) -> [u8; 4] {
        let mut keccak = ::tiny_keccak::Keccak::new_sha3_256();

        for element in iter {
            keccak.update(element.as_bytes());
        }

        let mut out = [0u8; 4];
        keccak.finalize(&mut out);
        out
    }

    // For testing
    impl Request {
        pub fn new<M: Message>(input: M::Input) -> Self
        // TODO
        where
            M::Input: 'static
        {
            let sel = M::selector();

            Request {
                function_selector: sel,
                value: Box::new(input),
            }
        }
    }

    pub trait Response {
        fn output_for<M: Message>(self) -> Option<M::Output>
        where
            M::Output: Any;
    }

    use std::any::Any;

    impl<Head: Any, Rest> Response for Either<Head, Rest>
    where
        Rest: Response,
    {
        fn output_for<M: Message>(self) -> Option<M::Output>
        where
            M::Output: Any,
        {
            use std::{mem, ptr};
            use std::any::TypeId;

            match self {
                Either::Left(left) => if TypeId::of::<Head>() == TypeId::of::<M::Output>() {
                    let out = unsafe { ptr::read(&left as *const Head as *const M::Output) };

                    mem::forget(left);

                    Some(out)
                } else {
                    None
                },
                Either::Right(right) => right.output_for::<M>(),
            }
        }
    }

    impl Response for ! {
        fn output_for<M: Message>(self) -> Option<M::Output>
        where
            M::Output: Any,
        {
            None
        }
    }

    // TODO: Should we build on deployment or should there be a "deploy" step? Building on deployment is _way_
    //       simpler.
    pub const fn deploy_data() -> DeployData {
        DeployData {
            deployer: U256::new(),
        }
    }

    pub struct ContractInstance<'a, S, T: 'a> {
        env: &'a EthEnv,
        state: S,
        contract: &'a T,
    }

    impl<'a, S, T> ContractInstance<'a, S, T>
    where
        T: ContractDef<S>,
        T::Output: Response,
    {
        pub fn call<M: Message>(&mut self, input: M::Input) -> Option<M::Output>
        where
            T: RespondsTo<M>,
            M::Output: 'static,
            // TODO
            M::Input: 'static
        {
            Response::output_for::<M>(self.contract.send_request(
                self.env,
                &mut self.state,
                Request::new::<M>(input),
            ))
        }
    }

    pub trait ContractDef<State> {
        type Output: Response + 'static;

        // We have this function to allow easy testing for users. For a lot of functions they don't need
        // to deploy to the blockchain at all.
        fn send_request(&self, _env: &EthEnv, state: &mut State, input: Request) -> Self::Output;

        fn construct(&self, txdata: TxInfo) -> State;

        fn deploy<'a>(&'a self, env: &'a EthEnv, txdata: TxInfo) -> ContractInstance<'a, State, Self>
        where
            Self: Sized,
        {
            let state = self.construct(txdata);
            ContractInstance {
                env,
                state,
                contract: self,
            }
        }

        fn call<M: Message>(
            &self,
            env: &EthEnv,
            state: &mut State,
            input: M::Input,
        ) -> Option<M::Output>
        where
            Self::Output: Response,
            Self: Sized,
            M::Output: 'static,
            // TODO
            M::Input: 'static,
        {
            Response::output_for::<M>(self.send_request(env, state, Request::new::<M>(input)))
        }
    }

    impl<C, H> ContractDef<C::Output> for Contract<C, H>
    where
        C: Constructor,
        H: Handlers<C::Output>,
    {
        type Output = H::Output;

        fn construct(&self, txdata: TxInfo) -> C::Output {
            self.constructor.call(txdata)
        }

        fn send_request(
            &self,
            env: &EthEnv,
            state: &mut C::Output,
            input: Request,
        ) -> Self::Output {
            self.handlers.handle(env, state, input).expect("No method")
        }
    }

    pub struct MessageHandler<F, SW> {
        handler: F,
        write_state: PhantomData<SW>,
    }

    pub trait StateWriter<State> {
        fn write_state(state: &State);
    }

    pub struct NoWriteState;
    pub struct WriteState;

    impl<S> StateWriter<S> for NoWriteState {
        fn write_state(_state: &S) {}
    }

    impl<S> StateWriter<S> for WriteState {
        fn write_state(state: &S) {
            write_state_generic(state)
        }
    }

    impl<F> MessageHandler<F, NoWriteState> {
        fn new(hnd: F) -> Self {
            MessageHandler {
                handler: hnd,
                write_state: PhantomData,
            }
        }
    }

    impl<F> MessageHandler<F, WriteState> {
        fn new_mut(hnd: F) -> Self {
            MessageHandler {
                handler: hnd,
                write_state: PhantomData,
            }
        }
    }

    impl<F: Copy, S> Copy for MessageHandler<F, S> {}

    impl<F: Clone, S> Clone for MessageHandler<F, S> {
        fn clone(&self) -> Self {
            MessageHandler {
                handler: self.handler.clone(),
                write_state: self.write_state,
            }
        }
    }

    pub trait Handlers<State> {
        type Output: Response + 'static;

        fn handle(&self, env: &EthEnv, state: &mut State, request: Request)
            -> Result<Self::Output>;
    }

    use either::Either;

    macro_rules! impl_handlers {
        ($statename:ident, $($any:tt)*) => {
    impl<M, Rest, $statename, SW> Handlers<$statename> for ((PhantomData<M>, MessageHandler<for<'a> fn(&'a EthEnv, &'a $($any)*, M::Input) -> M::Output, SW>), Rest)
    where
        M: Message,
        <M as Message>::Input: 'static,
        <M as Message>::Output: 'static,
        Rest: Handlers<$statename>,
        SW: StateWriter<$statename>,
    {
        type Output = Either<<M as Message>::Output, <Rest as Handlers<$statename>>::Output>;

        // TODO: Pre-hash?
        fn handle(&self, env: &EthEnv, state: &mut $statename, request: Request) -> Result<Self::Output> {
            fn deserialize<Out>(req: Request) -> Out where Out: 'static {
                *req.value.downcast::<Out>().unwrap()
            }

            if M::selector() == request.function_selector {
                let head = self.0;
                let out = (head.1.handler)(env, state, deserialize(request));
                SW::write_state(&state);
                Ok(Either::Left(out))
            } else {
                self.1.handle(env, state, request).map(Either::Right)
            }
        }
    }
        }
    }

    impl_handlers!(State, State);
    impl_handlers!(State, mut State);

    impl<State> Handlers<State> for () {
        type Output = !;

        fn handle(
            &self,
            _env: &EthEnv,
            _state: &mut State,
            _request: Request,
        ) -> Result<Self::Output> {
            Err(NoMethodError)
        }
    }

    fn write_state_generic<T>(val: &T) {
        unsafe { write_state(::core::mem::size_of::<T>(), val as *const T as *const u8) };
    }

    pub trait ArgSignature {
        type Iter: IntoIterator<Item = &'static str>;
        fn arg_sig() -> Self::Iter;
    }

    // We use an iterator so that we can implement this with macro_rules macros
    // without allocating
    pub trait SolidityType {
        type Iter: IntoIterator<Item = &'static str>;
        fn solname() -> Self::Iter;
    }

    macro_rules! impl_soltype {
        ($typ:ty, $out:expr) => {
            impl SolidityType for $typ {
                type Iter = iter::Once<&'static str>;

                fn solname() -> Self::Iter {
                    iter::once($out)
                }
            }
        };
    }

    impl_soltype!(bool, "bool");
    impl_soltype!(u8, "uint8");
    impl_soltype!(u16, "uint16");
    impl_soltype!(u32, "uint32");
    impl_soltype!(u64, "uint64");
    impl_soltype!(i8, "int8");
    impl_soltype!(i16, "int16");
    impl_soltype!(i32, "int32");
    impl_soltype!(i64, "int64");

    macro_rules! sol_array {
        (@capture $e:expr) => {
            stringify!($e)
        };
        ($n:expr) => {
            impl<T> SolidityType for [T; $n]
            where T: SolidityType
            {
                type Iter = iter::Chain<<T::Iter as IntoIterator>::IntoIter, iter::Once<&'static str>>;

                fn solname() -> Self::Iter {
                    T::solname().into_iter().chain(iter::once(sol_array!(@capture [$n])))
                }
            }
        };
        ($n:expr $(, $rest:expr)*) => {
            sol_array!($n);
            sol_array!($($rest),*);
        };
    }

    sol_array!(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128, 256, 512, 1024
    );

    impl<T> SolidityType for Vec<T>
    where
        T: SolidityType,
    {
        type Iter = iter::Chain<<T::Iter as IntoIterator>::IntoIter, iter::Once<&'static str>>;

        fn solname() -> Self::Iter {
            T::solname().into_iter().chain(iter::once("[]"))
        }
    }

    impl<T> ArgSignature for T
    where
        T: SolidityType,
    {
        type Iter = iter::Chain<
            iter::Chain<iter::Once<&'static str>, <T::Iter as IntoIterator>::IntoIter>,
            iter::Once<&'static str>,
        >;

        fn arg_sig() -> Self::Iter {
            iter::once("(")
                .chain(T::solname().into_iter())
                .chain(iter::once(")"))
        }
    }

    macro_rules! tup_sig {
        (@chain_type_inner $name:ident) => {
            <$name::Iter as ::core::iter::IntoIterator>::IntoIter
        };
        (@chain_type_inner $name:ident $($rest:ident)*) => {
            iter::Chain<
                iter::Chain<
                    <$name::Iter as ::core::iter::IntoIterator>::IntoIter,
                    iter::Once<&'static str>,
                >,
                tup_sig!(@chain_type_inner $($rest)*),
            >
        };
        (@chain_type ) => {
            iter::Once<&'static str>
        };
        (@chain_type $($name:ident)+) => {
            iter::Chain<
                iter::Chain<iter::Once<&'static str>, tup_sig!(@chain_type_inner $($name)*)>,
                iter::Once<&'static str>,
            >
        };
        (@chain_inner $name:ident) => {
            $name::solname().into_iter()
        };
        (@chain_inner $name:ident $($rest:ident)+) => {
            $name::solname().into_iter().chain(iter::once(","))
                .chain(tup_sig!(@chain_inner $($rest)+))
        };
        (@chain ) => {
            iter::once("()");
        };
        (@chain $($name:ident)+) => {
            iter::once("(").chain(tup_sig!(@chain_inner $($name)+)).chain(iter::once(")"));
        };
        ($($name:ident),*) => {
            impl<$($name),*> ArgSignature for ($($name,)*)
            where
            $(
                $name : SolidityType,
            )*
            {
                type Iter = tup_sig!(@chain_type $($name)*);

                fn arg_sig() -> Self::Iter {
                    tup_sig!(@chain $($name)*)
                }
            }
        };
    }

    macro_rules! tup_sigs {
        ($name:ident $($rest:ident)*) => {
            tup_sig!($name $(, $rest)*);
            tup_sigs!($($rest)*);
        };
        () => {
            tup_sig!();
        };
    }

    tup_sigs!(A B C D E F G H I J K L M N O P Q);

    pub trait Message {
        type Input: for<'a> Deserialize<'a> + ArgSignature;
        type Output: Serialize;

        // TODO: Pre-hash?
        const NAME: &'static str;
    }

    pub trait MessageExt {
        type Iter: IntoIterator<Item = &'static str>;
        fn signature() -> Self::Iter;

        fn selector() -> [u8; 4] {
            make_selector(Self::signature())
        }
    }

    impl<T> MessageExt for T
    where
        T: Message,
        T::Input: ArgSignature,
    {
        type Iter = iter::Chain<
            iter::Once<&'static str>,
            <<T::Input as ArgSignature>::Iter as IntoIterator>::IntoIter,
        >;

        fn signature() -> Self::Iter {
            iter::once(Self::NAME).chain(<Self as Message>::Input::arg_sig().into_iter())
        }
    }

    pub trait RespondsTo<T> {}

    impl<C, Msg, Handler, Rest> RespondsTo<Msg> for Contract<C, ((PhantomData<Msg>, Handler), Rest)> {}
    impl<C, Msg, Head, Rest> RespondsTo<Msg> for Contract<C, (Head, Rest)>
    where
        Contract<C, Rest>: RespondsTo<Msg>,
    {
    }

    // This is essentially a hack to get around the fact that `FnOnce`'s internals are
    // unstable
    pub trait Constructor {
        type Output;

        fn call(&self, txinfo: TxInfo) -> Self::Output;
    }

    impl<Out> Constructor for fn(TxInfo) -> Out {
        type Output = Out;

        fn call(&self, txinfo: TxInfo) -> Self::Output {
            self(txinfo)
        }
    }

    pub struct Contract<Constructor, Handle> {
        constructor: Constructor,
        handlers: Handle,
    }

    impl Contract<(), ()> {
        pub const fn new() -> Self {
            Contract {
                constructor: (),
                handlers: (),
            }
        }

        // We enforce the `'static` bound here instead of when we check the
        // `ContractDef` bound so that we get better error messages.
        //
        // It's necessary since we serialize the closure's state at deploy-time
        // and then deserialize it on-chain.
        //
        // Also, we shouldn't allow you to put handlers before the constructor,
        // since that's a footgun (it'll work if the state and init are the same
        // type but not otherwise).

        pub const fn constructor<Out>(
            self,
            constructor: fn(TxInfo) -> Out,
        ) -> Contract<fn(TxInfo) -> Out, ()> {
            Contract {
                constructor: constructor,
                handlers: self.handlers,
            }
        }
    }

    type Handler<M, St> =
        for<'a> fn(&'a EthEnv, &'a St, <M as Message>::Input) -> <M as Message>::Output;
    type HandlerMut<M, St> =
        for<'a> fn(&'a EthEnv, &'a mut St, <M as Message>::Input) -> <M as Message>::Output;

    impl<Cons: Constructor + Copy, Handle: Handlers<Cons::Output> + Copy> Contract<Cons, Handle> {
        const fn with_handler<M, H, SW>(
            self,
            handler: H,
        ) -> Contract<Cons, ((PhantomData<M>, MessageHandler<H, SW>), Handle)> {
            Contract {
                constructor: self.constructor,
                handlers: (
                    (
                        PhantomData,
                        MessageHandler {
                            handler,
                            write_state: PhantomData,
                        },
                    ),
                    self.handlers,
                ),
            }
        }

        pub const fn on_msg<M>(
            self,
            handler: Handler<M, Cons::Output>,
        ) -> Contract<
            Cons,
            (
                (
                    PhantomData<M>,
                    MessageHandler<Handler<M, Cons::Output>, NoWriteState>,
                ),
                Handle,
            ),
        >
        where
            M: Message,
        {
            self.with_handler(handler)
        }

        pub const fn on_msg_mut<M>(
            self,
            handler: HandlerMut<M, Cons::Output>,
        ) -> Contract<
            Cons,
            (
                (
                    PhantomData<M>,
                    MessageHandler<HandlerMut<M, Cons::Output>, WriteState>,
                ),
                Handle,
            ),
        >
        where
            M: Message,
        {
            self.with_handler(handler)
        }
    }

    pub struct DeployData {
        deployer: U256,
    }

    // Will it ever be possible to get arbitrary blocks?
    pub struct Block(());

    impl Block {
        pub fn beneficiary(&self) -> U256 {
            unimplemented!();
        }

        pub fn timestamp(&self) -> U256 {
            unimplemented!();
        }

        pub fn number(&self) -> U256 {
            unimplemented!();
        }

        pub fn difficulty(&self) -> U256 {
            unimplemented!();
        }

        pub fn gas_limit(&self) -> U256 {
            unimplemented!();
        }
    }

    pub struct Account {
        address: U256,
    }

    impl Account {
        pub fn balance(&self) -> U256 {
            unimplemented!()
        }
    }

    pub struct EthEnv(());

    impl EthEnv {
        #[cfg(test)]
        pub fn new() -> Self {
            EthEnv(())
        }
    }

    pub trait RemoteContract {}

    impl EthEnv {
        // TODO: Do we use an owned blockchain since everything's accessed through methods
        //       anyway?
        fn blockchain(&self) -> &BlockChain {
            unimplemented!()
        }

        fn account_at(&self, _addr: U256) -> Result<Account> {
            unimplemented!()
        }

        // We use different types for remote vs local contracts since
        // they require different functions to get the code

        // `impl Contract` is a `RemoteContract`
        fn contract_at(&self, _addr: U256) -> Result<&impl RemoteContract> {
            struct Dummy;

            impl RemoteContract for Dummy {}

            Ok(&Dummy)
        }

        // `impl Contract` is a `LocalContract`
        fn current_contract(&self) -> Result<&impl RemoteContract> {
            struct Dummy;

            impl RemoteContract for Dummy {}

            Ok(&Dummy)
        }
    }

    pub struct BlockChain(());

    impl BlockChain {
        pub fn current(&self) -> &Block {
            unimplemented!();
        }

        pub fn block_hash(&self, _number: u8) -> U256 {
            unimplemented!();
        }
    }

    pub trait ExternalContract {
        // Compiles to `CODESIZE` + `CODECOPY` (TODO: This should be dynamically-sized but
        // owned but we can't do that without `alloca`, so we can just write a `Box<[u8]>`-
        // esque type that allocates on the "heap")
        fn code(&self) -> &[u8];
        fn call(&self, method: &[u8], args: &[u8]) -> &[u8];
    }

    // This is essentially writing `T`'s bytes to a static location in the contract's state and
    // so is vulnerable to type confusion and other nasty bugs. Our API makes it safe by
    // enforcing essentially a state machine where only one type is valid in this location at
    // any given time.
    //
    // This should be an inbuilt fn from the runtime but this is a shim
    unsafe fn write_state(size: usize, pointer: *const u8) {
        use core::ptr;
        use pwasm_ethereum::{H256, U256};

        const STEP_SIZE: usize = 32;

        // TODO: uninit
        let mut buffer = [0; STEP_SIZE];

        // We can't write more than `usize::MAX` bytes anyway since our
        // state is a Rust struct.
        for i in 0..size / STEP_SIZE {
            let key = H256::from(U256::from(i));

            let offset = i * STEP_SIZE;
            let remaining = size - offset;

            ptr::copy_nonoverlapping(
                pointer.offset(offset as isize),
                buffer.as_mut_ptr(),
                STEP_SIZE.min(remaining),
            );

            pwasm_ethereum::write(&key, &buffer);
        }
    }

    // This should be an inbuilt fn from the runtime but this is a shim
    unsafe fn read_state(size: usize, pointer: *mut u8) {
        use core::ptr;
        use pwasm_ethereum::{H256, U256};

        const STEP_SIZE: usize = 32;

        // We can't write more than `usize::MAX` bytes anyway since our
        // state is a Rust struct.
        for i in 0..size / STEP_SIZE {
            let key = H256::from(U256::from(i));

            let offset = i * STEP_SIZE;
            let remaining = size - offset;

            let buffer = pwasm_ethereum::read(&key);

            ptr::copy_nonoverlapping(
                buffer.as_ptr(),
                pointer.offset(offset as isize),
                STEP_SIZE.min(remaining),
            );
        }
    }
}

macro_rules! messages {
    ($name:ident($($typ:ty),*); $($rest:tt)*) => {
        messages!($name($($typ),*) -> (); $($rest)*);
    };
    ($name:ident($($typ:ty),*) -> $out:ty; $($rest:tt)*) => {
        struct $name;

        impl $crate::pwasm::Message for $name {
            type Input = ($($typ),*);
            type Output = $out;

            const NAME: &'static str = stringify!($name);
        }

        messages!($($rest)*);
    };
    () => {}
}

mod example {
    use pwasm::{Contract, ContractDef};

    messages! {
        Add(u32);
        Get() -> u32;
    }

    pub struct State {
        current: u32,
        calls_to_add: usize,
    }

    pub const fn contract() -> impl ContractDef<State> {
        Contract::new()
            .constructor(|_txdata| State {
                current: 1u32,
                calls_to_add: 0usize,
            })
            .on_msg_mut::<Add>(|_env, state, to_add| {
                state.calls_to_add += 1;
                state.current += to_add;
            })
            .on_msg::<Get>(|_env, state, ()| state.current)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn tuple_sigs() {
        use pwasm::ArgSignature;

        assert_eq!(
            <(u8, u16, u32)>::arg_sig().collect::<String>(),
            "(uint8,uint16,uint32)"
        );
    }

    #[test]
    fn message_sigs() {
        use pwasm::MessageExt;

        messages! {
            Foo(u32, u64, u16);
            UseArray([u32; 5], Vec<bool>);
            Get() -> usize;
            OneArg(u64);
        }

        assert_eq!(
            Foo::signature().into_iter().collect::<String>(),
            "Foo(uint32,uint64,uint16)"
        );
        assert_eq!(
            UseArray::signature().into_iter().collect::<String>(),
            "UseArray(uint32[5],bool[])"
        );
        assert_eq!(Get::signature().into_iter().collect::<String>(), "Get()");
        assert_eq!(
            OneArg::signature().into_iter().collect::<String>(),
            "OneArg(uint64)"
        );
    }

    #[test]
    fn request() {
        #![allow(non_camel_case_types)]

        use pwasm::Request;

        messages! {
            foo(u32, u64, u16);
        }

        let request = Request::new::<foo>((0, 1, 2));
        // assert_eq!(request.function_selector, [0; 4]);
    }

    #[test]
    fn contract() {
        use pwasm::{Contract, ContractDef, EthEnv, TxInfo};

        messages! {
            Add(u32);
            Get() -> u32;
            Unused();
        }

        pub struct State {
            current: u32,
            calls_to_add: usize,
        }

        // TODO: I don't know how this will work
        let env = EthEnv::new();

        let definition = Contract::new()
            .constructor(|_txdata| State {
                current: 1u32,
                calls_to_add: 0usize,
            })
            .on_msg_mut::<Add>(|_env, state, to_add| {
                state.calls_to_add += 1;
                state.current += to_add;
            })
            .on_msg::<Get>(|_env, state, ()| state.current);

        // `TxInfo` is the information on the existing transaction
        let mut contract = definition.deploy(&env, TxInfo::new());

        let out: () = contract.call::<Add>(1).unwrap();
        let val: u32 = contract.call::<Get>(()).unwrap();

        // Doesn't compile
        // contract.call::<Unused>(()).unwrap();

        assert_eq!(val, 2);
    }
}
