#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "nightly", feature(overlapping_marker_traits))]
#![feature(never_type)]
#![allow(dead_code)]

extern crate serde;
extern crate either;
extern crate core;
extern crate tiny_keccak;

pub mod shim {
    pub struct U256(());
    pub struct H256(());

    impl U256 {
        pub fn new() -> Self {
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

    pub struct NoMethodError;

    pub struct TxInfo(());

    pub type Result<T> = StdResult<T, NoMethodError>;

    pub struct Request {
        pub function_selector: [u8; 4],
    }

    // For testing
    impl Request {
        pub fn new<M: Message>(_input: M::Input) -> Self {
            let mut keccak = ::tiny_keccak::Keccak::new_sha3_256();
            for element in M::signature() {
                keccak.update(element.as_bytes());
            }
            let mut out = [0u8; 4];
            keccak.finalize(&mut out);

            Request {
                function_selector: out,
            }
        }
    }

    pub trait Response<To: Message> {
        fn to(data: To::Output) -> Self;
    }

    pub fn deploy_data() -> DeployData {
        DeployData {
            deployer: U256::new(),
        }
    }

    pub trait ContractDef<State> {
        type Output;

        fn handle_message(self, state: &mut State, input: Request) -> Self::Output;
    }

    impl<C, H> ContractDef<C::Output> for Contract<C, H>
    where
        C: Constructor,
        H: Handlers<C::Output>,
    {
        type Output = H::Output;

        fn handle_message(self, _state: &mut C::Output, _input: Request) -> Self::Output {
            unimplemented!()
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
        fn write_state(_state: &S) {
        }
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
        type Output;

        fn handle(self, env: &EthEnv, state: State, name: &str, msg_data: &[u8]) -> Result<()>;
    }

    use either::Either;

    macro_rules! impl_handlers {
        ($statename:ident, $($any:tt)*) => {
    impl<M, Rest, $statename, SW> Handlers<$statename> for ((PhantomData<M>, MessageHandler<for<'a> fn(&'a EthEnv, &'a $($any)*, M::Input) -> M::Output, SW>), Rest)
    where
        M: Message,
        Rest: Handlers<$statename>,
        SW: StateWriter<$statename>,
    {
        type Output = Either<<M as Message>::Output, <Rest as Handlers<$statename>>::Output>;

        // TODO: Pre-hash?
        fn handle(self, env: &EthEnv, mut state: $statename, name: &str, msg_data: &[u8]) -> Result<()> {
            fn deserialize<In, Out>(_: In) -> Out {
                unimplemented!()
            }

            if M::NAME == name {
                let head = self.0;
                (head.1.handler)(env, &mut state, deserialize(msg_data));
                SW::write_state(&state);
                Ok(())
            } else {
                self.1.handle(env, state, name, msg_data)
            }
        }
    }
        }
    }

    impl_handlers!(State, State);
    impl_handlers!(State, mut State);

    impl<State> Handlers<State> for () {
        type Output = !;

        fn handle(self, _env: &EthEnv, _state: State, _name: &str, _msg_data: &[u8]) -> Result<()> {
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

        fn selector() -> u32 {
            unimplemented!()
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

    pub struct TxData(());

    #[cfg(feature = "nightly")]
    pub trait RespondsTo<T> {}

    #[cfg(feature = "nightly")]
    impl<C, Msg, Handler, Rest> RespondsTo<Msg> for Contract<C, ((PhantomData<Msg>, Handler), Rest)> {}
    #[cfg(feature = "nightly")]
    impl<C, Msg, Head, Rest> RespondsTo<Msg> for Contract<C, (Head, Rest)>
    where
        Contract<C, Rest>: RespondsTo<Msg>,
    {
    }

    // This is essentially a hack to get around the fact that `FnOnce::Output` is
    // unstable
    pub trait Constructor {
        type Output;
    }

    impl<Out> Constructor for fn(TxInfo) -> Out {
        type Output = Out;
    }

    pub struct Contract<Constructor, Handle> {
        constructor: Constructor,
        handlers: Handle,
    }

    impl Contract<(), ()> {
        pub fn new() -> Self {
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

        pub fn constructor<Out>(
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
        // We can't make this return an `impl Trait`-style result because it would require
        // HKT.

        fn with_handler<M, H, SW>(
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

        pub fn on_msg<M>(
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

        pub fn on_msg_mut<M>(
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
    use pwasm::{self, Contract, ContractDef};

    messages! {
        Add(u32);
        Get() -> u32;
    }

    pub struct State {
        current: u32,
        calls_to_add: usize,
    }

    pub fn contract() -> impl ContractDef<State> {
        let _deploy_info: ::pwasm::DeployData = pwasm::deploy_data();

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

        use ::pwasm::Request;

        messages! {
            foo(u32, u64, u16);
        }

        assert_eq!(Request::new::<foo>((0, 1, 2)).function_selector, [0; 4]);
    }
}
