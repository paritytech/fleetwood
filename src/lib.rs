#![cfg_attr(not(feature = "std"), no_std)]
#![allow(dead_code)]

#[macro_use]
extern crate lazy_static;

extern crate bigint;
extern crate bincode;
extern crate core;
extern crate either;
extern crate parity_hash;
extern crate pwasm_abi;
extern crate serde;
extern crate tiny_keccak;

pub mod shim {
    use parity_hash::H256;
    use std::collections::HashMap;
    use std::sync::Mutex;

    lazy_static! {
        static ref STORAGE_MUTEX: Mutex<HashMap<[u8; 32], [u8; 32]>> = Mutex::new(HashMap::new());
    }

    pub fn write(key: &H256, val: &[u8; 32]) {
        STORAGE_MUTEX
            .lock()
            .unwrap()
            .insert(key.clone().0, val.clone());
    }

    pub fn read(key: &H256) -> [u8; 32] {
        STORAGE_MUTEX
            .lock()
            .unwrap()
            .get(&key.0)
            .cloned()
            .unwrap_or([0; 32])
    }
}

use shim as pwasm_ethereum;

mod pwasm {
    use bigint::U256;
    use core::hash::Hash;
    use core::iter;
    use core::marker::PhantomData;
    use core::result::Result as StdResult;
    use parity_hash::H256;
    use pwasm_abi::eth::{Sink, Stream};
    use std::io;

    use pwasm_ethereum;

    use serde::{Deserialize, Serialize};

    fn increment(hash: &mut H256) {
        let mut overflow = true;

        loop {
            for i in hash[..].iter_mut() {
                if overflow {
                    let (val, new_overflow) = i.overflowing_add(1);
                    *i = val;
                    overflow = new_overflow;
                } else {
                    return;
                }
            }
        }
    }

    pub mod marker {
        pub trait BorrowMarker {
            type Store: Default;

            fn mark_changed(_store: &Self::Store) {}
            fn is_changed(_store: &Self::Store) -> bool {
                false
            }
        }

        pub trait BorrowMut: BorrowMarker {}

        pub struct Mut;
        pub struct Immut;

        impl BorrowMut for Mut {}

        impl BorrowMarker for Immut {
            type Store = ();
        }

        impl BorrowMarker for Mut {
            type Store = ::std::cell::Cell<bool>;

            fn mark_changed(store: &Self::Store) {
                store.set(true);
            }

            fn is_changed(store: &Self::Store) -> bool {
                store.get()
            }
        }
    }

    use self::marker::{BorrowMarker, BorrowMut};

    pub struct Field<'a, Marker, T: 'a>
    where
        Marker: BorrowMarker,
        T: ::serde::Serialize,
    {
        key: H256,
        _marker: PhantomData<(Marker, &'a T)>,
        current: ::std::cell::UnsafeCell<Option<T>>,
        changes: Marker::Store,
    }

    impl<'a, M, T> Field<'a, M, T>
    where
        M: BorrowMarker,
        T: for<'any> ::serde::Deserialize<'any> + ::serde::Serialize,
    {
        pub fn new(name: &'static str) -> Self {
            // TODO: we want to use a cheaper hashing algorithm for this.
            let mut keccak = ::tiny_keccak::Keccak::new_sha3_256();

            keccak.update(name.as_bytes());

            let mut out = [0u8; 32];
            keccak.finalize(&mut out);
            Field {
                key: H256(out),
                current: ::std::cell::UnsafeCell::new(None),
                changes: Default::default(),
                _marker: Default::default(),
            }
        }

        unsafe fn populate(&self) {
            if (*self.current.get()).is_none() {
                *self.current.get() = Some(
                    ::bincode::deserialize_from::<_, T>(EthStateReader::new(self.key.clone()))
                        .expect("Couldn't deserialize"),
                );
            }
        }
    }

    impl<'a, T> Field<'a, self::marker::Immut, T>
    where
        T: for<'any> ::serde::Deserialize<'any> + ::serde::Serialize,
    {
        pub fn get(name: &'static str) -> Self {
            Self::new(name)
        }
    }

    impl<'a, T: 'a> Field<'a, self::marker::Mut, T>
    where
        T: for<'any> ::serde::Deserialize<'any> + ::serde::Serialize,
    {
        pub fn get_mut(name: &'static str) -> Self {
            Self::new(name)
        }

        pub fn set(name: &'static str, val: T) {
            Self::get_mut(name).insert(val)
        }
    }

    impl<'a, M, T> Field<'a, M, T>
    where
        M: BorrowMut + BorrowMarker,
        T: for<'any> ::serde::Deserialize<'any> + ::serde::Serialize,
    {
        pub fn insert(&self, val: T) {
            M::mark_changed(&self.changes);
            unsafe { *self.current.get() = Some(val) };
        }
    }

    impl<'a, M, T> ::std::ops::Drop for Field<'a, M, T>
    where
        M: BorrowMarker,
        T: ::serde::Serialize,
    {
        fn drop(&mut self) {
            if M::is_changed(&self.changes) {
                use std::io::Write;
                let mut writer =
                    EthStateWriter::new(::std::mem::replace(&mut self.key, Default::default()));
                ::bincode::serialize_into(
                    &mut writer,
                    unsafe { &*self.current.get() }.as_ref().unwrap(),
                ).unwrap();
                writer.flush().unwrap();
            }
        }
    }

    impl<'a, M, T> ::std::ops::Deref for Field<'a, M, T>
    where
        M: BorrowMarker,
        T: for<'any> ::serde::Deserialize<'any> + ::serde::Serialize,
    {
        type Target = T;

        fn deref(&self) -> &T {
            unsafe {
                self.populate();

                Option::as_ref(&mut *self.current.get()).unwrap()
            }
        }
    }

    impl<'a, M, T> ::std::ops::DerefMut for Field<'a, M, T>
    where
        M: BorrowMut + BorrowMarker,
        T: for<'any> ::serde::Deserialize<'any> + ::serde::Serialize,
    {
        fn deref_mut(&mut self) -> &mut T {
            unsafe {
                self.populate();

                M::mark_changed(&self.changes);

                Option::as_mut(&mut *self.current.get()).unwrap()
            }
        }
    }

    struct EthStateReader {
        key: H256,
        val: [u8; 32],
        index: usize,
    }

    impl EthStateReader {
        fn new(key: H256) -> Self {
            EthStateReader {
                key,
                val: pwasm_ethereum::read(&key),
                index: 0,
            }
        }
    }

    impl io::Read for EthStateReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let len = buf.len();
            let mut index = 0;

            while !buf[index..].is_empty() {
                let buf = &mut buf[index..];

                let to_consume = (self.val.len() - self.index).min(buf.len());
                buf[..to_consume].copy_from_slice(&self.val[self.index..self.index + to_consume]);

                self.index += to_consume;
                index += to_consume;

                if self.index == self.val.len() {
                    increment(&mut self.key);
                    self.val = pwasm_ethereum::read(&self.key);
                    self.index = 0;
                }
            }

            Ok(len)
        }
    }

    struct EthStateWriter {
        key: H256,
        val: [u8; 32],
        index: usize,
    }

    impl EthStateWriter {
        fn new(key: H256) -> Self {
            EthStateWriter {
                key,
                val: [0; 32],
                index: 0,
            }
        }
    }

    impl io::Write for EthStateWriter {
        fn write(&mut self, mut val: &[u8]) -> io::Result<usize> {
            let len = val.len();
            while !val.is_empty() {
                if self.index == self.val.len() {
                    self.flush()?;
                    increment(&mut self.key);
                    self.index = 0;
                }

                let consumed = (self.val.len() - self.index).min(val.len());

                self.val[self.index..self.index + consumed].copy_from_slice(&val[..consumed]);

                self.index += consumed;

                val = &val[consumed..];
            }

            Ok(len)
        }

        fn flush(&mut self) -> io::Result<()> {
            pwasm_ethereum::write(&self.key, &self.val);

            Ok(())
        }
    }

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

    #[derive(Debug)]
    pub enum Error {
        NoMethodError,
        DecodeError(::pwasm_abi::eth::Error),
    }

    impl From<::pwasm_abi::eth::Error> for Error {
        fn from(other: ::pwasm_abi::eth::Error) -> Self {
            Error::DecodeError(other)
        }
    }

    pub type Result<T> = StdResult<T, Error>;

    pub struct Request {
        selector: [u8; 4],
        input: Vec<u8>,
    }

    impl Request {
        pub fn new(input: Vec<u8>) -> Option<Self> {
            if input.len() >= 4 {
                Some(Request {
                    selector: [input[0], input[1], input[2], input[3]],
                    input,
                })
            } else {
                None
            }
        }

        pub fn to_stream(&self) -> Stream {
            Stream::new(&self.input[4..])
        }

        pub fn deserialize_input_for<M: Message>(
            &self,
        ) -> StdResult<M::Input, ::pwasm_abi::eth::Error> {
            DecodeSolidityArgs::pop(&mut self.to_stream())
        }

        pub fn function_selector(&self) -> [u8; 4] {
            [self.input[0], self.input[1], self.input[2], self.input[3]]
        }

        // For testing
        pub fn serialize<M: Message>(input: M::Input) -> Self
        where
            M::Input: EncodeSolidityArgs,
        {
            let sel = M::selector();
            let mut out = sel.to_vec();
            let mut sink = Sink::new(input.count());
            input.push(&mut sink);
            sink.drain_to(&mut out);

            Request {
                selector: sel,
                input: out,
            }
        }
    }

    fn make_selector<'a, I: IntoIterator<Item = &'a str>>(iter: I) -> [u8; 4] {
        let mut keccak = ::tiny_keccak::Keccak::new_keccak256();

        for element in iter {
            keccak.update(element.as_bytes());
        }

        let mut out = [0u8; 4];
        keccak.finalize(&mut out);
        out
    }

    pub trait DecodeSolidityArgs: Sized {
        fn pop(stream: &mut Stream) -> StdResult<Self, ::pwasm_abi::eth::Error>;
    }

    pub trait EncodeSolidityArgs: Sized {
        fn push(self, stream: &mut Sink);
        fn count(&self) -> usize;
    }

    macro_rules! impl_solidityargs_abitype {
        ($($typ:ty,)*) => {
            $(
                impl DecodeSolidityArgs for $typ {
                    fn pop(stream: &mut Stream) -> StdResult<Self, ::pwasm_abi::eth::Error> {
                        stream.pop()
                    }
                }

                impl DecodeSolidityArgs for Vec<$typ> {
                    fn pop(stream: &mut Stream) -> StdResult<Self, ::pwasm_abi::eth::Error> {
                        stream.pop()
                    }
                }

                impl EncodeSolidityArgs for $typ {
                    fn push(self, sink: &mut Sink) {
                        sink.push(self)
                    }

                    fn count(&self) -> usize {
                        1
                    }
                }

                impl EncodeSolidityArgs for Vec<$typ> {
                    fn push(self, sink: &mut Sink) {
                        sink.push(self)
                    }

                    fn count(&self) -> usize {
                        self.len()
                    }
                }
            )*
        };
    }

    impl_solidityargs_abitype! {
        Vec<u8>,
        U256,
        ::parity_hash::Address,
        ::parity_hash::H256,
        u32,
        u64,
        i32,
        i64,
        bool,
        [u8; 1],
        [u8; 2],
        [u8; 3],
        [u8; 4],
        [u8; 5],
        [u8; 6],
        [u8; 7],
        [u8; 8],
        [u8; 9],
        [u8; 10],
        [u8; 11],
        [u8; 12],
        [u8; 13],
        [u8; 14],
        [u8; 15],
        [u8; 16],
        [u8; 17],
        [u8; 18],
        [u8; 19],
        [u8; 20],
        [u8; 21],
        [u8; 22],
        [u8; 23],
        [u8; 24],
        [u8; 25],
        [u8; 26],
        [u8; 27],
        [u8; 28],
        [u8; 29],
        [u8; 30],
        [u8; 31],
        [u8; 32],
    }

    macro_rules! impl_solidityargs_tup {
        ($first:ident $(, $rest:ident)*) => {
            impl<$first, $($rest,)*> DecodeSolidityArgs for ($first, $($rest,)*)
            where
                $first: DecodeSolidityArgs,
                $($rest : DecodeSolidityArgs,)*
            {
                #[allow(non_snake_case)]
                fn pop(stream: &mut Stream) -> StdResult<Self, ::pwasm_abi::eth::Error> {
                    let $first: $first = <$first as DecodeSolidityArgs>::pop(stream)?;
                    $(
                        let $rest: $rest = <$rest as DecodeSolidityArgs>::pop(stream)?;
                    )*
                    Ok((
                        $first,
                        $($rest),*
                    ))
                }
            }

            impl<$first, $($rest,)*> EncodeSolidityArgs for ($first, $($rest,)*)
            where
                $first: EncodeSolidityArgs,
                $($rest : EncodeSolidityArgs,)*
            {
                #[allow(non_snake_case)]
                fn push(self, sink: &mut Sink) {
                    let (
                        $first,
                        $($rest,)*
                    ) = self;
                    $first.push(sink);
                    $($rest.push(sink);)*
                }

                #[allow(non_snake_case)]
                fn count(&self) -> usize {
                    let $first = 1;
                    $(
                        let $rest = 1;
                    )*

                    $first $(+ $rest)*
                }
            }

            impl_solidityargs_tup!($($rest),*);
        };
        () => {
            impl DecodeSolidityArgs for () {
                fn pop(_: &mut Stream) -> StdResult<Self, ::pwasm_abi::eth::Error> {
                    Ok(())
                }
            }

            impl EncodeSolidityArgs for () {
                fn push(self, _: &mut Sink) { }
                fn count(&self) -> usize {
                    0
                }
            }
        };
    }

    impl_solidityargs_tup!(A, B, C, D, E, F, G, H, I, J, K);

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
            use std::any::TypeId;
            use std::{mem, ptr};

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

    impl Response for Void {
        fn output_for<M: Message>(self) -> Option<M::Output>
        where
            M::Output: Any,
        {
            None
        }
    }

    // TODO: Should we build on deployment or should there be a "deploy" step? Building on deployment is _way_
    //       simpler.
    pub fn deploy_data() -> DeployData {
        DeployData {
            deployer: U256::from(0),
        }
    }

    pub struct ContractInstance<'a, S, T: 'a> {
        pub env: &'a EthEnv,
        pub state: S,
        contract: &'a T,
    }

    impl<'a, S, T> ContractInstance<'a, S, T>
    where
        S: Default,
        T: ContractDef<State = S>,
        T::Output: Response,
    {
        pub fn call<M: Message>(&mut self, input: M::Input) -> M::Output
        where
            // TODO
            M::Output: 'static,
            M::Input: DecodeSolidityArgs + EncodeSolidityArgs,
        {
            Response::output_for::<M>(self.contract.send_request(
                self.env,
                &mut self.state,
                Request::serialize::<M>(input),
            )).expect("Didn't respond to message")
        }
    }

    pub trait ContractDef {
        type Input;
        type State: Default;
        type Output: Response + 'static;

        // We have this function to allow easy testing for users. For a lot of functions they don't need
        // to deploy to the blockchain at all.
        fn send_request(
            &self,
            _env: &EthEnv,
            state: &mut Self::State,
            input: Request,
        ) -> Self::Output;

        fn construct(&self, state: &mut Self::State, input: Self::Input);

        fn deploy<'a>(
            &'a self,
            env: &'a EthEnv,
            input: Self::Input,
        ) -> ContractInstance<'a, Self::State, Self>
        where
            Self: Sized,
        {
            let mut state = Default::default();
            self.construct(&mut state, input);
            ContractInstance {
                env,
                state,
                contract: self,
            }
        }

        fn call<M: Message>(
            &self,
            env: &EthEnv,
            state: &mut Self::State,
            input: M::Input,
        ) -> Option<M::Output>
        where
            Self::Output: Response,
            Self: Sized,
            M::Output: 'static,
            M::Input: EncodeSolidityArgs,
        {
            Response::output_for::<M>(self.send_request(env, state, Request::serialize::<M>(input)))
        }
    }

    impl<C, H> ContractDef for Contract<C, H>
    where
        C: Constructor,
        H: Handlers<C::State>,
        C::State: Default,
    {
        type State = C::State;
        type Input = C::Input;
        type Output = H::Output;

        fn construct(&self, state: &mut C::State, txdata: C::Input) {
            self.constructor.call(state, txdata)
        }

        fn send_request(&self, env: &EthEnv, state: &mut C::State, input: Request) -> Self::Output {
            self.handlers.handle(env, state, input).expect("No method")
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
            impl<M, Rest, $statename> Handlers<$statename> for (
                (
                    PhantomData<M>,
                    for<'a> fn(&'a EthEnv, &'a $($any)*, M::Input) -> M::Output,
                ),
                Rest
            )
            where
                M: Message,
                <M as Message>::Input: for<'a> Deserialize<'a>,
                <M as Message>::Output: 'static,
                Rest: Handlers<$statename>,
            {
                type Output = Either<<M as Message>::Output, <Rest as Handlers<$statename>>::Output>;

                // TODO: Pre-hash?
                fn handle(&self, env: &EthEnv, state: &mut $statename, request: Request) -> Result<Self::Output> {
                    if M::selector() == request.function_selector() {
                        let head = self.0;
                        let out = (head.1)(env, state, request.deserialize_input_for::<M>()?);
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

    pub enum Void {}

    impl<State> Handlers<State> for () {
        type Output = Void;

        fn handle(
            &self,
            _env: &EthEnv,
            _state: &mut State,
            _request: Request,
        ) -> Result<Self::Output> {
            Err(Error::NoMethodError)
        }
    }

    pub trait SolidityTypeNames {
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
    impl_soltype!(U256, "uint256");
    impl_soltype!(::parity_hash::Address, "address");

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

    impl<T> SolidityTypeNames for T
    where
        T: SolidityType,
    {
        type Iter = iter::Chain<
            iter::Chain<iter::Once<&'static str>, <T::Iter as IntoIterator>::IntoIter>,
            iter::Once<&'static str>,
        >;

        #[inline]
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
            impl<$($name),*> SolidityTypeNames for ($($name,)*)
            where
            $(
                $name : SolidityType,
            )*
            {
                type Iter = tup_sig!(@chain_type $($name)*);

                #[inline]
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

    // Any more than this and it becomes extremely slow to compile
    tup_sigs!(A B C D E F G H I J K L M N O P Q);

    pub trait Message {
        type Input: DecodeSolidityArgs + SolidityTypeNames;
        type Output: EncodeSolidityArgs;

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
        T::Input: SolidityTypeNames,
    {
        type Iter = iter::Chain<
            iter::Once<&'static str>,
            <<T::Input as SolidityTypeNames>::Iter as IntoIterator>::IntoIter,
        >;

        fn signature() -> Self::Iter {
            iter::once(Self::NAME).chain(<Self as Message>::Input::arg_sig().into_iter())
        }
    }

    // This is essentially a hack to get around the fact that `FnOnce`'s internals are
    // unstable
    pub trait Constructor {
        type State;
        type Input;

        fn call(&self, state: &mut Self::State, txinfo: Self::Input);
    }

    impl<State, I> Constructor for fn(&mut State, I) {
        type State = State;
        type Input = I;

        fn call(&self, state: &mut Self::State, txinfo: I) {
            self(state, txinfo)
        }
    }

    impl<State> Constructor for fn(&mut State) {
        type State = State;
        type Input = ();

        fn call(&self, state: &mut Self::State, _: ()) {
            self(state)
        }
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

        #[inline(always)]
        pub fn constructor<State, Input>(
            self,
            constructor: fn(&mut State, Input),
        ) -> Contract<fn(&mut State, Input), ()>
        where
            State: Default,
        {
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

    impl<Cons, Handle> Contract<Cons, Handle>
    where
        Cons: Constructor + Copy,
        Handle: Handlers<Cons::State> + Copy,
    {
        #[inline(always)]
        fn with_handler<M, H>(self, handler: H) -> Contract<Cons, ((PhantomData<M>, H), Handle)> {
            Contract {
                constructor: self.constructor,
                handlers: ((PhantomData, handler), self.handlers),
            }
        }

        #[inline(always)]
        pub fn on_msg<M>(
            self,
            handler: Handler<M, Cons::State>,
        ) -> Contract<Cons, ((PhantomData<M>, Handler<M, Cons::State>), Handle)>
        where
            M: Message,
        {
            self.with_handler(handler)
        }

        #[inline(always)]
        pub fn on_msg_mut<M>(
            self,
            handler: HandlerMut<M, Cons::State>,
        ) -> Contract<Cons, ((PhantomData<M>, HandlerMut<M, Cons::State>), Handle)>
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

}

macro_rules! state {
    (
        struct $name:ident {
            $(
                $field:ident : $typ:ty
            ),*
            $(,)*
        }
    ) => {
        struct $name {
            __inner: ::std::marker::PhantomData<$name>,
        }

        state! {
            @impl $name {
                $(
                    $field : $typ
                ),*
            }
        }
    };
    (
        pub struct $name:ident {
            $(
                $field:ident : $typ:ty
            ),*
            $(,)*
        }
    ) => {
        pub struct $name {
            __inner: ::std::marker::PhantomData<$name>,
        }

        state! {
            @impl $name {
                $(
                    $field : $typ
                ),*
            }
        }
    };
    (
        @impl $name:ident {
            $(
                $field:ident : $typ:ty
            ),*
            $(,)*
        }
    ) => {
        impl Default for $name {
            fn default() -> Self {
                $name {
                    __inner: ::std::marker::PhantomData,
                }
            }
        }

        pub trait __State<'a>: Sized {
            type Marker: $crate::pwasm::marker::BorrowMarker;
            $(
                fn $field(self) -> $crate::pwasm::Field<'a, Self::Marker, $typ> {
                    $crate::pwasm::Field::new(stringify!($field))
                }
            )*
        }

        impl<'a> __State<'a> for &'a mut $name {
            type Marker = $crate::pwasm::marker::Mut;
        }

        impl<'a> __State<'a> for &'a $name {
            type Marker = $crate::pwasm::marker::Immut;
        }
    };
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

    state! {
        pub struct State {
            current: u32,
            calls_to_add: usize,
        }
    }

    pub fn contract() -> impl ContractDef {
        Contract::new()
            .constructor(|state: &mut State, ()| {
                state.current().insert(1);
                state.calls_to_add().insert(0);
            })
            .on_msg_mut::<Add>(|_env, state, to_add| {
                *state.calls_to_add() += 1;
                *state.current() += to_add;
            })
            .on_msg::<Get>(|_env, state, ()| *state.current())
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn tuple_sigs() {
        use pwasm::SolidityTypeNames;

        assert_eq!(
            <(u8, i32, u32)>::arg_sig().collect::<String>(),
            "(uint8,int32,uint32)"
        );
    }

    #[test]
    fn message_sigs() {
        use pwasm::MessageExt;

        messages! {
            Foo(u32, u64, i32);
            UseArray(Vec<u32>, Vec<bool>);
            Get() -> u32;
            OneArg(u64);
        }

        assert_eq!(
            Foo::signature().into_iter().collect::<String>(),
            "Foo(uint32,uint64,int32)"
        );
        assert_eq!(
            UseArray::signature().into_iter().collect::<String>(),
            "UseArray(uint32[],bool[])"
        );
        assert_eq!(Get::signature().into_iter().collect::<String>(), "Get()");
        assert_eq!(
            OneArg::signature().into_iter().collect::<String>(),
            "OneArg(uint64)"
        );
    }

    #[test]
    fn real_world_message_sigs() {
        #![allow(non_camel_case_types)]

        use ::pwasm::MessageExt;

        fn to_byte_array(i: u32) -> [u8; 4] {
            [
                (i >> 24) as u8,
                (i >> 16) as u8,
                (i >> 8) as u8,
                i as u8,
            ]
        }

        use bigint::U256;
        use parity_hash::Address;

        messages! {
            totalSupply() -> U256;
            balanceOf(Address) -> U256;
            transfer(Address, U256) -> bool;
        }

        // These are from the expanded form of the example ERC20 token contract
        // from step 5 of the pwasm tutorial
        assert_eq!(totalSupply::selector(), to_byte_array(404098525u32));
        assert_eq!(balanceOf::selector(), to_byte_array(1889567281u32));
        assert_eq!(transfer::selector(), to_byte_array(2835717307u32));
    }

    #[test]
    fn request() {
        #![allow(non_camel_case_types)]

        use pwasm::Request;

        messages! {
            foo(u32, u64, i32);
        }

        let _request = Request::serialize::<foo>((0, 1, 2));
    }

    #[test]
    fn contract() {
        use pwasm::{Contract, ContractDef, EthEnv};

        messages! {
            Add(u32);
            Get() -> u32;
            GetNoMacro() -> u32;
            AssertVec();
            Unused();
        }

        state! {
            struct State {
                current: u32,
                calls_to_add: usize,
                vec: Vec<usize>,
            }
        }

        // TODO: Probably you won't be able to create a new instance of the "proper"
        //       `EthEnv`, only a dummy version.
        let env = EthEnv::new();

        let definition = Contract::new()
            .constructor(|state: &mut State, _txdata| {
                state.current().insert(1);
                state.calls_to_add().insert(0);
                state.vec().insert((0..1024usize).collect::<Vec<_>>());
            })
            .on_msg_mut::<Add>(|_env, state, to_add| {
                *state.calls_to_add() += 1;
                *state.current() += to_add;
            })
            .on_msg::<Get>(|_env, state, ()| *state.current())
            .on_msg::<GetNoMacro>(|_env, _state, ()| {
                use pwasm::Field;

                static CURRENT: &str = "current";

                let val: u32 = *Field::get(CURRENT);
                // No safety rails here, you can modify the state in an "immutable"
                // message handler. This will fail if you were called with `call_static`
                // however
                Field::set(CURRENT, 10u32);
                let out = *Field::get(CURRENT);
                Field::set(CURRENT, val);

                out
            })
            .on_msg::<AssertVec>(|_env, state, ()| {
                assert_eq!(*state.vec(), (0..1024usize).collect::<Vec<_>>());
            });

        let mut contract = definition.deploy(&env, ());

        let () = contract.call::<Add>(1);
        let val: u32 = contract.call::<Get>(());
        let val_no_macro: u32 = contract.call::<GetNoMacro>(());

        contract.call::<AssertVec>(());

        assert_eq!(val_no_macro, 10);
        assert_eq!(val, 2);
        assert_eq!(*contract.state.current(), 2,);
        assert_eq!(*contract.state.calls_to_add(), 1,);
    }
}
