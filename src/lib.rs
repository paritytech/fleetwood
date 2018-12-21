//! # Fleetwood - a portable WebAssembly smart contract library
//!
//! So the boss is breathing down your neck and you need to get a contract on a WebAssembly
//! chain in the next 30 minutes? Well fret not, for I have an obscure and niche solution to
//! your obscure and niche problem.
//!
//! ## What is Fleetwood?
//!
//! Fleetwood is an [eDSL][edsl] for creating smart contracts in Rust. It was created as a
//! response to the many languages for creating smart contracts that exist in the Ethereum
//! ecosystem. Specifically, it attempts to fix the problems with custom languages (i.e.
//! lack of tooling and a necessity to reinvent the language-design wheel) by embedding it
//! as a library in Rust. You can see more of the thinking in [my first article explaining
//! my thinking][brighter-future] and [my second article selling the concept to Solidity
//! developers][why-write-in-rust].
//!
//! [edsl]: https://wiki.haskell.org/Embedded_domain_specific_language
//! [brighter-future]: http://troubles.md/posts/rust-smart-contracts/
//! [why-write-in-rust]: http://troubles.md/posts/why-write-smart-contracts-in-rust/

#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate serde_derive;

extern crate bincode;
extern crate either;
extern crate metrohash;
extern crate pwasm_std;
extern crate pwasm_abi;
extern crate serde;
extern crate tiny_keccak;

use environment::{Key, Value};
use metrohash::MetroHash;
use pwasm_std::types::{Address, H256};
use pwasm_abi::{
    eth::{Sink, Stream},
    types::U256,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::cell::{Cell, UnsafeCell};
use std::hash::{BuildHasher, Hash};
use std::io::{self, Cursor};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::result::Result as StdResult;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{fmt, iter};

mod dummy;
pub mod environment;

#[inline]
fn increment(hash: &mut [u8]) {
    let mut overflow = true;

    loop {
        for i in hash.iter_mut() {
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

pub enum Void {}

pub trait Wrapper<T> {}

pub trait FieldName {
    fn name(&self) -> &'static str;
}

pub struct RuntimeFieldName(&'static str);

impl FieldName for RuntimeFieldName {
    #[inline]
    fn name(&self) -> &'static str {
        self.0
    }
}

pub struct Field<Env: environment::HasStorage, T, N: FieldName = RuntimeFieldName> {
    key: Env::Key,
    _marker: PhantomData<(N, T)>,
    current: UnsafeCell<Option<T>>,
    changes: Cell<bool>,
}

impl<Env: environment::HasStorage, T, N: FieldName> Field<Env, T, N>
where
    T: for<'any> ::serde::Deserialize<'any> + ::serde::Serialize,
{
    pub fn set(&mut self, val: T) {
        self.changes.set(true);
        unsafe { *self.current.get() = Some(val) };
    }

    pub fn flush(&mut self) {
        if self.changes.get() {
            let mut v = Default::default();
            let mut writer = StateWriter::<Env>::new(&mut self.key, &mut v);
            ::bincode::serialize_into(
                &mut writer,
                unsafe { &*self.current.get() }.as_ref().unwrap(),
            ).unwrap();
            writer.flush_internal();
        }
    }
}

impl<Env: environment::HasStorage, T, N: FieldName + Default> From<T> for Field<Env, T, N>
where
    T: ::serde::Serialize,
    Env::Key: Key,
{
    fn from(from: T) -> Self {
        let name = N::default();
        Field {
            key: Key::from_u64(quickhash(name.name().as_bytes())),
            current: UnsafeCell::new(Some(from)),
            changes: Cell::new(true),
            _marker: Default::default(),
        }
    }
}

impl<Env: environment::HasStorage, T, N: FieldName + Default> Default for Field<Env, T, N>
where
    T: ::serde::Serialize,
    Env::Key: Key,
{
    fn default() -> Self {
        let name = N::default();
        Field {
            key: Key::from_u64(quickhash(name.name().as_bytes())),
            current: UnsafeCell::new(None),
            changes: Default::default(),
            _marker: Default::default(),
        }
    }
}

fn quickhash(name: &[u8]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    // TODO: we want to use a cheaper hashing algorithm for this.
    let mut hasher = DefaultHasher::new();

    name.hash(&mut hasher);

    hasher.finish()
}

impl<Env: environment::HasStorage, T> Field<Env, T>
where
    T: for<'any> ::serde::Deserialize<'any> + ::serde::Serialize,
    Env::Key: Key,
{
    pub fn new(name: &'static str) -> Self {
        Field {
            key: Key::from_u64(quickhash(name.as_bytes())),
            current: UnsafeCell::new(None),
            changes: Default::default(),
            _marker: Default::default(),
        }
    }

    pub fn read(name: &'static str) -> Self {
        Self::new(name)
    }

    pub fn write(name: &'static str, val: T) {
        Self::new(name).set(val)
    }
}

impl<Env: environment::HasStorage, T, N: FieldName> Field<Env, T, N>
where
    T: for<'any> ::serde::Deserialize<'any> + ::serde::Serialize,
{
    unsafe fn populate(&self) {
        if (*self.current.get()).is_none() {
            *self.current.get() = Some(
                // TODO: This relies on (de)serialize impls being correct and the only
                //       method to store data being via `Field`. We should have a better
                //       way to do this.
                ::bincode::deserialize_from::<_, T>(StateReader::<Env>::new(self.key.clone()))
                    .expect("Couldn't deserialize"),
            );
        }
    }
}

impl<Env: environment::HasStorage, T, N: FieldName> Deref for Field<Env, T, N>
where
    T: for<'any> ::serde::Deserialize<'any> + ::serde::Serialize,
{
    type Target = T;

    fn deref(&self) -> &T {
        unsafe {
            self.populate();

            Option::as_ref(&*self.current.get()).unwrap()
        }
    }
}

impl<Env: environment::HasStorage, T, N: FieldName> DerefMut for Field<Env, T, N>
where
    T: for<'any> ::serde::Deserialize<'any> + ::serde::Serialize,
{
    fn deref_mut(&mut self) -> &mut T {
        unsafe {
            self.populate();

            self.changes.set(true);

            Option::as_mut(&mut *self.current.get()).unwrap()
        }
    }
}

struct StateReader<Env: environment::HasStorage> {
    key: Env::Key,
    val: Cursor<Env::Value>,
}

impl<Env: environment::HasStorage> StateReader<Env> {
    fn new(key: Env::Key) -> Self {
        let val = Cursor::new(Env::read(&key));
        StateReader { key, val }
    }
}

impl<Env: environment::HasStorage> io::Read for StateReader<Env> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let len = buf.len();
        let mut index = 0;

        while !buf[index..].is_empty() {
            if self.val.position() == self.val.get_ref().as_ref().len() as _ {
                increment(self.key.as_mut());
                self.val = Cursor::new(Env::read(&self.key));
            }

            let consumed = self.val.read(&mut buf[index..])?;

            index += consumed;
        }

        Ok(len)
    }
}

struct StateWriter<'a, Env: environment::HasStorage>
where
    Env::Key: 'a,
    Env::Value: 'a,
{
    key: &'a mut Env::Key,
    val: &'a mut Env::Value,
    index: usize,
}

impl<'a, Env: environment::HasStorage> StateWriter<'a, Env> {
    fn new(key: &'a mut Env::Key, scratch: &'a mut Env::Value) -> Self {
        StateWriter {
            key,
            val: scratch,
            index: 0,
        }
    }

    fn flush_internal(&self) {
        Env::write(self.key, self.val);
    }
}

impl<'a, Env: environment::HasStorage> io::Write for StateWriter<'a, Env> {
    #[inline]
    fn write(&mut self, val: &[u8]) -> io::Result<usize> {
        fn write_internal<'a, Env: environment::HasStorage>(
            this: &mut StateWriter<'a, Env>,
            val: &[u8],
        ) -> io::Result<()> {
            if val.is_empty() {
                return Ok(());
            }

            if Env::Value::is_finished(this.index) {
                this.flush_internal();
                increment(this.key.as_mut());
                this.index = 0;
            }

            let consumed = {
                let mut writeable = this.val.as_writeable();
                writeable.set_position(this.index as u64);

                let consumed = writeable.write(val)?;
                this.index = writeable.position() as usize;

                consumed
            };

            write_internal(this, &val[consumed..])
        }

        write_internal(self, val)?;

        Ok(val.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_internal();
        Ok(())
    }
}

static DATABASE_SEED: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Serialize, Deserialize)]
pub struct IncrementingState(usize);

impl IncrementingState {
    fn new() -> Self {
        IncrementingState(DATABASE_SEED.fetch_add(1, Ordering::Relaxed))
    }
}

#[allow(deprecated)]
impl BuildHasher for IncrementingState {
    type Hasher = MetroHash;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        MetroHash::with_seed(self.0 as _)
    }
}

// Replacement for `HashMap` that doesn't require serializing/deserializing the
// full map every time you attempt to run a handler.
#[derive(Serialize, Deserialize)]
pub struct Database<Env, K, V, S = IncrementingState> {
    builder: S,
    _marker: PhantomData<(fn() -> Env, K, V)>,
}

impl<E, K, V, S> fmt::Debug for Database<E, K, V, S>
where
    S: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("Database")
            .field("builder", &self.builder)
            .finish()
    }
}

impl<Env: environment::HasStorage, K: Hash, V: Serialize + for<'a> Deserialize<'a>>
    Database<Env, K, V>
{
    pub fn new() -> Self {
        Database {
            builder: IncrementingState::new(),
            _marker: PhantomData,
        }
    }

    fn hash(&self, key: &K) -> Env::Key {
        use std::hash::Hasher;

        // TODO: we want to use a cheaper hashing algorithm for this.
        let mut hasher = self.builder.build_hasher();
        key.hash(&mut hasher);

        let hash =hasher.finish();

        println!("{}", hash);

        Env::Key::from_u64(hash)
    }

    pub fn insert(&mut self, key: &K, val: V) {
        let (mut k, mut v) = (self.hash(key), Default::default());
        let mut writer = StateWriter::<Env>::new(&mut k, &mut v);
        ::bincode::serialize_into(&mut writer, &val).unwrap();
        writer.flush_internal();
    }

    pub fn get(&self, key: &K) -> V {
        ::bincode::deserialize_from(StateReader::<Env>::new(self.hash(key)))
            .expect("Couldn't deserialize")
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
    input: Vec<u8>,
}

impl Request {
    pub fn new(input: Vec<u8>) -> Option<Self> {
        if input.len() >= 4 {
            Some(Request { input })
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

        Request { input: out }
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

pub trait SolidityReturnType: EncodeSolidityArgs {}

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

                impl SolidityReturnType for $typ { }

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
    pwasm_std::types::Address,
    pwasm_std::types::H256,
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

        impl SolidityReturnType for () { }
    };
}

impl_solidityargs_tup!(A, B, C, D, E, F, G, H, I, J, K);

pub trait Response {
    fn output_for<M: Message>(self) -> Option<M::Output>
    where
        M::Output: Any;

    fn serialize(self) -> Vec<u8>;
}

impl<Head: Any + EncodeSolidityArgs, Rest> Response for Either<Head, Rest>
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

    fn serialize(self) -> Vec<u8> {
        match self {
            Either::Left(left) => {
                let mut sink = Sink::new(left.count());
                left.push(&mut sink);
                sink.finalize_panicking()
            }
            Either::Right(right) => right.serialize(),
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

    fn serialize(self) -> Vec<u8> {
        unimplemented!()
    }
}

pub struct ContractInstance<'a, Env: 'a, T: ContractDef<Env> + 'a> {
    pub env: &'a Env,
    pub state: T::State,
    contract: &'a T,
}

impl<'a, Env, T> ContractInstance<'a, Env, T>
where
    T: ContractDef<Env>,
    T::State: ContractState,
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

pub trait ContractState {
    /// Get a state with all the data set to nothing. This is _not_ the same as
    /// `Default
    fn empty() -> Self;
    /// Flush the state to the database (assuming this is a list of fields, this just
    /// means recursively flushing each field)
    fn flush(&mut self);
}

pub trait ContractDef<Env> {
    type Input: DecodeSolidityArgs;
    type State: ContractState;
    type Output: Response + 'static;

    fn send_request(&self, _env: &Env, state: &mut Self::State, input: Request) -> Self::Output;

    fn construct(&self, env: &mut Env, input: Self::Input) -> Self::State;

    fn construct_raw(&self, env: &mut Env, input: &[u8]) -> Self::State {
        self.construct(
            env,
            DecodeSolidityArgs::pop(&mut Stream::new(input)).expect("Failed to construct"),
        )
    }

    fn deploy<'a>(&'a self, env: &'a mut Env, input: Self::Input) -> ContractInstance<'a, Env, Self>
    where
        Self: Sized,
    {
        let state = self.construct(env, input);
        ContractInstance {
            env,
            state,
            contract: self,
        }
    }

    fn call<M: Message>(
        &self,
        env: &Env,
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

impl<Env, C, H> ContractDef<Env> for Contract<Env, C, H>
where
    C: Constructor<Env>,
    H: Handlers<Env, C::Output>,
    C::Output: ContractState,
    C::Input: DecodeSolidityArgs,
{
    type Input = C::Input;
    type Output = H::Output;
    type State = C::Output;

    fn construct(&self, env: &mut Env, input: C::Input) -> Self::State {
        self.constructor.call(env, input)
    }

    fn send_request(&self, env: &Env, state: &mut C::Output, input: Request) -> Self::Output {
        self.handlers.handle(env, state, input).expect("No method")
    }
}

pub trait Handlers<Env, State> {
    type Output: Response + 'static;

    fn handle(&self, env: &Env, state: &mut State, request: Request) -> Result<Self::Output>;
}

use either::Either;

macro_rules! impl_handlers {
    ($statename:ident, $($any:tt)*) => {
        impl<M, Rest, Env, $statename> Handlers<Env, $statename> for (
            (
                PhantomData<M>,
                for<'a> fn(&'a Env, &'a $($any)*, M::Input) -> M::Output,
            ),
            Rest
        )
        where
            M: Message,
            <M as Message>::Input: for<'a> Deserialize<'a>,
            <M as Message>::Output: 'static,
            Rest: Handlers<Env, $statename>,
        {
            type Output = Either<<M as Message>::Output, <Rest as Handlers<Env, $statename>>::Output>;

            // TODO: Pre-hash?
            fn handle(&self, env: &Env, state: &mut $statename, request: Request) -> Result<Self::Output> {
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

impl<Env, State> Handlers<Env, State> for () {
    type Output = Void;

    fn handle(&self, _env: &Env, _state: &mut State, _request: Request) -> Result<Self::Output> {
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
impl_soltype!(pwasm_std::types::Address, "address");

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

sol_array!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128, 256, 512, 1024);

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
        <$name::Iter as ::std::iter::IntoIterator>::IntoIter
    };
    (@chain_type_inner $name:ident $($rest:ident)*) => {
        iter::Chain<
            iter::Chain<
                <$name::Iter as ::std::iter::IntoIterator>::IntoIter,
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
    type Output: SolidityReturnType;

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
pub trait Constructor<Env> {
    type Output;
    type Input;

    fn call(&self, env: &mut Env, input: Self::Input) -> Self::Output;
}

impl<Env, Output, I> Constructor<Env> for fn(&mut Env, I) -> Output {
    type Output = Output;
    type Input = I;

    fn call(&self, env: &mut Env, input: I) -> Self::Output {
        self(env, input)
    }
}

impl<Env, Output> Constructor<Env> for fn(&mut Env) -> Output {
    type Output = Output;
    type Input = ();

    fn call(&self, env: &mut Env, _: ()) -> Self::Output {
        self(env)
    }
}

#[derive(Default)]
pub struct Contract<Env, Constructor, Handle> {
    constructor: Constructor,
    handlers: Handle,
    _marker: PhantomData<fn() -> Env>,
}

impl<Env> Contract<Env, (), ()> {
    // We `inline` all of these because we expect them to be called
    // only once per program. We don't `inline(always)` because
    // it's possible that you have multiple contracts.
    #[inline]
    pub fn new() -> Self {
        Contract {
            constructor: (),
            handlers: (),
            _marker: Default::default(),
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
    #[inline]
    pub fn constructor<State, Input>(
        self,
        constructor: fn(&mut Env, Input) -> State,
    ) -> Contract<Env, fn(&mut Env, Input) -> State, ()>
    where
        State: ContractState,
    {
        Contract {
            constructor,
            handlers: self.handlers,
            _marker: Default::default(),
        }
    }
}

type Handler<Env, M, St> =
    for<'a> fn(&'a Env, &'a St, <M as Message>::Input) -> <M as Message>::Output;
type HandlerMut<Env, M, St> =
    for<'a> fn(&'a Env, &'a mut St, <M as Message>::Input) -> <M as Message>::Output;

impl<Env, Cons, Handle> Contract<Env, Cons, Handle>
where
    Cons: Constructor<Env> + Copy,
    Handle: Handlers<Env, Cons::Output> + Copy,
{
    #[inline(always)]
    fn with_handler<M, H>(self, handler: H) -> Contract<Env, Cons, ((PhantomData<M>, H), Handle)> {
        Contract {
            constructor: self.constructor,
            handlers: ((PhantomData, handler), self.handlers),
            _marker: Default::default(),
        }
    }

    #[inline(always)]
    pub fn on_msg<M>(
        self,
        handler: Handler<Env, M, Cons::Output>,
    ) -> Contract<Env, Cons, ((PhantomData<M>, Handler<Env, M, Cons::Output>), Handle)>
    where
        M: Message,
    {
        self.with_handler(handler)
    }

    #[inline(always)]
    pub fn on_msg_mut<M>(
        self,
        handler: HandlerMut<Env, M, Cons::Output>,
    ) -> Contract<Env, Cons, ((PhantomData<M>, HandlerMut<Env, M, Cons::Output>), Handle)>
    where
        M: Message,
    {
        self.with_handler(handler)
    }
}

pub struct DummyEnv(());

impl DummyEnv {
    pub fn new() -> Self {
        DummyEnv(())
    }

    pub fn immediate_caller(&self) -> Address {
        dummy::sender()
    }

    pub fn original_caller(&self) -> Address {
        dummy::origin()
    }

    // We use different types for remote vs local contracts since
    // they require different functions to get the code

    pub fn current_address(&self) -> Address {
        dummy::address()
    }
}

pub trait RemoteContract {}

impl environment::HasStorage for DummyEnv {
    type Key = H256;
    type Value = [u8; 32];

    fn read(key: &Self::Key) -> Self::Value {
        dummy::read(key)
    }

    fn write(key: &Self::Key, value: &Self::Value) {
        dummy::write(key, value)
    }
}

pub trait ExternalContract {
    // Compiles to `CODESIZE` + `CODECOPY` (TODO: This should be dynamically-sized but
    // owned but we can't do that without `alloca`, so we can just write a `Box<[u8]>`-
    // esque type that allocates on the "heap")
    fn code(&self) -> &[u8];
    fn call(&self, method: &[u8], args: &[u8]) -> &[u8];
}

#[macro_export]
macro_rules! state {
    (@fields $name:ident $( $field:ident : $typ:ty, )*) => {
        mod $name {
            $(
                #[allow(non_camel_case_types)]
                #[derive(Default)]
                pub struct $field;

                impl $crate::FieldName for $field {
                    #[inline]
                    fn name(&self) -> &'static str {
                        stringify!($field)
                    }
                }
            )*
        }
    };
    (
        $visibility:vis struct $name:ident {
            $(
                $field:ident : $typ:ty
            ),*
            $(,)*
        }
    ) => {
        $visibility struct $name<Env: $crate::environment::HasStorage> {
            $(
                $field: $crate::Field<Env, $typ, fields::$field>,
            )*
        }

        impl<E: $crate::environment::HasStorage> $crate::ContractState for $name<E> {
            fn empty() -> Self {
                $name {
                    $( $field : $crate::Field::default(), )*
                }
            }

            fn flush(&mut self) {
                $(
                    self.$field.flush();
                )*
            }
        }

        state! {
            @fields fields $( $field : $typ, )*
        }
    };
}

#[macro_export]
macro_rules! messages {
    ($name:ident($($typ:ty),*); $($rest:tt)*) => {
        messages!($name($($typ),*) -> (); $($rest)*);
    };
    ($name:ident($($typ:ty),*) -> $out:ty; $($rest:tt)*) => {
        struct $name;

        impl $crate::Message for $name {
            type Input = ($($typ),*);
            type Output = $out;

            const NAME: &'static str = stringify!($name);
        }

        messages!($($rest)*);
    };
    () => {};
}

#[cfg(all(feature = "std", test))]
mod test {
    #[test]
    fn tuple_sigs() {
        use SolidityTypeNames;

        assert_eq!(
            <(u8, i32, u32)>::arg_sig().collect::<String>(),
            "(uint8,int32,uint32)"
        );
    }

    #[test]
    fn message_sigs() {
        use MessageExt;

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

        use MessageExt;

        fn to_byte_array(i: u32) -> [u8; 4] {
            [(i >> 24) as u8, (i >> 16) as u8, (i >> 8) as u8, i as u8]
        }

        use pwasm_abi::types::U256;
        use pwasm_std::types::Address;

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

        use Request;

        messages! {
            foo(u32, u64, i32);
        }

        let _request = Request::serialize::<foo>((0, 1, 2));
    }

    #[test]
    fn contract() {
        use {Contract, ContractDef, DummyEnv};

        messages! {
            Add(u32);
            Get() -> u32;
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
        //       `DummyEnv`, only a dummy version.
        let mut env = DummyEnv::new();

        let definition = Contract::new()
            .constructor(|_: &mut DummyEnv, _txdata| State::<DummyEnv> {
                current: 1.into(),
                calls_to_add: 0.into(),
                vec: (0..1024usize).collect::<Vec<_>>().into(),
            })
            .on_msg_mut::<Add>(|_env, state, to_add| {
                *state.calls_to_add += 1;
                *state.current += to_add;
            })
            .on_msg::<Get>(|_env, state, ()| *state.current)
            .on_msg::<AssertVec>(|_env, state, ()| {
                assert_eq!(*state.vec, (0..1024usize).collect::<Vec<_>>());
            });

        let mut contract = definition.deploy(&mut env, ());

        let () = contract.call::<Add>(1);
        let val: u32 = contract.call::<Get>(());

        contract.call::<AssertVec>(());

        assert_eq!(val, 2);
        assert_eq!(*contract.state.current, 2);
        assert_eq!(*contract.state.calls_to_add, 1);
    }

    #[test]
    fn erc20() {
        use pwasm_abi::types::U256;
        use pwasm_std::types::Address;
        use {Contract, ContractDef, Database, DummyEnv};

        messages! {
            TotalSupply() -> U256;
            BalanceOf(Address) -> U256;
            Transfer(Address, U256) -> bool;
        }

        state! {
            struct State {
                balances: Database<DummyEnv, Address, U256>,
                total: U256,
            }
        }

        let mut env = DummyEnv::new();

        let definition = Contract::new()
            .constructor(|env: &mut DummyEnv, total_supply: U256| {
                let mut database = Database::new();
                database.insert(&env.original_caller(), total_supply.clone());
                println!("{:?}", database);
                State::<DummyEnv> {
                    total: total_supply.into(),
                    balances: database.into(),
                }
            })
            .on_msg::<BalanceOf>(|_env, state, address| state.balances.get(&address))
            .on_msg::<TotalSupply>(|_env, state, ()| *state.total)
            .on_msg_mut::<Transfer>(|env, state, (to, amount)| {
                if amount == U256::zero() {
                    return false;
                }

                let from = env.original_caller();

                let existing_from = state.balances.get(&from);
                let existing_to = state.balances.get(&to);

                println!("{:?}", *state.balances);

                if existing_from >= amount {
                    state.balances.insert(&from, existing_from - amount);
                    state.balances.insert(&to, existing_to + amount);

                    true
                } else {
                    false
                }
            });

        let total = U256::from(1_000_000u64);
        let transfer_to = Address::from([123; 160]);
        let mut contract = definition.deploy(&mut env, total.clone());

        let total_in_contract = contract.call::<TotalSupply>(());
        let balance = contract.call::<BalanceOf>(DummyEnv::new().original_caller());
        let empty_balance = contract.call::<BalanceOf>(transfer_to);
        let success = contract.call::<Transfer>((transfer_to, 1_000u64.into()));
        let failure = !contract.call::<Transfer>((transfer_to, 1_000_000u64.into()));
        let post_transfer_balance = contract.call::<BalanceOf>(transfer_to);

        assert!(success);
        assert!(failure);
        assert!(total_in_contract == total);
        assert!(balance == total);
        assert!(empty_balance == U256::zero());
        assert!(post_transfer_balance == U256::from(1_000u64));
    }
}
