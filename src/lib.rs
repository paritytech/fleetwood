#![no_std]

extern crate pwasm_ethereum;

mod pwasm {
    use core::marker::PhantomData;
    use core::result::Result as StdResult;

    struct NoMethodError;

    type Result<T> = StdResult<T, NoMethodError>;

    pub trait RespondsTo<T> {}

    pub trait Handlers<State> {
        fn handle(&self, env: &EthEnv, state: State, name: &str, msg_data: &[u8]) -> Result<()>;
    }

    trait MessageHandlerMut<M: Message, State>:
        for<'a> FnOnce(&'a EthEnv, &'a mut State, <M as Message>::Input) -> <M as Message>::Output
    {
}
    trait MessageHandler<M: Message, State>:
        for<'a> FnOnce(&'a EthEnv, &'a State, <M as Message>::Input) -> <M as Message>::Output
    {
}

    macro_rules! impl_handlers {
        ($msg_handler:ident, $state:ident, $state_borrowed:expr, $write_state:expr) => {
            impl<A, Rest> Handlers<State> for ((PhantomData<A>, $msg_handler<A, State>), Rest)
            where
                A: Message,
                Rest: Handlers<State>,
            {
                // TODO: Pre-hash?
                fn handle(
                    &self,
                    env: &EthEnv,
                    $state: State,
                    name: &str,
                    msg_data: &[u8],
                ) -> Result<()> {
                    if A::NAME == name {
                        ((self.0).1)(env, $state_borrowed, deserialize(msg_data));
                        $write_state;
                        Ok(())
                    } else {
                        Rest::handle(&functions.1, env, $state, name, msg_data)
                    }
                }
            }
        };
    }

    impl_handlers!(MessageHandler, state, &state, {});
    // TODO: How does `write_state` work?
    impl_handlers!(MessageHandlerMut, state, &mut state, write_state(&state));

    impl<State> Handlers<State> for () {
        fn handle(&self, env: &EthEnv, state: State, name: &str, msg_data: &[u8]) -> Result<()> {
            Err(NoMethodError)
        }
    }

    pub trait Message {
        type Input: Deserialize;
        type Output: Serialize;

        // TODO: Pre-hash?
        const NAME: &'static str;
    }

    pub struct TxData(());

    pub struct RawContract {}

    trait IntoRawContract {
        fn build_and_ret(self);
    }

    impl<S, C, Msg, Handler, Rest> RespondsTo<Msg>
        for Contract<S, C, ((PhantomData<Msg>, Handler), Rest)>
    {
    }
    impl<S, C, Msg, Head, Rest> RespondsTo<Msg> for Contract<S, C, (Head, Rest)>
    where
        Contract<S, C, Rest>: RespondsTo<Msg>,
    {
    }

    // This is essentially a hack to get around the fact that `FnOnce::Output` is
    // unstable
    pub trait Constructor {
        type Output;
    }

    impl<F, Out> Constructor for F
    where
        F: FnOnce(TxInfo) -> Out,
    {
        type Output = Out;
    }

    pub struct Contract<Constructor, Handle> {
        name: &'static str,
        constructor: Constructor,
        handlers: Handle,
    }

    impl Contract<(), ()> {
        const fn new(name: &'static str) -> Self {
            Contract {
                name,
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
        const fn constructor<Cons>(self, constructor: Cons) -> Contract<Cons, ()>
        where
            Cons: Constructor + 'static,
        {
            Contract {
                constructor: constructor,
                handlers: self.handlers,
                name: self.name,
            }
        }
    }

    impl<Cons: Constructor + Copy, Handle: Handlers<Cons::Output> + Copy> Contract<Cons, Handle> {
        // We can't make this return an `impl Trait`-style result because it would require
        // HKT.
        const fn with_handler<M, H>(
            self,
            handler: H,
        ) -> Contract<Cons, ((PhantomData<M>, H), Handle)> {
            Contract {
                name: self.name,
                constructor: self.constructor,
                handlers: ((PhantomData, handler), self.handlers),
            }
        }

        const fn on_msg<M: Message>(
            self,
            handler: MessageHandler<M, Cons::Output>,
        ) -> Contract<Cons, ((PhantomData<M>, MessageHandler<M, Cons::Output>), Handle)> {
            self.with_handler(handler)
        }

        const fn on_msg_mut<M: Message>(
            self,
            handler: MessageHandlerMut<M, Cons::Output>,
        ) -> Contract<Cons, ((PhantomData<M>, MessageHandlerMut<M, Cons::Output>), Handle)>
        {
            self.with_handler(handler)
        }
    }

    pub struct DeployData<'a> {
        env: &'a EthEnv,
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
        // TODO: Do we use an owned blockchain since everything's accessed through methods
        //       anyway?
        fn blockchain(&self) -> &BlockChain {
            unimplemented!()
        }

        fn account_at(&self, addr: U256) -> Result<Account> {
            unimplemented!()
        }

        // We use different types for remote vs local contracts since
        // they require different functions to get the code

        // `impl Contract` is a `RemoteContract`
        fn contract_at(&self, addr: U256) -> Result<&impl Contract> {
            unimplemented!()
        }

        // `impl Contract` is a `LocalContract`
        fn current_contract(&self) -> Result<&impl Contract> {
            unimplemented!()
        }
    }

    pub struct BlockChain(());

    pub impl BlockChain {
        pub fn current(&self) -> &Block {
            unimplemented!();
        }

        pub fn block_hash(&self, number: u8) -> U256 {
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
    () => {}
}

fn main() {
    messages! {
        Add(usize);
        Get() -> usize;
    }

    struct State {
        current: usize,
        calls_to_add: usize,
    }

    const DEPLOY_INFO: DeployInfo = pwasm_ethereum::deploy_data();

    let initial = 1u64;

    Contract::new()
        .constructor(|_txdata| State {
            current: initial as usize,
            calls_to_add: 0usize,
        })
        .on_msg_mut::<Add>(|_env, state, to_add| {
            state.calls_to_add += 1;
            state.current += to_add;
        })
        .on_msg::<Get>(|_env, state, ()| state.current)
        .listen()
}

// This takes the `deploy` function and converts the generic `ContractDef` to a monomorphic
// type then exports a `#[no_mangle]` function that the runtime can call.
//
// The runtime is responsible for stripping everything out of the binary that isn't used by
// the constructor or message handlers of the returned contract.
eth_deploy!(deploy);

// This is essentially writing `T`'s bytes to a static location in the contract's state and
// so is vulnerable to type confusion and other nasty bugs. Our API makes it safe by
// enforcing essentially a state machine where only one type is valid in this location at
// any given time.
//
// This should be an inbuilt fn from the runtime but this is a shim
unsafe fn write_state<T>(size: usize, pointer: *const u8) {
    use pwasm_ethereum::{H256, U256};
    use std::{mem, ptr};

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
unsafe fn read_state(size: usize, pointer: *mut u8) -> T {
    use pwasm_ethereum::{H256, U256};
    use std::{mem, ptr};

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

    out
}

// `eth_deploy` expands to...
#[no_mangle]
extern "C" fn __ethereum_deploy(deploy_data: DeployData) -> () {
    let contract = deploy(deploy_data);

    // TODO: Where do we allocate this? An `ArrayVec`?
    let initial = contract.state.serialize();

    pwasm::ret(ContractInternal {});
}
