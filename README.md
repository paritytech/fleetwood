# Fleetwood

This is a (in-development) eDSL library for writing portable, efficient and safe smart contracts in Rust. The goal is to be _at least_ as ergonomic as writing Solidity while improving on it in library and tooling support, but that's only the absolute minimum goal. The complete list of goals are:

* To allow easy integration with the Rust library ecosystem;
* To allow Rust tooling to work out-of-the-box on smart contract code. Auto-completion, syntax highlighting, code coverage for tests, go-to definition and other RLS goodies, these should all work without any custom configuration;
* To compile to code that is at least as minimal and efficient as if you used the low-level eWASM function calls directly;
* To make it as difficult as possible to write incorrect code;
* To make the code as easy-to-read as possible (ideally, even if you're not familiar with Fleetwood);
* To make it possible to write a smart contract that will compile for different chains with no runtime overhead;
* To allow you to use the specific details of any given chain if you want to;
* To make testing as easy as using `cargo test`, and make most testing be possible without a blockchain client at all;
* To make building as easy as using `cargo build` (this doesn't necessarily mean that you will be able to just type `cargo build`, it might need a seperate subcommand).

## Roadmap:

### July 2018

Usable MVP, it should be possible to write an ERC20-style token contract that is generic over backend, and compile and deploy it for Ethereum.

I'm retiscent to write a further roadmap here since the results from this MVP should be used to inform the continuing development of this library.
