# Fleetwood

## Description

An [eDSL](https://en.wikipedia.org/wiki/Domain-specific_language) library for writing portable, efficient and safe smart contracts in the [Rust programming language](https://www.rust-lang.org/). The goal is to be _at least_ as ergonomic as writing Solidity while improving on it in library and tooling support, but that's only the absolute minimum goal. The complete list of goals are:

## Goals

| Goals | |
|:-:|:-|
| **Ecosystem** | Allow for easy integration with the current Rust library ecosystem. |
| **Tooling** | Make the great Rust tooling work out-of-the-box for smart contract code. This includes auto-completion, syntax highlighting, code coverage for tests, go-to definitions and other IDE goodies. These should all work without any custom configuration. |
| **Testing** | Make smart contract code as easy to test as using `cargo test`, and make most testing be possible without a blockchain environment at all. |
| **Building** | Make building of smart contract code as easy as using `cargo build`. This does not necessarily mean that you will be able to just type `cargo build`. It might need a separate subcommand. |

| Key Attributes | |
|:-:|:-|
| **Efficienct** | Compile smart contract code to machine code that is _at least_ as efficient as if you used the low-level pWasm function calls directly. |
| **Robust** | Make it as simple as possible to write code that just does what is expected and as difficult as possible to write incorrect or exploitable code. |
| **Simple** | Smart contract code should be as easy-to-read as possible. Ideally, even if you are not familiar with Fleetwood. |

## Current State

As of now Fleetwood is a usable minimum viable product. It is possible to write an ERC20-style token contract that is generic over backend, and compile and deploy it for Ethereum.

## Development

The majority of work is currently put into the new implementation with codename [pDSL](https://github.com/Robbepop/pdsl) featuring the same goals and key attributes as set by Fleetwood. It is targeted for [SRML contracts](https://github.com/paritytech/substrate/tree/master/srml/contract) shipped by [Substrate](https://github.com/paritytech/substrate) and under heavy development.

## Future Work

It would be nice to be able to write smart contracts that are easily compiled for different chains with no runtime overhead while allowing to use specific details of the underlying chain. While developing the Fleetwood technology stack we are trying to uphold this future goal by considering interoperability of new features in accordance to it.
