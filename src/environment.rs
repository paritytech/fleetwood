use parity_hash::H256;
use std::io::{self, Cursor};

pub enum Writeable<'a> {
    Slice(Cursor<&'a mut [u8]>),
    Vec(Cursor<&'a mut Vec<u8>>),
}

impl<'a> Writeable<'a> {
    pub fn set_position(&mut self, index: u64) {
        match self {
            Writeable::Slice(slice) => slice.set_position(index),
            Writeable::Vec(vec) => vec.set_position(index),
        }
    }

    pub fn position(&self) -> u64 {
        match self {
            Writeable::Slice(slice) => slice.position(),
            Writeable::Vec(vec) => vec.position(),
        }
    }
}

impl<'a> io::Write for Writeable<'a> {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        match self {
            Writeable::Slice(slice) => slice.write(input),
            Writeable::Vec(vec) => vec.write(input),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Writeable::Slice(slice) => slice.flush(),
            Writeable::Vec(vec) => vec.flush(),
        }
    }
}

pub trait Key: AsRef<[u8]> + AsMut<[u8]> + Clone {
    fn from_u64(val: u64) -> Self;
}

pub trait Value: AsRef<[u8]> + AsMut<[u8]> + Default {
    fn is_finished(index: usize) -> bool;
    fn as_writeable(&mut self) -> Writeable;
}

impl Value for [u8; 32] {
    fn is_finished(index: usize) -> bool {
        index >= 32
    }

    #[inline(always)]
    fn as_writeable(&mut self) -> Writeable {
        Writeable::Slice(Cursor::new(self))
    }
}

impl Value for Vec<u8> {
    fn is_finished(_: usize) -> bool {
        false
    }

    #[inline(always)]
    fn as_writeable(&mut self) -> Writeable {
        Writeable::Vec(Cursor::new(self))
    }
}

impl Key for H256 {
    fn from_u64(val: u64) -> Self {
        let mut out = [0; 32];
        out[0] = (val << 56) as u8;
        out[1] = (val << 48) as u8;
        out[2] = (val << 40) as u8;
        out[3] = (val << 32) as u8;
        out[4] = (val << 24) as u8;
        out[5] = (val << 16) as u8;
        out[6] = (val << 8) as u8;
        out[7] = val as u8;
        H256(out)
    }
}

pub trait HasStorage {
    type Key: Key;
    type Value: Value;

    fn read(key: &Self::Key) -> Self::Value;
    fn write(key: &Self::Key, value: &Self::Value);
}

pub trait Environment: HasStorage {}
