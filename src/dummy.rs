use pwasm_std::types::{Address, H256};
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static! {
    static ref STORAGE_MUTEX: Mutex<HashMap<[u8; 32], [u8; 32]>> = Mutex::new(HashMap::new());
}

pub fn address() -> Address {
    Address::from([3; 20])
}

pub fn sender() -> Address {
    Address::from([2; 20])
}

pub fn origin() -> Address {
    Address::from([1; 20])
}

pub fn write(key: &H256, val: &[u8; 32]) {
    STORAGE_MUTEX.lock().unwrap().insert(key.to_fixed_bytes(), val.clone());
}

pub fn read(key: &H256) -> [u8; 32] {
    STORAGE_MUTEX
        .lock()
        .unwrap()
        .get(key.as_bytes())
        .cloned()
        .unwrap_or([0; 32])
}
