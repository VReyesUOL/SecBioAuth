use crate::common::config::{BMDB1, FRGC, PUT};

mod common;
mod gpu;

fn main() {
    println!("Hello, world!");
    gpu::auth(PUT);
    gpu::auth(BMDB1);
    gpu::auth(FRGC);

    println!("Goodbye");
}
