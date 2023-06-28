use std::{alloc::Layout, fmt::Debug};

pub trait Instruction: Debug {}

#[derive(Copy, Clone, Debug)]
pub enum VectorKind {
    SingleXmm,
    Xmm,
    Ymm,
    Zmm,
}

impl VectorKind {
    pub fn f64_width(&self) -> usize {
        match self {
            Self::SingleXmm => 1,
            Self::Xmm => 2,
            Self::Ymm => 4,
            Self::Zmm => 8,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VectorRegister {
    pub kind: VectorKind,
    index: u8,
}

impl VectorRegister {
    pub fn new(index: u8, kind: VectorKind) -> Self {
        assert!(index < 31);
        Self { index, kind }
    }

    pub fn index(&self) -> u8 {
        self.index
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ScalarRegister {
    index: u8,
}

impl ScalarRegister {
    pub fn new(index: u8) -> Self {
        assert!(index < 12);
        Self { index }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SimpleAddress {
    pub base_register: ScalarRegister,
    pub offset: i32,
    pub layout: Layout,
}

impl SimpleAddress {
    pub fn for_type<T: Sized>(base_register: ScalarRegister, offset: i32) -> Self {
        Self {
            base_register,
            offset,
            layout: Layout::new::<T>(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct F64VecLoad {
    pub address: SimpleAddress,
    pub into: VectorRegister,
}

#[derive(Copy, Clone, Debug)]
pub struct MaskedF64VecLoad {
    pub load: F64VecLoad,
    pub mask: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct ZeroVec(pub VectorRegister);

#[derive(Copy, Clone, Debug)]
pub enum FmaDest {
    A,
    B,
    C,
}

#[derive(Copy, Clone, Debug)]
pub struct FMAdd {
    // a * b + c
    pub a: VectorRegister,
    pub b: VectorRegister,
    pub c: VectorRegister,
    pub dest: FmaDest,
}

#[derive(Copy, Clone, Debug)]
pub struct FMMemAdd {
    // a * *b + c
    pub a: VectorRegister,
    pub b: SimpleAddress,
    pub c: VectorRegister,
    pub dest: FmaDest,
}

impl Instruction for F64VecLoad {}
impl Instruction for MaskedF64VecLoad {}
impl Instruction for ZeroVec {}
impl Instruction for FMAdd {}
impl Instruction for FMMemAdd {}
