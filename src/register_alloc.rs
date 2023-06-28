#[derive(Clone)]
pub struct RegisterAlloc {
    useful_registers: Vec<u8>,
}

impl RegisterAlloc {
    pub fn new(registers: u8) -> Self {
        Self {
            useful_registers: (0..registers).collect(),
        }
    }

    pub fn is_free(&self, reg: u8) -> bool {
        self.useful_registers.contains(&reg)
    }

    pub fn free_count(&self) -> usize {
        self.useful_registers.len()
    }

    pub fn acquire(&mut self) -> Option<u8> {
        self.useful_registers.pop()
    }

    pub fn release(&mut self, reg: u8) {
        assert!(!self.is_free(reg));
        self.useful_registers.push(reg)
    }
}
