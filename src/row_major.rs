use std::alloc::Layout;

use crate::{
    instructions::{
        F64VecLoad, FMMemAdd, FmaDest, Instruction, ScalarRegister, SimpleAddress, VectorKind,
        VectorRegister, ZeroVec,
    },
    register_alloc::{self, RegisterAlloc},
};

pub struct RowMajorMultiplier {
    // JIT compiled row-major matrix-vector multiplier.
    // This involves more operations than a column-major operation
    // since you have to shuffle and internally sum the vectors as the end,
    // however it has far more natural fma-friendly parallelism.
    //
    // Before the days of fma, column-major was strictly better
    // as you can get parallelism through both many rows at a time,
    // AND through doing a tree-sum across columns.
    //
    // Now, however, tree sums aren't better from an instructions-executed angle
    // since they prevent the use of FMAs
    shape: (usize, usize),
    mul_into: unsafe fn(*const f64, *const f64, *mut f64),
}

impl RowMajorMultiplier {
    #[inline]
    pub fn multiply_info(&self, row_major: &[f64], vector: &[f64], into: &mut [f64]) {
        assert_eq!(row_major.len(), self.shape.0 * self.shape.1);
        assert_eq!(vector.len(), self.shape.1);
        assert_eq!(into.len(), self.shape.1);
        unsafe { (self.mul_into)(row_major.as_ptr(), vector.as_ptr(), into.as_mut_ptr()) }
    }
}

fn row_major_offset(rows: usize, cols: usize, row_idx: usize, col_idx: usize) -> usize {
    assert!(row_idx < rows);
    assert!(col_idx < cols);

    (cols * row_idx) + col_idx
}

#[allow(clippy::too_many_arguments)]
fn row_major_instruction_list_from(
    matrix_rows: usize,
    matrix_cols: usize,
    start_at_row: usize,
    end_at_row: usize,
    in_matrix_register: ScalarRegister,
    in_vector_register: ScalarRegister,
    out_vector_register: ScalarRegister,
    register_alloc: &mut RegisterAlloc,
    kind_to_use: VectorKind,
) -> Vec<Box<dyn Instruction>> {
    let starting_register_set = register_alloc.clone();
    let mut register_alloc = scopeguard::guard(register_alloc, |r| *r = starting_register_set);
    assert!(start_at_row < end_at_row);
    assert!(end_at_row <= matrix_rows);

    assert!(end_at_row - start_at_row >= register_alloc.free_count() + 1);

    let target_load = VectorRegister::new(register_alloc.acquire().unwrap(), kind_to_use);

    let accumulators: Vec<_> = (start_at_row..end_at_row)
        .map(|_| register_alloc.acquire().unwrap())
        .map(|idx| VectorRegister::new(idx, kind_to_use))
        .collect();

    // First zero everything out
    let mut instructions: Vec<_> = accumulators
        .iter()
        .copied()
        .map(ZeroVec)
        .map(|v| {
            let b: Box<dyn Instruction> = Box::new(v);
            b
        })
        .collect();

    let mut current_col = 0;

    while current_col + kind_to_use.f64_width() < matrix_cols {
        let vec_target_offset = current_col;
        let vec_target_addr = SimpleAddress::for_type::<f64>(
            in_vector_register,
            vec_target_offset.try_into().unwrap(),
        );

        // TODO wrap this into a loading function
        let load_target = F64VecLoad {
            into: target_load,
            address: vec_target_addr,
        };

        instructions.push(Box::new(load_target));

        for (vec_idx, row_idx) in (start_at_row..end_at_row).enumerate() {
            let load_from = row_major_offset(matrix_rows, matrix_cols, row_idx, current_col) * 8;
            let matrix_addr =
                SimpleAddress::for_type::<f64>(in_matrix_register, load_from.try_into().unwrap());
            let fma_from_mem = FMMemAdd {
                a: target_load,
                b: matrix_addr,
                c: accumulators[vec_idx],
                dest: FmaDest::C,
            };
            instructions.push(Box::new(fma_from_mem));
        }
        current_col += kind_to_use.f64_width();
    }

    assert_eq!(
        current_col, matrix_cols,
        "Haven't implemented masked final row yet"
    );

    todo!("Haven't implemented final row shuffling");

    instructions
}

pub fn row_major_instruction_list(matrix_rows: usize, matrix_cols: usize) {}
