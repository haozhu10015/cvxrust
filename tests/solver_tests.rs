use cvxrust::atoms::{index, slice};
use cvxrust::prelude::*;
use nalgebra::DMatrix;

const TOL: f64 = 1e-4;

#[test]
fn test_matrix_solution_recovery_preserves_shape_and_order() {
    let x = variable((2, 2));
    let target = constant_dmatrix(DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]));

    let sol = Problem::minimize(sum(&x))
        .subject_to([x.eq(target)])
        .solve()
        .expect("problem should solve");

    let vals = x.value(&sol);
    if let Array::Dense(m) = vals {
        assert!((m[(0, 0)] - 1.0).abs() < TOL);
        assert!((m[(0, 1)] - 2.0).abs() < TOL);
        assert!((m[(1, 0)] - 3.0).abs() < TOL);
        assert!((m[(1, 1)] - 4.0).abs() < TOL);
    } else {
        panic!("expected dense matrix solution");
    }
}

#[test]
fn test_matrix_first_axis_slice_constraints_affect_selected_rows() {
    let x = variable((3, 4));

    let sol = Problem::minimize(sum(&x))
        .subject_to([x.ge(0.0), slice(&x, 0, 2).ge(1.0)])
        .solve()
        .expect("problem should solve");

    let vals = x.value(&sol);
    if let Array::Dense(m) = vals {
        for col in 0..4 {
            assert!((m[(0, col)] - 1.0).abs() < TOL);
            assert!((m[(1, col)] - 1.0).abs() < TOL);
            assert!(m[(2, col)].abs() < TOL);
        }
    } else {
        panic!("expected dense matrix solution");
    }
}

#[test]
fn test_matrix_first_axis_index_constraints_affect_selected_row() {
    let x = variable((3, 4));

    let sol = Problem::minimize(sum(&x))
        .subject_to([x.ge(0.0), index(&x, 1).ge(2.0)])
        .solve()
        .expect("problem should solve");

    let vals = x.value(&sol);
    if let Array::Dense(m) = vals {
        for col in 0..4 {
            assert!(m[(0, col)].abs() < TOL);
            assert!((m[(1, col)] - 2.0).abs() < TOL);
            assert!(m[(2, col)].abs() < TOL);
        }
    } else {
        panic!("expected dense matrix solution");
    }
}
