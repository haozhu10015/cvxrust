//! Internal broadcasting helpers for atom construction and canonicalization.

use crate::atoms::affine::{matmul, promote};
use crate::expr::{Expr, Shape, ones};

pub(crate) fn broadcast_exprs(lhs: Expr, rhs: Expr) -> (Expr, Expr) {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let target_shape = lhs_shape
        .broadcast(&rhs_shape)
        .unwrap_or_else(|| panic!("cannot broadcast shapes {} and {}", lhs_shape, rhs_shape));

    if lhs_shape == target_shape && rhs_shape == target_shape {
        return (lhs, rhs);
    }
    if lhs_shape.rows() == rhs_shape.rows() && lhs_shape.cols() == rhs_shape.cols() {
        return (lhs, rhs);
    }

    let lhs = broadcast_to(lhs, &lhs_shape, &target_shape)
        .unwrap_or_else(|| panic!("cannot broadcast shape {} to {}", lhs_shape, target_shape));
    let rhs = broadcast_to(rhs, &rhs_shape, &target_shape)
        .unwrap_or_else(|| panic!("cannot broadcast shape {} to {}", rhs_shape, target_shape));

    (lhs, rhs)
}

pub(crate) fn broadcast_to(expr: Expr, expr_shape: &Shape, target_shape: &Shape) -> Option<Expr> {
    if expr_shape == target_shape {
        return Some(expr);
    }

    if expr_shape.is_scalar_like() {
        return Some(promote(&expr, target_shape.clone()));
    }

    broadcast_2d_to(expr, expr_shape, target_shape)
}

pub(crate) fn broadcast_elementwise_exprs(
    exprs: impl IntoIterator<Item = Expr>,
) -> (Shape, Vec<Expr>) {
    let exprs: Vec<Expr> = exprs.into_iter().collect();
    let target_shape = exprs
        .iter()
        .map(|expr| expr.shape())
        .reduce(|acc, shape| {
            acc.broadcast(&shape)
                .unwrap_or_else(|| panic!("cannot broadcast shapes {} and {}", acc, shape))
        })
        .expect("elementwise atom requires at least one expression");

    let exprs = exprs
        .into_iter()
        .map(|expr| {
            let expr_shape = expr.shape();
            broadcast_to(expr, &expr_shape, &target_shape).unwrap_or_else(|| {
                panic!("cannot broadcast shape {} to {}", expr_shape, target_shape)
            })
        })
        .collect();

    (target_shape, exprs)
}

fn broadcast_2d_to(expr: Expr, expr_shape: &Shape, target_shape: &Shape) -> Option<Expr> {
    if !expr_shape.is_matrix() || !target_shape.is_matrix() {
        return None;
    }

    if expr_shape.rows() == 1 && target_shape.rows() > 1 && expr_shape.cols() == target_shape.cols()
    {
        let left = ones((target_shape.rows(), 1));
        return Some(matmul(&left, &expr));
    }

    if expr_shape.cols() == 1 && target_shape.cols() > 1 && expr_shape.rows() == target_shape.rows()
    {
        let right = ones((1, target_shape.cols()));
        return Some(matmul(&expr, &right));
    }

    None
}
