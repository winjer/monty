use std::fmt::{self, Write};

use strum::Display;

/// Binary operators for arithmetic, bitwise, and boolean operations.
///
/// Uses strum `Display` derive with per-variant serialization for operator symbols.
#[derive(Clone, Debug, PartialEq, Display)]
pub(crate) enum Operator {
    #[strum(serialize = "+")]
    Add,
    #[strum(serialize = "-")]
    Sub,
    #[strum(serialize = "*")]
    Mult,
    #[strum(serialize = "@")]
    MatMult,
    #[strum(serialize = "/")]
    Div,
    #[strum(serialize = "%")]
    Mod,
    #[strum(serialize = "**")]
    Pow,
    #[strum(serialize = "<<")]
    LShift,
    #[strum(serialize = ">>")]
    RShift,
    #[strum(serialize = "|")]
    BitOr,
    #[strum(serialize = "^")]
    BitXor,
    #[strum(serialize = "&")]
    BitAnd,
    #[strum(serialize = "//")]
    FloorDiv,
    // bool operators
    #[strum(serialize = "and")]
    And,
    #[strum(serialize = "or")]
    Or,
}

/// Defined separately since these operators always return a bool
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CmpOperator {
    Eq,
    NotEq,
    Lt,
    LtE,
    Gt,
    GtE,
    Is,
    IsNot,
    In,
    NotIn,
    // we should support floats too, either via a Number type, or ModEqInt and ModEqFloat
    ModEq(i64),
}

impl fmt::Display for CmpOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eq => f.write_str("=="),
            Self::NotEq => f.write_str("!="),
            Self::Lt => f.write_char('<'),
            Self::LtE => f.write_str("<="),
            Self::Gt => f.write_char('>'),
            Self::GtE => f.write_str(">="),
            Self::Is => f.write_str("is"),
            Self::IsNot => f.write_str("is not"),
            Self::In => f.write_str("in"),
            Self::NotIn => f.write_str("not in"),
            Self::ModEq(v) => write!(f, "% X == {v}"),
        }
    }
}
