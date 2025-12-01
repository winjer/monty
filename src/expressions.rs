use std::fmt::{self, Write};
use std::str::FromStr;

use crate::args::ArgExprs;
use crate::builtins::Builtins;
use crate::exceptions::{ExcType, ExceptionRaise};
use crate::object::{Attr, Object};
use crate::operators::{CmpOperator, Operator};
use crate::parse::CodeRange;
use crate::values::bytes::bytes_repr;
use crate::values::str::string_repr;

#[derive(Debug, Clone)]
pub(crate) struct Identifier<'c> {
    pub position: CodeRange<'c>,
    pub name: String,
    pub id: usize,
}

impl<'c> Identifier<'c> {
    pub fn new(name: String, position: CodeRange<'c>) -> Self {
        Self { name, position, id: 0 }
    }
}

/// Represents a callable entity in the Python runtime.
///
/// A callable can be a builtin function, an exception type (which acts as a constructor),
/// or an identifier that will be resolved during preparation.
#[derive(Debug, Clone)]
pub(crate) enum Callable<'c> {
    Builtin(Builtins),
    Exception(ExcType),
    // TODO can we remove Ident here and thereby simplify Callable?
    Ident(Identifier<'c>),
}

impl fmt::Display for Callable<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Builtin(b) => write!(f, "{b}"),
            Self::Exception(exc) => write!(f, "{exc}"),
            Self::Ident(i) => f.write_str(&i.name),
        }
    }
}

/// Parses a callable from its string representation.
///
/// Attempts to resolve the name as a builtin function first, then as an exception type.
/// Returns an error if the name doesn't match any known builtin or exception.
///
/// This is used during the preparation phase to resolve identifier names into their
/// corresponding builtin or exception type callables.
///
/// # Examples
/// - `"print".parse::<Callable>()` returns `Ok(Callable::Builtin(Builtins::Print))`
/// - `"ValueError".parse::<Callable>()` returns `Ok(Callable::Exception(ExcType::ValueError))`
/// - `"unknown".parse::<Callable>()` returns `Err(())`
impl FromStr for Callable<'static> {
    type Err = ();

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        if let Ok(builtin) = name.parse::<Builtins>() {
            Ok(Self::Builtin(builtin))
        } else if let Ok(exc_type) = name.parse::<ExcType>() {
            Ok(Self::Exception(exc_type))
        } else {
            Err(())
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Expr<'c> {
    Constant(Const),
    Name(Identifier<'c>),
    Call {
        callable: Callable<'c>,
        args: ArgExprs<'c>,
    },
    AttrCall {
        object: Identifier<'c>,
        attr: Attr,
        args: ArgExprs<'c>,
    },
    Op {
        left: Box<ExprLoc<'c>>,
        op: Operator,
        right: Box<ExprLoc<'c>>,
    },
    CmpOp {
        left: Box<ExprLoc<'c>>,
        op: CmpOperator,
        right: Box<ExprLoc<'c>>,
    },
    List(Vec<ExprLoc<'c>>),
    Tuple(Vec<ExprLoc<'c>>),
    Subscript {
        object: Box<ExprLoc<'c>>,
        index: Box<ExprLoc<'c>>,
    },
    Dict(Vec<(ExprLoc<'c>, ExprLoc<'c>)>),
}

impl fmt::Display for Expr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant(object) => write!(f, "{object}"),
            Self::Name(identifier) => f.write_str(&identifier.name),
            Self::Call { callable, args } => write!(f, "{callable}{args}"),
            Self::AttrCall { object, attr, args } => write!(f, "{}.{}{}", object.name, attr, args),
            Self::Op { left, op, right } => write!(f, "{left} {op} {right}"),
            Self::CmpOp { left, op, right } => write!(f, "{left} {op} {right}"),
            Self::List(itms) => {
                write!(
                    f,
                    "[{}]",
                    itms.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
                )
            }
            Self::Tuple(itms) => {
                write!(
                    f,
                    "({})",
                    itms.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
                )
            }
            Self::Subscript { object, index } => write!(f, "{object}[{index}]"),
            Self::Dict(pairs) => {
                if pairs.is_empty() {
                    f.write_str("{}")
                } else {
                    f.write_char('{')?;
                    let mut iter = pairs.iter();
                    if let Some((k, v)) = iter.next() {
                        write!(f, "{k}: {v}")?;
                    }
                    for (k, v) in iter {
                        write!(f, ", {k}: {v}")?;
                    }
                    f.write_char('}')
                }
            }
        }
    }
}

impl Expr<'_> {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::Constant(Const::None))
    }
}

/// Represents values that can be produced purely from the parser/prepare pipeline.
///
/// Const values are intentionally detached from the runtime heap so we can keep
/// parse-time transformations (constant folding, namespace seeding, etc.) free from
/// reference-count semantics. Only once execution begins are these literals turned
/// into real `Object`s that participate in the interpreter's runtime rules.
///
/// Note: unlike the AST `Constant` type, we store tuples only as expressions since they
/// can't always be recorded as constants.
#[derive(Debug, Clone, PartialEq)]
pub enum Const {
    Ellipsis,
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
}

impl Const {
    /// Converts the literal into its runtime `Object` counterpart.
    ///
    /// This is the only place parse-time data crosses the boundary into runtime
    /// semantics, ensuring every literal follows the same conversion path.
    pub fn to_object(&self) -> Object<'_> {
        match self {
            Self::Ellipsis => Object::Ellipsis,
            Self::None => Object::None,
            Self::Bool(b) => Object::Bool(*b),
            Self::Int(v) => Object::Int(*v),
            Self::Float(v) => Object::Float(*v),
            Self::Str(s) => Object::InternString(s),
            Self::Bytes(b) => Object::InternBytes(b),
        }
    }
}

impl fmt::Display for Const {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ellipsis => f.write_str("..."),
            Self::None => f.write_str("None"),
            Self::Bool(true) => f.write_str("True"),
            Self::Bool(false) => f.write_str("False"),
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Str(v) => f.write_str(&string_repr(v)),
            Self::Bytes(v) => f.write_str(&bytes_repr(v)),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ExprLoc<'c> {
    pub position: CodeRange<'c>,
    pub expr: Expr<'c>,
}

impl fmt::Display for ExprLoc<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // don't show position as that should be displayed separately
        write!(f, "{}", self.expr)
    }
}

impl<'c> ExprLoc<'c> {
    pub fn new(position: CodeRange<'c>, expr: Expr<'c>) -> Self {
        Self { position, expr }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Node<'c> {
    Pass,
    Expr(ExprLoc<'c>),
    Return(ExprLoc<'c>),
    ReturnNone,
    Raise(Option<ExprLoc<'c>>),
    Assign {
        target: Identifier<'c>,
        object: ExprLoc<'c>,
    },
    OpAssign {
        target: Identifier<'c>,
        op: Operator,
        object: ExprLoc<'c>,
    },
    SubscriptAssign {
        target: Identifier<'c>,
        index: ExprLoc<'c>,
        value: ExprLoc<'c>,
    },
    For {
        target: Identifier<'c>,
        iter: ExprLoc<'c>,
        body: Vec<Node<'c>>,
        or_else: Vec<Node<'c>>,
    },
    If {
        test: ExprLoc<'c>,
        body: Vec<Node<'c>>,
        or_else: Vec<Node<'c>>,
    },
}

#[derive(Debug)]
pub enum FrameExit<'c, 'e> {
    Return(Object<'e>),
    // Yield(Object),
    #[allow(dead_code)] // Planned for future use
    Raise(ExceptionRaise<'c>),
}
