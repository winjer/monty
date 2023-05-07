mod parse;
mod prepare;
// mod run;

use rustpython_parser::ast::Constant;

use crate::parse::parse_code;
use crate::prepare::prepare;

fn main() {
    let code = "if a and b:\n x = '1'\n";
    let nodes = parse_code(code, None).unwrap();
    dbg!(&nodes);
    let nodes = prepare(nodes).unwrap();
    dbg!(nodes);
}

impl Default for Value {
    fn default() -> Self {
        Self::Undefined
    }
}

#[derive(Debug, Clone)]
enum Value {
    Undefined,
    Ellipsis,
    None,
    True,
    False,
    Int(i64),
    Bytes(Vec<u8>),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Range(i64),
}

#[derive(Clone, Debug)]
enum Operator {
    And,
    Or,
    Add,
    Sub,
    Mult,
    MatMult,
    Div,
    Mod,
    Pow,
    LShift,
    RShift,
    BitOr,
    BitXor,
    BitAnd,
    FloorDiv,
}

#[derive(Debug, Clone)]
enum Expr<T, Funcs> {
    Constant(Value),
    Name(T),
    Call {
        func: Funcs,
        args: Vec<Expr<T, Funcs>>,
        // kwargs: Vec<(T, Expr<T, Funcs>)>,
    },
    Op {
        left: Box<Expr<T, Funcs>>,
        op: Operator,
        right: Box<Expr<T, Funcs>>,
    },
    List(Vec<Expr<T, Funcs>>),
}

#[derive(Debug, Clone)]
enum Node<Vars, Funcs> {
    Pass,
    Expr(Expr<Vars, Funcs>),
    Assign {
        target: Vars,
        value: Box<Expr<Vars, Funcs>>,
    },
    For {
        target: Expr<Vars, Funcs>,
        iter: Expr<Vars, Funcs>,
        body: Vec<Node<Vars, Funcs>>,
        or_else: Vec<Node<Vars, Funcs>>,
    },
    If {
        test: Expr<Vars, Funcs>,
        body: Vec<Node<Vars, Funcs>>,
        or_else: Vec<Node<Vars, Funcs>>,
    },
}
