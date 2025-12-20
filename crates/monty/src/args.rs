use crate::{
    exceptions::ExcType,
    expressions::{ExprLoc, Identifier},
    heap::Heap,
    intern::Interns,
    run_frame::RunResult,
    value::Value,
    ParseError, PyObject, ResourceTracker,
};

/// Type for method call arguments.
///
/// Uses specific variants for common cases (0-2 arguments).
/// Most Python method calls have at most 2 arguments, so this optimization
/// eliminates the Vec heap allocation overhead for the vast majority of calls.
#[derive(Debug)]
pub enum ArgValues {
    Zero,
    One(Value),
    Two(Value, Value),
    Many(Vec<Value>),
    // TODO kwarg types
}

impl ArgValues {
    /// Checks that zero arguments were passed.
    pub fn check_zero_args(&self, name: &str) -> RunResult<()> {
        match self {
            Self::Zero => Ok(()),
            _ => Err(ExcType::type_error_no_args(name, self.count())),
        }
    }

    /// Checks that exactly one argument was passed, returning it.
    pub fn get_one_arg(self, name: &str) -> RunResult<Value> {
        match self {
            Self::One(a) => Ok(a),
            _ => Err(ExcType::type_error_arg_count(name, 1, self.count())),
        }
    }

    /// Checks that exactly two arguments were passed, returning them as a tuple.
    pub fn get_two_args(self, name: &str) -> RunResult<(Value, Value)> {
        match self {
            Self::Two(a1, a2) => Ok((a1, a2)),
            _ => Err(ExcType::type_error_arg_count(name, 2, self.count())),
        }
    }

    /// Checks that one or two arguments were passed, returning them as a tuple.
    pub fn get_one_two_args(self, name: &str) -> RunResult<(Value, Option<Value>)> {
        match self {
            Self::One(a) => Ok((a, None)),
            Self::Two(a1, a2) => Ok((a1, Some(a2))),
            Self::Zero => Err(ExcType::type_error_at_least(name, 1, self.count())),
            Self::Many(_) => Err(ExcType::type_error_at_most(name, 2, self.count())),
        }
    }

    /// Create a new namespace for a function arguments
    pub fn inject_into_namespace(self, namespace: &mut Vec<Value>) {
        match self {
            Self::Zero => (),
            Self::One(a) => {
                namespace.push(a);
            }
            Self::Two(a1, a2) => {
                namespace.push(a1);
                namespace.push(a2);
            }
            Self::Many(args) => {
                namespace.extend(args);
            }
        }
    }

    /// Converts the arguments into a Vec of PyObjects.
    ///
    /// This is used when passing arguments to external functions.
    pub fn into_py_objects(self, heap: &mut Heap<impl ResourceTracker>, interns: &Interns) -> Vec<PyObject> {
        match self {
            Self::Zero => vec![],
            Self::One(a) => vec![PyObject::new(a, heap, interns)],
            Self::Two(a1, a2) => vec![PyObject::new(a1, heap, interns), PyObject::new(a2, heap, interns)],
            Self::Many(args) => args.into_iter().map(|v| PyObject::new(v, heap, interns)).collect(),
        }
    }

    /// Returns the number of arguments.
    fn count(&self) -> usize {
        match self {
            Self::Zero => 0,
            Self::One(_) => 1,
            Self::Two(_, _) => 2,
            Self::Many(args) => args.len(),
        }
    }

    /// Properly drops all values in the arguments, decrementing reference counts.
    ///
    /// This must be called when discarding `ArgValues` that may contain `Value::Ref`
    /// variants to maintain correct reference counts on the heap.
    pub fn drop_with_heap(self, heap: &mut Heap<impl ResourceTracker>) {
        match self {
            Self::Zero => {}
            Self::One(v) => v.drop_with_heap(heap),
            Self::Two(v1, v2) => {
                v1.drop_with_heap(heap);
                v2.drop_with_heap(heap);
            }
            Self::Many(args) => {
                for v in args {
                    v.drop_with_heap(heap);
                }
            }
        }
    }
}

/// A keyword argument in a function call expression.
#[derive(Debug, Clone)]
pub struct Kwarg {
    pub key: Identifier,
    pub value: ExprLoc,
}

/// Expressions that make up a function call's arguments.
#[derive(Debug, Clone)]
pub enum ArgExprs {
    Zero,
    One(Box<ExprLoc>),
    Two(Box<ExprLoc>, Box<ExprLoc>),
    Args(Vec<ExprLoc>),
    Kwargs(Vec<Kwarg>),
    ArgsKargs { args: Vec<ExprLoc>, kwargs: Vec<Kwarg> },
}

impl ArgExprs {
    pub fn new(args: Vec<ExprLoc>, kwargs: Vec<Kwarg>) -> Self {
        if !kwargs.is_empty() {
            if args.is_empty() {
                Self::Kwargs(kwargs)
            } else {
                Self::ArgsKargs { args, kwargs }
            }
        } else if args.len() > 2 {
            Self::Args(args)
        } else {
            let mut iter = args.into_iter();
            if let Some(first) = iter.next() {
                if let Some(second) = iter.next() {
                    Self::Two(Box::new(first), Box::new(second))
                } else {
                    Self::One(Box::new(first))
                }
            } else {
                Self::Zero
            }
        }
    }

    /// Applies a transformation function to all `ExprLoc` elements in the args.
    ///
    /// This is used during the preparation phase to recursively prepare all
    /// argument expressions before execution.
    pub fn prepare_args(
        &mut self,
        mut f: impl FnMut(ExprLoc) -> Result<ExprLoc, ParseError>,
    ) -> Result<(), ParseError> {
        // Swap self with Empty to take ownership, then rebuild
        let taken = std::mem::replace(self, Self::Zero);
        *self = match taken {
            Self::Zero => Self::Zero,
            Self::One(arg) => Self::One(Box::new(f(*arg)?)),
            Self::Two(arg1, arg2) => Self::Two(Box::new(f(*arg1)?), Box::new(f(*arg2)?)),
            Self::Args(args) => Self::Args(args.into_iter().map(&mut f).collect::<Result<Vec<_>, _>>()?),
            Self::Kwargs(kwargs) => Self::Kwargs(
                kwargs
                    .into_iter()
                    .map(|kwarg| {
                        Ok(Kwarg {
                            key: kwarg.key,
                            value: f(kwarg.value)?,
                        })
                    })
                    .collect::<Result<Vec<_>, ParseError>>()?,
            ),
            Self::ArgsKargs { args, kwargs } => {
                let args = args.into_iter().map(&mut f).collect::<Result<Vec<_>, ParseError>>()?;
                let kwargs = kwargs
                    .into_iter()
                    .map(|kwarg| {
                        Ok(Kwarg {
                            key: kwarg.key,
                            value: f(kwarg.value)?,
                        })
                    })
                    .collect::<Result<Vec<_>, ParseError>>()?;
                Self::ArgsKargs { args, kwargs }
            }
        };
        Ok(())
    }
}
