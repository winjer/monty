use std::fmt::Write;
use std::str::FromStr;

/// Built-in functions for the Python interpreter.
///
/// This module contains the `Builtins` enum representing all supported built-in
/// functions (print, len, str, etc.).
use strum::{Display, EnumString};

use crate::args::ArgValues;
use crate::exceptions::{exc_err_fmt, ExcType};

use crate::heap::{Heap, HeapData};
use crate::intern::Interns;
use crate::io::PrintWriter;
use crate::resource::ResourceTracker;
use crate::run_frame::RunResult;
use crate::types::PyTrait;
use crate::value::Value;

/// Enumerates every interpreter-native Python builtins
///
/// Uses strum derives for automatic `Display`, `FromStr`, and `AsRef<str>` implementations.
/// All variants serialize to lowercase (e.g., `Print` -> "print").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Builtins {
    Function(BuiltinsFunctions),
    /// An exception type constructor like `ValueError`, `TypeError`, etc.
    ExcType(ExcType),
}

impl Builtins {
    /// Calls this builtin with the given arguments.
    ///
    /// # Arguments
    /// * `heap` - The heap for allocating objects
    /// * `args` - The arguments to pass to the callable
    /// * `interns` - String storage for looking up interned names in error messages
    /// * `writer` - The writer for print output
    pub fn call(
        self,
        heap: &mut Heap<impl ResourceTracker>,
        args: ArgValues,
        interns: &Interns,
        writer: &mut impl PrintWriter,
    ) -> RunResult<Value> {
        match self {
            Self::Function(b) => b.call(heap, args, interns, writer),
            Self::ExcType(exc) => exc.call(heap, args, interns),
        }
    }

    /// Writes the Python repr() string for this callable to a formatter.
    pub fn py_repr_fmt<W: Write>(self, f: &mut W) -> std::fmt::Result {
        match self {
            Self::Function(b) => write!(f, "<built-in function {b}>"),
            Self::ExcType(e) => write!(f, "<class '{e}'>"),
        }
    }

    pub fn py_type(self) -> &'static str {
        match self {
            Self::Function(_) => "builtin_function_or_method",
            Self::ExcType(_) => "type",
        }
    }
}

impl FromStr for Builtins {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Ok(b) = BuiltinsFunctions::from_str(s) {
            Ok(Self::Function(b))
        } else if let Ok(exc) = ExcType::from_str(s) {
            Ok(Self::ExcType(exc))
        } else {
            Err(())
        }
    }
}

/// Enumerates every interpreter-native Python builtin functions like `print`, `len`, etc.
///
/// Uses strum derives for automatic `Display` and `FromStr` implementations.
/// All variants serialize to lowercase (e.g., `Print` -> "print").
#[derive(Debug, Clone, Copy, Display, EnumString, PartialEq, Eq)]
#[strum(serialize_all = "lowercase")]
pub enum BuiltinsFunctions {
    Print,
    Len,
    Str,
    Repr,
    Id,
    Range,
    Hash,
}

impl BuiltinsFunctions {
    /// Executes the builtin with the provided positional arguments.
    ///
    /// The `interns` parameter provides access to interned string content for py_str and py_repr.
    /// The `writer` parameter is used for print output.
    fn call(
        self,
        heap: &mut Heap<impl ResourceTracker>,
        args: ArgValues,
        interns: &Interns,
        writer: &mut impl PrintWriter,
    ) -> RunResult<Value> {
        match self {
            Self::Print => {
                match &args {
                    ArgValues::Zero => {}
                    ArgValues::One(a) => {
                        writer.stdout_write(a.py_str(heap, interns));
                    }
                    ArgValues::Two(a1, a2) => {
                        writer.stdout_write(a1.py_str(heap, interns));
                        writer.stdout_push(' ');
                        writer.stdout_write(a2.py_str(heap, interns));
                    }
                    ArgValues::Many(many) => {
                        let mut iter = many.iter();
                        writer.stdout_write(iter.next().unwrap().py_str(heap, interns));
                        for value in iter {
                            writer.stdout_push(' ');
                            writer.stdout_write(value.py_str(heap, interns));
                        }
                    }
                }
                writer.stdout_push('\n');
                args.drop_with_heap(heap);
                Ok(Value::None)
            }
            Self::Len => {
                let value = args.get_one_arg("len")?;
                let result = match value.py_len(heap, interns) {
                    Some(len) => Ok(Value::Int(len as i64)),
                    None => {
                        exc_err_fmt!(ExcType::TypeError; "object of type {} has no len()", value.py_repr(heap, interns))
                    }
                };
                value.drop_with_heap(heap);
                result
            }
            Self::Str => {
                let value = args.get_one_arg("str")?;
                let heap_id = heap.allocate(HeapData::Str(value.py_str(heap, interns).into_owned().into()))?;
                value.drop_with_heap(heap);
                Ok(Value::Ref(heap_id))
            }
            Self::Repr => {
                let value = args.get_one_arg("repr")?;
                let heap_id = heap.allocate(HeapData::Str(value.py_repr(heap, interns).into_owned().into()))?;
                value.drop_with_heap(heap);
                Ok(Value::Ref(heap_id))
            }
            Self::Id => {
                let value = args.get_one_arg("id")?;
                let id = value.id();
                // For heap values, we intentionally don't drop to prevent heap slot reuse
                // which would cause id([]) == id([]) to return True (same slot reused).
                // For immediate values, dropping is a no-op since they don't use heap slots.
                // This is an acceptable trade-off: small leak for heap values passed to id(),
                // but correct semantics for value identity.
                if matches!(value, Value::Ref(_)) {
                    #[cfg(feature = "dec-ref-check")]
                    std::mem::forget(value);
                } else {
                    value.drop_with_heap(heap);
                }
                Ok(Value::Int(id as i64))
            }
            Self::Range => {
                let value = args.get_one_arg("range")?;
                let result = value.as_int();
                value.drop_with_heap(heap);
                Ok(Value::Range(result?))
            }
            Self::Hash => {
                let value = args.get_one_arg("hash")?;
                let result = match value.py_hash_u64(heap, interns) {
                    Some(hash) => Ok(Value::Int(hash as i64)),
                    None => Err(ExcType::type_error_unhashable(value.py_type(Some(heap)))),
                };
                value.drop_with_heap(heap);
                result
            }
        }
    }
}
