use std::cmp::Ordering;

use crate::args::{ArgExprs, ArgValues};
use crate::exceptions::{internal_err, InternalRunError, SimpleException};
use crate::expressions::{Expr, ExprLoc, Identifier, NameScope};
use crate::fstring::{fstring_interpolation, FStringPart};

use crate::heap::{Heap, HeapData};
use crate::intern::{ExtFunctionId, Interns};
use crate::io::PrintWriter;
use crate::namespace::{NamespaceId, Namespaces};
use crate::operators::{CmpOperator, Operator};
use crate::resource::ResourceTracker;
use crate::run_frame::RunResult;
use crate::types::{Dict, List, PyTrait, Str, Tuple};
use crate::value::{Attr, Value};

/// Container for evaluation context that holds all state needed during expression evaluation.
///
/// This struct bundles together the namespaces, local namespace index, heap, and string storage
/// to avoid passing them as separate parameters to every evaluation function.
/// It simplifies function signatures and makes the evaluation code more readable.
///
/// # Lifetimes
/// * `'h` - Lifetime of the mutable borrows (namespaces and heap)
/// * `'s` - Lifetime of the string storage and the print writer reference
///
/// # Type Parameters
/// * `T` - The resource tracker type for enforcing execution limits
/// * `W` - The writer type for print output
pub struct EvaluateExpr<'h, 's, T: ResourceTracker, W: PrintWriter> {
    /// The namespace stack containing all scopes (global, local, etc.)
    pub namespaces: &'h mut Namespaces,
    /// Index of the current local namespace in the namespace stack
    pub local_idx: NamespaceId,
    /// The heap for allocating and managing heap-allocated objects
    pub heap: &'h mut Heap<T>,
    /// String storage for looking up interned names
    pub interns: &'s Interns,
    /// Writer for print output
    pub writer: &'s mut W,
}

/// Similar to the legacy `ok!()` macro, this gives shorthand for returning early
/// when a function call result is found
macro_rules! return_ext_call {
    ($expr:expr) => {
        match $expr {
            EvalResult::Value(value) => value,
            EvalResult::ExternalCall(ext_call) => return Ok(EvalResult::ExternalCall(ext_call)),
        }
    };
}
pub(crate) use return_ext_call;

impl<'h, 's, T: ResourceTracker, W: PrintWriter> EvaluateExpr<'h, 's, T, W> {
    /// Creates a new `EvaluateExpr` with the given evaluation context.
    ///
    /// # Arguments
    /// * `namespaces` - The namespace stack containing all scopes
    /// * `local_idx` - Index of the current local namespace
    /// * `heap` - The heap for object allocation
    /// * `interns` - String storage for looking up interned names
    /// * `writer` - The writer for print output
    pub fn new(
        namespaces: &'h mut Namespaces,
        local_idx: NamespaceId,
        heap: &'h mut Heap<T>,
        interns: &'s Interns,
        writer: &'s mut W,
    ) -> Self {
        Self {
            namespaces,
            local_idx,
            heap,
            interns,
            writer,
        }
    }

    /// Evaluates an expression node and returns a value.
    ///
    /// This is the primary evaluation method that recursively evaluates expressions
    /// and returns the resulting value. The returned value may be a heap reference
    /// that the caller is responsible for dropping via `drop_with_heap`.
    pub fn evaluate_use(&mut self, expr_loc: &ExprLoc) -> RunResult<EvalResult<Value>> {
        match &expr_loc.expr {
            Expr::Literal(literal) => Ok(EvalResult::Value((*literal).into())),
            Expr::Builtin(builtins) => Ok(EvalResult::Value(Value::Builtin(*builtins))),
            Expr::Name(ident) => self
                .namespaces
                .get_var_value(self.local_idx, self.heap, ident, self.interns)
                .map(EvalResult::Value),
            Expr::Call { callable, args } => {
                let args = return_ext_call!(self.evaluate_args(args)?);
                callable.call(
                    self.namespaces,
                    self.local_idx,
                    self.heap,
                    args,
                    self.interns,
                    self.writer,
                )
            }
            Expr::AttrCall { object, attr, args } => self.attr_call(object, attr, args),
            Expr::Op { left, op, right } => match op {
                // Handle boolean operators with short-circuit evaluation.
                // These return the actual operand value, not a boolean.
                Operator::And => self.eval_and(left, right),
                Operator::Or => self.eval_or(left, right),
                _ => self.eval_op(left, op, right),
            },
            Expr::CmpOp { left, op, right } => {
                let b = return_ext_call!(self.cmp_op(left, op, right)?);
                Ok(EvalResult::Value(b.into()))
            }
            Expr::List(elements) => {
                let mut values = Vec::with_capacity(elements.len());
                for e in elements {
                    let v = return_ext_call!(self.evaluate_use(e)?);
                    values.push(v);
                }
                let heap_id = self.heap.allocate(HeapData::List(List::new(values)))?;
                Ok(EvalResult::Value(Value::Ref(heap_id)))
            }
            Expr::Tuple(elements) => {
                let mut values = Vec::with_capacity(elements.len());
                for e in elements {
                    let v = return_ext_call!(self.evaluate_use(e)?);
                    values.push(v);
                }
                let heap_id = self.heap.allocate(HeapData::Tuple(Tuple::new(values)))?;
                Ok(EvalResult::Value(Value::Ref(heap_id)))
            }
            Expr::Subscript { object, index } => {
                let obj = return_ext_call!(self.evaluate_use(object)?);
                let key = return_ext_call!(self.evaluate_use(index)?);
                let result = obj.py_getitem(&key, self.heap, self.interns);
                // Drop temporary references to object and key
                obj.drop_with_heap(self.heap);
                key.drop_with_heap(self.heap);
                result.map(EvalResult::Value)
            }
            Expr::Dict(pairs) => {
                let mut eval_pairs = Vec::with_capacity(pairs.len());
                for (key_expr, value_expr) in pairs {
                    let key = return_ext_call!(self.evaluate_use(key_expr)?);
                    let value = return_ext_call!(self.evaluate_use(value_expr)?);
                    eval_pairs.push((key, value));
                }
                let dict = Dict::from_pairs(eval_pairs, self.heap, self.interns)?;
                let dict_id = self.heap.allocate(HeapData::Dict(dict))?;
                Ok(EvalResult::Value(Value::Ref(dict_id)))
            }
            Expr::Not(operand) => {
                let b = return_ext_call!(self.evaluate_bool(operand)?);
                Ok(EvalResult::Value(Value::Bool(!b)))
            }
            Expr::UnaryMinus(operand) => {
                let val = return_ext_call!(self.evaluate_use(operand)?);
                match val {
                    Value::Int(n) => Ok(EvalResult::Value(Value::Int(-n))),
                    Value::Float(f) => Ok(EvalResult::Value(Value::Float(-f))),
                    _ => {
                        use crate::exceptions::{exc_fmt, ExcType};
                        let type_name = val.py_type(Some(self.heap));
                        // Drop the value before returning error to avoid ref counting leak
                        val.drop_with_heap(self.heap);
                        Err(
                            exc_fmt!(ExcType::TypeError; "bad operand type for unary -: '{type_name}'")
                                .with_position(expr_loc.position)
                                .into(),
                        )
                    }
                }
            }
            Expr::FString(parts) => self.evaluate_fstring(parts),
            Expr::IfElse { test, body, orelse } => {
                let b = return_ext_call!(self.evaluate_bool(test)?);
                if b {
                    self.evaluate_use(body)
                } else {
                    self.evaluate_use(orelse)
                }
            }
        }
    }

    /// Evaluates an expression node and discards the returned value.
    ///
    /// This is an optimization for statement expressions where the result
    /// is not needed. It avoids unnecessary allocations in some cases
    /// (e.g., pure literals) while still evaluating side effects.
    pub fn evaluate_discard(&mut self, expr_loc: &ExprLoc) -> RunResult<EvalResult<()>> {
        match &expr_loc.expr {
            // TODO, is this right for callable?
            Expr::Literal(_) | Expr::Builtin(_) => Ok(EvalResult::Value(())),
            Expr::Name(ident) => {
                // For discard, we just need to verify the variable exists
                match ident.scope {
                    NameScope::Cell => {
                        // Cell variable - look up from namespace and verify it's a cell
                        let namespace = self.namespaces.get(self.local_idx);
                        if let Value::Ref(cell_id) = namespace.get(ident.namespace_id()) {
                            // Just verify we can access it - don't need the value
                            let _ = self.heap.get_cell_value_ref(*cell_id);
                            Ok(EvalResult::Value(()))
                        } else {
                            panic!("Cell variable slot doesn't contain a cell reference - prepare-time bug");
                        }
                    }
                    _ => self
                        .namespaces
                        .get_var_mut(self.local_idx, ident, self.interns)
                        .map(|_| EvalResult::Value(())),
                }
            }
            Expr::Call { callable, args } => {
                let args = return_ext_call!(self.evaluate_args(args)?);
                let eval_result = callable.call(
                    self.namespaces,
                    self.local_idx,
                    self.heap,
                    args,
                    self.interns,
                    self.writer,
                )?;
                let value = return_ext_call!(eval_result);
                value.drop_with_heap(self.heap);
                Ok(EvalResult::Value(()))
            }
            Expr::AttrCall { object, attr, args } => {
                let result = return_ext_call!(self.attr_call(object, attr, args)?);
                result.drop_with_heap(self.heap);
                Ok(EvalResult::Value(()))
            }
            Expr::Op { left, op, right } => {
                // Handle and/or with short-circuit evaluation
                let result = match op {
                    Operator::And => return_ext_call!(self.eval_and(left, right)?),
                    Operator::Or => return_ext_call!(self.eval_or(left, right)?),
                    _ => return_ext_call!(self.eval_op(left, op, right)?),
                };
                result.drop_with_heap(self.heap);
                Ok(EvalResult::Value(()))
            }
            Expr::CmpOp { left, op, right } => self.cmp_op(left, op, right).map(|_| EvalResult::Value(())),
            Expr::List(elements) => {
                for el in elements {
                    return_ext_call!(self.evaluate_discard(el)?);
                }
                Ok(EvalResult::Value(()))
            }
            Expr::Tuple(elements) => {
                for el in elements {
                    return_ext_call!(self.evaluate_discard(el)?);
                }
                Ok(EvalResult::Value(()))
            }
            Expr::Subscript { object, index } => {
                return_ext_call!(self.evaluate_discard(object)?);
                return_ext_call!(self.evaluate_discard(index)?);
                Ok(EvalResult::Value(()))
            }
            Expr::Dict(pairs) => {
                for (key_expr, value_expr) in pairs {
                    return_ext_call!(self.evaluate_discard(key_expr)?);
                    return_ext_call!(self.evaluate_discard(value_expr)?);
                }
                Ok(EvalResult::Value(()))
            }
            Expr::Not(operand) | Expr::UnaryMinus(operand) => self.evaluate_discard(operand),
            Expr::FString(parts) => {
                // Still need to evaluate for side effects, then drop
                let result = return_ext_call!(self.evaluate_fstring(parts)?);
                result.drop_with_heap(self.heap);
                Ok(EvalResult::Value(()))
            }
            Expr::IfElse { test, body, orelse } => {
                let b = return_ext_call!(self.evaluate_bool(test)?);
                if b {
                    self.evaluate_discard(body)
                } else {
                    self.evaluate_discard(orelse)
                }
            }
        }
    }

    /// Evaluates an expression for its truthiness (boolean result).
    ///
    /// This is a specialized helper for conditionals that returns a `bool`
    /// directly rather than a `Value`. It includes optimizations for
    /// comparison operators, `not`, and `and`/`or` to avoid creating
    /// intermediate `Value::Bool` objects.
    pub fn evaluate_bool(&mut self, expr_loc: &ExprLoc) -> RunResult<EvalResult<bool>> {
        match &expr_loc.expr {
            Expr::CmpOp { left, op, right } => self.cmp_op(left, op, right),
            // Optimize `not` to avoid creating intermediate Value::Bool
            Expr::Not(operand) => {
                let val = return_ext_call!(self.evaluate_use(operand)?);
                let result = !val.py_bool(self.heap, self.interns);
                val.drop_with_heap(self.heap);
                Ok(EvalResult::Value(result))
            }
            // Optimize `and`/`or` with short-circuit and direct boolean conversion
            Expr::Op { left, op, right } if matches!(op, Operator::And | Operator::Or) => {
                let result = match op {
                    Operator::And => self.eval_and(left, right)?,
                    Operator::Or => self.eval_or(left, right)?,
                    _ => unreachable!(),
                };
                let value = return_ext_call!(result);
                let bool_result = value.py_bool(self.heap, self.interns);
                value.drop_with_heap(self.heap);
                Ok(EvalResult::Value(bool_result))
            }
            _ => {
                let obj = return_ext_call!(self.evaluate_use(expr_loc)?);
                let result = obj.py_bool(self.heap, self.interns);
                // Drop temporary reference
                obj.drop_with_heap(self.heap);
                Ok(EvalResult::Value(result))
            }
        }
    }

    /// Evaluates a binary operator expression (`+, -, %`, etc.).
    fn eval_op(&mut self, left: &ExprLoc, op: &Operator, right: &ExprLoc) -> RunResult<EvalResult<Value>> {
        let lhs = return_ext_call!(self.evaluate_use(left)?);
        // If evaluating right triggers an external call, we must clean up lhs before returning
        let rhs = match self.evaluate_use(right)? {
            EvalResult::Value(v) => v,
            EvalResult::ExternalCall(ext_call) => {
                lhs.drop_with_heap(self.heap);
                return Ok(EvalResult::ExternalCall(ext_call));
            }
        };
        let op_result: Option<Value> = match op {
            Operator::Add => lhs.py_add(&rhs, self.heap, self.interns)?,
            Operator::Sub => lhs.py_sub(&rhs, self.heap)?,
            Operator::Mod => lhs.py_mod(&rhs),
            Operator::Mult => lhs.py_mult(&rhs, self.heap, self.interns)?,
            Operator::Div => lhs.py_div(&rhs, self.heap)?,
            Operator::FloorDiv => lhs.py_floordiv(&rhs, self.heap)?,
            Operator::Pow => lhs.py_pow(&rhs, self.heap)?,
            _ => {
                // Drop temporary references before early return
                lhs.drop_with_heap(self.heap);
                rhs.drop_with_heap(self.heap);
                return internal_err!(InternalRunError::TodoError; "Operator {op:?} not yet implemented");
            }
        };
        if let Some(object) = op_result {
            // Drop temporary references to operands now that the operation is complete
            lhs.drop_with_heap(self.heap);
            rhs.drop_with_heap(self.heap);
            Ok(EvalResult::Value(object))
        } else {
            let lhs_type = lhs.py_type(Some(self.heap));
            let rhs_type = rhs.py_type(Some(self.heap));
            // Drop temporary references before returning error
            lhs.drop_with_heap(self.heap);
            rhs.drop_with_heap(self.heap);
            SimpleException::operand_type_error(left, op, right, lhs_type, rhs_type)
        }
    }

    /// Evaluates the `and` operator with short-circuit evaluation.
    ///
    /// Returns the first falsy value encountered, or the last value if all are truthy.
    fn eval_and(&mut self, left: &ExprLoc, right: &ExprLoc) -> RunResult<EvalResult<Value>> {
        let lhs = return_ext_call!(self.evaluate_use(left)?);
        if lhs.py_bool(self.heap, self.interns) {
            // Drop left operand since we're returning the right one
            lhs.drop_with_heap(self.heap);
            self.evaluate_use(right)
        } else {
            // Short-circuit: return the falsy left operand
            Ok(EvalResult::Value(lhs))
        }
    }

    /// Evaluates the `or` operator with short-circuit semantics.
    ///
    /// Returns the first truthy value encountered, or the last value if all are falsy.
    fn eval_or(&mut self, left: &ExprLoc, right: &ExprLoc) -> RunResult<EvalResult<Value>> {
        let lhs = return_ext_call!(self.evaluate_use(left)?);
        if lhs.py_bool(self.heap, self.interns) {
            // Short-circuit: return the truthy left operand
            Ok(EvalResult::Value(lhs))
        } else {
            // Drop left operand since we're returning the right one
            lhs.drop_with_heap(self.heap);
            self.evaluate_use(right)
        }
    }

    /// Evaluates a comparison expression and returns the boolean result.
    ///
    /// Comparisons always return bool because Python chained comparisons
    /// (e.g., `1 < x < 10`) would need the intermediate value, but we don't
    /// support chaining yet, so we can return bool directly.
    fn cmp_op(&mut self, left: &ExprLoc, op: &CmpOperator, right: &ExprLoc) -> RunResult<EvalResult<bool>> {
        let lhs = return_ext_call!(self.evaluate_use(left)?);
        // If evaluating right triggers an external call, we must clean up lhs before returning
        let rhs = match self.evaluate_use(right)? {
            EvalResult::Value(v) => v,
            EvalResult::ExternalCall(ext_call) => {
                lhs.drop_with_heap(self.heap);
                return Ok(EvalResult::ExternalCall(ext_call));
            }
        };

        let result = match op {
            CmpOperator::Eq => Some(lhs.py_eq(&rhs, self.heap, self.interns)),
            CmpOperator::NotEq => Some(!lhs.py_eq(&rhs, self.heap, self.interns)),
            CmpOperator::Gt => lhs.py_cmp(&rhs, self.heap, self.interns).map(Ordering::is_gt),
            CmpOperator::GtE => lhs.py_cmp(&rhs, self.heap, self.interns).map(Ordering::is_ge),
            CmpOperator::Lt => lhs.py_cmp(&rhs, self.heap, self.interns).map(Ordering::is_lt),
            CmpOperator::LtE => lhs.py_cmp(&rhs, self.heap, self.interns).map(Ordering::is_le),
            CmpOperator::Is => Some(lhs.is(&rhs)),
            CmpOperator::IsNot => Some(!lhs.is(&rhs)),
            CmpOperator::ModEq(v) => lhs.py_mod_eq(&rhs, *v),
            // In/NotIn are not yet supported
            _ => None,
        };

        if let Some(v) = result {
            lhs.drop_with_heap(self.heap);
            rhs.drop_with_heap(self.heap);
            Ok(EvalResult::Value(v))
        } else {
            let left_type = lhs.py_type(Some(self.heap));
            let right_type = rhs.py_type(Some(self.heap));
            lhs.drop_with_heap(self.heap);
            rhs.drop_with_heap(self.heap);
            SimpleException::cmp_type_error(left, op, right, left_type, right_type)
        }
    }

    /// Calls a method on an object: `object.attr(args)`.
    ///
    /// This evaluates `object`, looks up `attr`, calls the method with `args`,
    /// and handles proper cleanup of temporary values.
    fn attr_call(&mut self, object_ident: &Identifier, attr: &Attr, args: &ArgExprs) -> RunResult<EvalResult<Value>> {
        // Evaluate arguments first to avoid borrow conflicts
        let args = return_ext_call!(self.evaluate_args(args)?);

        // For Cell scope, look up the cell from the namespace and dereference
        if let NameScope::Cell = object_ident.scope {
            let namespace = self.namespaces.get(self.local_idx);
            let Value::Ref(cell_id) = namespace.get(object_ident.namespace_id()) else {
                panic!("Cell variable slot doesn't contain a cell reference - prepare-time bug")
            };
            // get_cell_value already handles refcount increment
            let mut cell_value = self.heap.get_cell_value(*cell_id);
            let result = cell_value.call_attr(self.heap, attr, args, self.interns);
            cell_value.drop_with_heap(self.heap);
            result.map(EvalResult::Value)
        } else {
            // For normal scopes, use get_var_mut
            let object = self
                .namespaces
                .get_var_mut(self.local_idx, object_ident, self.interns)?;
            object
                .call_attr(self.heap, attr, args, self.interns)
                .map(EvalResult::Value)
        }
    }

    /// Evaluates an f-string by processing its parts sequentially.
    ///
    /// Each part is either:
    /// - Literal: Appended directly to the result
    /// - Interpolation: Evaluate expression, apply conversion, apply format spec
    ///
    /// Reference counting: Intermediate values are properly dropped after formatting.
    /// The final result is a new heap-allocated string.
    fn evaluate_fstring(&mut self, parts: &[FStringPart]) -> RunResult<EvalResult<Value>> {
        let mut result = String::new();

        for part in parts {
            match part {
                FStringPart::Literal(s) => result.push_str(s),
                FStringPart::Interpolation {
                    expr,
                    conversion,
                    format_spec,
                } => {
                    // Evaluate the expression
                    let value = return_ext_call!(self.evaluate_use(expr)?);

                    // Process the interpolation (conversion + formatting)
                    // Note: return_ext_call! will return early on external call, before dropping value
                    // This is intentional - value must stay alive if we need to resume
                    return_ext_call!(fstring_interpolation(
                        self,
                        &mut result,
                        &value,
                        *conversion,
                        format_spec.as_ref()
                    )?);

                    // Drop the evaluated value (important for reference counting)
                    value.drop_with_heap(self.heap);
                }
            }
        }

        // Allocate result string on heap
        let heap_id = self.heap.allocate(HeapData::Str(Str::new(result)))?;
        Ok(EvalResult::Value(Value::Ref(heap_id)))
    }

    /// Evaluates function call arguments from expressions to values.
    fn evaluate_args(&mut self, args_expr: &ArgExprs) -> RunResult<EvalResult<ArgValues>> {
        match args_expr {
            ArgExprs::Zero => Ok(EvalResult::Value(ArgValues::Zero)),
            ArgExprs::One(arg) => {
                let arg = return_ext_call!(self.evaluate_use(arg)?);
                Ok(EvalResult::Value(ArgValues::One(arg)))
            }
            ArgExprs::Two(arg1, arg2) => {
                let arg1 = return_ext_call!(self.evaluate_use(arg1)?);
                // If evaluating arg2 triggers an external call, clean up arg1 first
                let arg2 = match self.evaluate_use(arg2)? {
                    EvalResult::Value(v) => v,
                    EvalResult::ExternalCall(ext_call) => {
                        arg1.drop_with_heap(self.heap);
                        return Ok(EvalResult::ExternalCall(ext_call));
                    }
                };
                Ok(EvalResult::Value(ArgValues::Two(arg1, arg2)))
            }
            ArgExprs::Args(args_exprs) => {
                let mut args: Vec<Value> = Vec::with_capacity(args_exprs.len());
                for arg_expr in args_exprs {
                    // If an external call is triggered, clean up all evaluated args first
                    let arg = match self.evaluate_use(arg_expr)? {
                        EvalResult::Value(v) => v,
                        EvalResult::ExternalCall(ext_call) => {
                            for arg in args {
                                arg.drop_with_heap(self.heap);
                            }
                            return Ok(EvalResult::ExternalCall(ext_call));
                        }
                    };
                    args.push(arg);
                }
                Ok(EvalResult::Value(ArgValues::Many(args)))
            }
            _ => todo!("Implement evaluation for kwargs"),
        }
    }
}

/// Return value from evaluating an expression.
///
/// Can be either a value or a marker indicating we must yield control to the host to call
/// this function.
#[derive(Debug)]
#[must_use]
pub enum EvalResult<T> {
    Value(T),
    ExternalCall(ExternalCall),
}

/// External function call that needs host resolution.
#[derive(Debug)]
pub struct ExternalCall {
    /// The name of the function being called.
    pub function_id: ExtFunctionId,
    /// The evaluated arguments to the function.
    pub args: ArgValues,
}

impl ExternalCall {
    /// Creates a new external function call.
    pub fn new(function_id: ExtFunctionId, args: ArgValues) -> Self {
        Self { function_id, args }
    }
}
