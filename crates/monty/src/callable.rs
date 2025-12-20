use crate::{
    args::ArgValues,
    builtins::Builtins,
    evaluate::{EvalResult, ExternalCall},
    exceptions::{exc_fmt, ExcType},
    expressions::Identifier,
    heap::{Heap, HeapData},
    intern::Interns,
    io::PrintWriter,
    namespace::{NamespaceId, Namespaces},
    resource::ResourceTracker,
    run_frame::RunResult,
    types::PyTrait,
    value::Value,
};

/// Target of a function call expression.
///
/// Represents a callable that can be either:
/// - A builtin function or exception resolved at parse time (`print`, `len`, `ValueError`, etc.)
/// - A name that will be looked up in the namespace at runtime (for callable variables)
///
/// Separate from Value to allow deriving Clone without Value's Clone restrictions.
#[derive(Debug, Clone, Copy)]
pub enum Callable {
    /// A builtin function like `print`, `len`, `str`, etc.
    Builtin(Builtins),
    /// A name to be looked up in the namespace at runtime (e.g., `x` in `x = len; x('abc')`).
    Name(Identifier),
}

impl Callable {
    /// Calls this callable with the given arguments.
    ///
    /// # Arguments
    /// * `namespaces` - The namespace namespaces containing all namespaces
    /// * `local_idx` - Index of the local namespace in namespaces
    /// * `heap` - The heap for allocating objects
    /// * `args` - The arguments to pass to the callable
    /// * `interns` - String storage for looking up interned names in error messages
    /// * `writer` - The writer for print output
    pub fn call(
        &self,
        namespaces: &mut Namespaces,
        local_idx: NamespaceId,
        heap: &mut Heap<impl ResourceTracker>,
        args: ArgValues,
        interns: &Interns,
        writer: &mut impl PrintWriter,
    ) -> RunResult<EvalResult<Value>> {
        match self {
            Callable::Builtin(b) => b.call(heap, args, interns, writer).map(EvalResult::Value),
            Callable::Name(ident) => {
                // Look up the callable in the namespace
                let value = namespaces.get_var(local_idx, ident, interns)?;

                match value {
                    Value::Builtin(builtin) => return builtin.call(heap, args, interns, writer).map(EvalResult::Value),
                    Value::Function(f_id) => {
                        return interns
                            .get_function(*f_id)
                            .call(namespaces, heap, args, interns, writer)
                            .map(EvalResult::Value)
                    }
                    Value::ExtFunction(f_id) => {
                        let f_id = *f_id;
                        return if let Some(return_value) = namespaces.take_return_value(heap) {
                            Ok(EvalResult::Value(return_value))
                        } else {
                            Ok(EvalResult::ExternalCall(ExternalCall::new(f_id, args)))
                        };
                    }
                    // Check for heap-allocated closure
                    Value::Ref(heap_id) => {
                        let heap_data = heap.get(*heap_id);
                        if let HeapData::Closure(f_id, cells) = heap_data {
                            let f = interns.get_function(*f_id);
                            // Clone the cells to release the borrow on heap_data before calling
                            // call_with_cells will inc_ref when injecting into the new namespace
                            let cells = cells.clone();
                            return f
                                .call_with_cells(namespaces, heap, args, &cells, interns, writer)
                                .map(EvalResult::Value);
                        }
                    }
                    _ => {}
                }
                let type_name = value.py_type(Some(heap));
                let err = exc_fmt!(ExcType::TypeError; "'{type_name}' object is not callable");
                Err(err.with_position(ident.position).into())
            }
        }
    }

    /// Returns true if this Callable is equal to another Callable.
    ///
    /// We assume functions with the same name and position in code are equal.
    pub fn py_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Builtin(b1), Self::Builtin(b2)) => b1 == b2,
            (Self::Name(n1), Self::Name(n2)) => n1.py_eq(n2),
            _ => false,
        }
    }

    pub fn py_type(&self) -> &'static str {
        match self {
            Self::Builtin(b) => b.py_type(),
            Self::Name(_) => "function",
        }
    }
}
