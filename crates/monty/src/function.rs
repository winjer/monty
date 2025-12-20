use std::fmt::Write;

use crate::{
    args::ArgValues,
    exceptions::{ExcType, RunError, SimpleException, StackFrame},
    expressions::{Identifier, Node},
    heap::{Heap, HeapId},
    intern::{Interns, StringId},
    io::PrintWriter,
    namespace::{NamespaceId, Namespaces},
    position::{FrameExit, NoPositionTracker},
    resource::ResourceTracker,
    run_frame::{RunFrame, RunResult},
    value::Value,
};

/// Stores a function definition.
///
/// Contains everything needed to execute a user-defined function: the body AST,
/// initial namespace layout, and captured closure cells. Functions are stored
/// on the heap and referenced via HeapId.
///
/// # Namespace Layout
///
/// The namespace has a predictable layout that allows sequential construction:
/// ```text
/// [params...][cell_vars...][free_vars...][locals...]
/// ```
/// - Slots 0..params.len(): function parameters
/// - Slots params.len()..params.len()+cell_var_count: cell refs for variables captured by nested functions
/// - Slots after cell_vars: free_var refs (captured from enclosing scope)
/// - Remaining slots: local variables
///
/// # Closure Support
///
/// - `free_var_enclosing_slots`: Enclosing namespace slots for captured variables.
///   At definition time, cells are captured from these slots and stored in a Closure.
///   At call time, they're pushed sequentially after cell_vars.
/// - `cell_var_count`: Number of cells to create for variables captured by nested functions.
///   At call time, cells are created and pushed sequentially after params.
#[derive(Debug, Clone)]
pub struct Function {
    /// The function name (used for error messages and repr).
    pub name: Identifier,
    /// The function parameter names as interned StringIds.
    pub params: Vec<StringId>,
    /// The prepared function body AST nodes.
    pub body: Vec<Node>,
    /// Size of the initial namespace (number of local variable slots).
    pub namespace_size: usize,
    /// Enclosing namespace slots for variables captured from enclosing scopes.
    ///
    /// At definition time: look up cell HeapId from enclosing namespace at each slot.
    /// At call time: captured cells are pushed sequentially (our slots are implicit).
    pub free_var_enclosing_slots: Vec<NamespaceId>,
    /// Number of cell variables (captured by nested functions).
    ///
    /// At call time, this many cells are created and pushed right after params.
    /// Their slots are implicitly params.len()..params.len()+cell_var_count.
    pub cell_var_count: usize,
}

impl Function {
    /// Create a new function definition.
    ///
    /// # Arguments
    /// * `name` - The function name identifier
    /// * `params` - The parameter names as interned StringIds
    /// * `body` - The prepared function body AST
    /// * `namespace_size` - Number of local variable slots needed
    /// * `free_var_enclosing_slots` - Enclosing namespace slots for captured variables
    /// * `cell_var_count` - Number of cells to create for variables captured by nested functions
    pub fn new(
        name: Identifier,
        params: Vec<StringId>,
        body: Vec<Node>,
        namespace_size: usize,
        free_var_enclosing_slots: Vec<NamespaceId>,
        cell_var_count: usize,
    ) -> Self {
        Self {
            name,
            params,
            body,
            namespace_size,
            free_var_enclosing_slots,
            cell_var_count,
        }
    }

    /// Returns true if this function has any free variables (is a closure).
    #[must_use]
    pub fn is_closure(&self) -> bool {
        !self.free_var_enclosing_slots.is_empty()
    }

    /// Returns true if this function is equal to another function.
    ///
    /// We assume functions are equal if they have the same name and position.
    pub fn py_eq(&self, other: &Self) -> bool {
        self.name.py_eq(&other.name)
    }

    /// Calls this function with the given arguments.
    ///
    /// This method is used for non-closure functions. For closures (functions with
    /// captured variables), use `call_with_cells` instead.
    ///
    /// # Arguments
    /// * `namespaces` - The namespace storage for managing all namespaces
    /// * `heap` - The heap for allocating objects
    /// * `args` - The arguments to pass to the function
    /// * `interns` - String storage for looking up interned names in error messages
    /// * `writer` - The writer for print output
    pub fn call(
        &self,
        namespaces: &mut Namespaces,
        heap: &mut Heap<impl ResourceTracker>,
        args: ArgValues,
        interns: &Interns,
        writer: &mut impl PrintWriter,
    ) -> RunResult<Value> {
        // Build namespace sequentially: [params][cell_vars][free_vars][locals]
        let mut namespace = Vec::with_capacity(self.namespace_size);

        // 1. Push arguments (slots 0..params.len())
        args.inject_into_namespace(&mut namespace);
        if namespace.len() != self.params.len() {
            return self.wrong_arg_count_error(namespace.len(), interns);
        }

        // 2. Push cell_var refs (slots params.len()..params.len()+cell_var_count)
        // These are cells for variables that nested functions capture from us
        for _ in 0..self.cell_var_count {
            let cell_id = heap.alloc_cell(Value::Undefined);
            namespace.push(Value::Ref(cell_id));
        }

        // 3. No free_vars for non-closure functions (call_with_cells handles those)

        // 4. Fill remaining slots with Undefined for local variables
        namespace.resize_with(self.namespace_size, || Value::Undefined);

        // Create a new local namespace for this function call
        let local_idx = namespaces.push(namespace);

        // Create stack frame for error tracebacks
        let parent_frame = StackFrame::new(self.name.position, self.name.name_id, None);

        // Execute the function body in a new frame
        let mut p = NoPositionTracker;
        let mut frame = RunFrame::function_frame(
            local_idx,
            self.name.name_id,
            Some(parent_frame),
            interns,
            &mut p,
            writer,
        );

        let result = frame.execute(namespaces, heap, &self.body);

        // Clean up the function's namespace (properly decrementing ref counts)
        namespaces.pop_with_heap(heap);

        map_result(result)
    }

    /// Calls this function as a closure with captured cells.
    ///
    /// # Arguments
    /// * `namespaces` - The namespace manager for all namespaces
    /// * `heap` - The heap for allocating objects
    /// * `args` - The arguments to pass to the function
    /// * `captured_cells` - Cell HeapIds captured from the enclosing scope
    /// * `interns` - String storage for looking up interned names in error messages
    /// * `writer` - The writer for print output
    ///
    /// This method is called when invoking a `Value::Closure`. The captured_cells
    /// are pushed sequentially after cell_vars in the namespace.
    pub fn call_with_cells(
        &self,
        namespaces: &mut Namespaces,
        heap: &mut Heap<impl ResourceTracker>,
        args: ArgValues,
        captured_cells: &[HeapId],
        interns: &Interns,
        writer: &mut impl PrintWriter,
    ) -> RunResult<Value> {
        // Build namespace sequentially: [params][cell_vars][free_vars][locals]
        let mut namespace = Vec::with_capacity(self.namespace_size);

        // 1. Push arguments (slots 0..params.len())
        args.inject_into_namespace(&mut namespace);
        if namespace.len() != self.params.len() {
            return self.wrong_arg_count_error(namespace.len(), interns);
        }

        // 2. Push cell_var refs (slots params.len()..params.len()+cell_var_count)
        // A closure can also have cell_vars if it has nested functions
        for _ in 0..self.cell_var_count {
            let cell_id = heap.alloc_cell(Value::Undefined);
            namespace.push(Value::Ref(cell_id));
        }

        // 3. Push free_var refs (captured cells from enclosing scope)
        // Order of captured_cells matches free_var_enclosing_slots
        for &cell_id in captured_cells {
            heap.inc_ref(cell_id);
            namespace.push(Value::Ref(cell_id));
        }

        // 4. Fill remaining slots with Undefined for local variables
        namespace.resize_with(self.namespace_size, || Value::Undefined);

        // Create a new local namespace for this function call
        let local_idx = namespaces.push(namespace);

        // Create stack frame for error tracebacks
        let parent_frame = StackFrame::new(self.name.position, self.name.name_id, None);

        // Execute the function body in a new frame
        let mut p = NoPositionTracker;
        let mut frame = RunFrame::function_frame(
            local_idx,
            self.name.name_id,
            Some(parent_frame),
            interns,
            &mut p,
            writer,
        );

        let result = frame.execute(namespaces, heap, &self.body);

        // Clean up the function's namespace (properly decrementing ref counts)
        namespaces.pop_with_heap(heap);

        map_result(result)
    }

    /// Writes the Python repr() string for this function to a formatter.
    pub fn py_repr_fmt<W: Write>(
        &self,
        f: &mut W,
        interns: &Interns,
        // TODO use actual heap_id
        heap_id: usize,
    ) -> std::fmt::Result {
        write!(
            f,
            "<function '{}' at 0x{:x}>",
            interns.get_str(self.name.name_id),
            heap_id
        )
    }

    /// Creates an error for wrong number of arguments.
    ///
    /// Handles both "missing required positional arguments" and "too many arguments" cases,
    /// formatting the error message to match CPython's style.
    ///
    /// # Arguments
    /// * `actual_count` - Number of arguments actually provided
    /// * `interns` - String storage for looking up interned names
    fn wrong_arg_count_error(&self, actual_count: usize, interns: &Interns) -> RunResult<Value> {
        let func_name = interns.get_str(self.name.name_id);
        let msg = if let Some(missing_count) = self.params.len().checked_sub(actual_count) {
            // Missing arguments - show actual parameter names
            let mut msg = format!(
                "{}() missing {} required positional argument{}: ",
                func_name,
                missing_count,
                if missing_count == 1 { "" } else { "s" }
            );
            let mut missing_names: Vec<_> = self.params[actual_count..]
                .iter()
                .map(|string_id| format!("'{}'", interns.get_str(*string_id)))
                .collect();
            let last = missing_names.pop().unwrap();
            if !missing_names.is_empty() {
                msg.push_str(&missing_names.join(", "));
                msg.push_str(", and ");
            }
            msg.push_str(&last);
            msg
        } else {
            // Too many arguments
            format!(
                "{}() takes {} positional argument{} but {} {} given",
                func_name,
                self.params.len(),
                if self.params.len() == 1 { "" } else { "s" },
                actual_count,
                if actual_count == 1 { "was" } else { "were" }
            )
        };
        Err(SimpleException::new(ExcType::TypeError, Some(msg))
            .with_position(self.name.position)
            .into())
    }
}

fn map_result(result: RunResult<Option<FrameExit>>) -> RunResult<Value> {
    match result? {
        Some(FrameExit::Return(obj)) => Ok(obj),
        Some(FrameExit::ExternalCall { .. }) => {
            // External function calls inside user-defined functions not yet supported
            Err(RunError::Exc(
                ExcType::not_implemented("external function calls inside user-defined functions").into(),
            ))
        }
        None => Ok(Value::None),
    }
}
