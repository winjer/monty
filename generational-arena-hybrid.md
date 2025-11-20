# Arena Hybrid Design

This document outlines the recommended architecture for Monty to achieve Python-compatible reference semantics while maintaining performance and safety.

## Executive Summary

**Current Problem**: Clone-based `Object` enum creates value semantics, incompatible with Python's reference semantics.

**Recommended Solution**: Hybrid design with:
- **Immediate values** (Int, Bool, None) stored inline
- **Heap objects** (List, Str, Dict) allocated in arena with unique IDs
- **Reference counting** for memory management
- **Monotonically increasing IDs** (never reused) for simplicity and safety

**Key Benefit**: Enables correct Python behavior for shared mutable state while maintaining performance for common cases.

## Why This Approach?

### Compared to Arc/Rc

| Issue | Arc<Mutex<Object>> | Rc<RefCell<Object>> | Arena + IDs |
|-------|-------------------|---------------------|-------------|
| **Object Identity** | ❌ Can't distinguish equal objects | ❌ Can't distinguish equal objects | ✅ Unique IDs |
| **Python `is` operator** | ❌ Impossible | ❌ Impossible | ✅ Compare IDs |
| **Mutability** | ⚠️ Mutex overhead | ⚠️ Runtime panics | ✅ Direct mutation |
| **Performance** | ⚠️ Atomic operations | 🟢 Good | 🟢 Excellent |
| **Implementation Complexity** | 🟢 Simple | 🟢 Simple | 🟢 Simple (no free list) |
| **Cache locality** | ❌ Scattered allocations | ❌ Scattered allocations | ✅ Contiguous arena |
| **GC-ready** | ⚠️ Hard to add cycle detection | ⚠️ Hard to add cycle detection | ✅ All objects in one place |
| **Debugging** | ⚠️ Hard to inspect | ⚠️ Hard to inspect | ✅ Can dump entire heap |

### Core Insight

Python's `is` operator requires **object identity**, which neither `Arc` nor `Rc` provides:

```python
a = [1, 2, 3]
b = [1, 2, 3]
c = a

# These must work:
assert a is c      # Same object (identity)
assert a == b      # Equal value
assert not (a is b)  # Different objects
```

With `Arc<Object>` or `Rc<Object>`, you cannot distinguish between "same object" and "equal objects" because you have no stable object ID.

### Design Simplification: No Free List

**Key Decision**: IDs are **never reused** - always append to vector.

**Alternative Considered**: Free list to recycle IDs (more memory efficient but complex)

**Why Simpler is Better**:

1. **No Use-After-Free Confusion**
   - With reuse: `id=5` might point to different objects at different times
   - Without reuse: `id=5` always refers to same object (or None if freed)
   - Stale references fail clearly instead of silently corrupting data

2. **Easier Debugging**
   - Monotonic IDs (0, 1, 2, 3...) are easier to trace
   - Object lifetime tracking is straightforward
   - No "ID 42 was reused 7 times" confusion

3. **Simpler Implementation**
   - No free list management logic
   - No choosing between "reuse slot" vs "allocate new"
   - `allocate()` is just: push and increment

4. **Natural Safety**
   - No need for generational indices
   - Accessing freed ID returns clear error
   - Thread-safe atomic increment is trivial (future enhancement)

**Trade-offs Accepted**:

- ❌ Vector keeps growing (but freed slots are just `None` = 1 byte)
- ❌ Can't reclaim vector capacity without compacting
- ❌ Iteration must skip `None` entries

**For Monty's Use Case**: These trade-offs are acceptable because:
- Executions are short-lived (heap cleared between runs)
- Memory overhead is minimal (`None` vs full object)
- Simplicity enables faster development and fewer bugs

## Design Overview

### Object Representation

```rust
/// Primary value type - fits in 16 bytes (2 words)
#[derive(Clone, Debug, PartialEq)]
pub enum Object {
    // Immediate values (stored inline, no heap allocation)
    Int(i64),
    Bool(bool),
    None,

    // Heap-allocated values (stored in arena)
    Ref(ObjectId),
}

/// Index into heap arena
pub type ObjectId = usize;
```

### Heap Structure

```rust
/// Central heap managing all allocated objects
pub struct Heap {
    /// All heap-allocated objects. None = freed slot.
    /// IDs are never reused - always append new objects.
    objects: Vec<Option<HeapObject>>,

    /// Next ID to allocate (monotonically increasing)
    next_id: ObjectId,
}

/// A single heap-allocated object
struct HeapObject {
    /// Reference count for memory management
    refcount: usize,

    /// Actual object data
    data: HeapData,
}

/// Data stored on heap
#[derive(Clone, Debug)]
pub enum HeapData {
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<Object>),
    Tuple(Vec<Object>),
    Dict(HashMap<Object, Object>),
    Exception(Exception),
    // Future: Function, Class, Instance, etc.
}
```

## Implementation Plan

### Phase 1: Core Heap Infrastructure (Foundation)

**Goal**: Basic heap allocation with reference counting

**Files to Create**:
- `src/heap.rs` - Heap and HeapObject implementation

**Changes Required**:
- `src/object.rs` - Modify Object enum to use hybrid design
- `src/lib.rs` - Add heap to Executor

**Implementation Steps**:

1. **Create `src/heap.rs`**:
```rust
pub struct Heap {
    objects: Vec<Option<HeapObject>>,
    next_id: ObjectId,
}

impl Heap {
    pub fn new() -> Self {
        Heap {
            objects: Vec::new(),
            next_id: 0,
        }
    }

    /// Allocate a new heap object, returns its ID
    /// IDs are never reused - always append
    pub fn allocate(&mut self, data: HeapData) -> ObjectId {
        let id = self.next_id;
        self.objects.push(Some(HeapObject {
            refcount: 1,
            data,
        }));
        self.next_id += 1;
        id
    }

    /// Increment reference count
    pub fn inc_ref(&mut self, id: ObjectId) {
        if let Some(Some(obj)) = self.objects.get_mut(id) {
            obj.refcount += 1;
        }
    }

    /// Decrement reference count, free if zero
    pub fn dec_ref(&mut self, id: ObjectId) {
        if let Some(Some(obj)) = self.objects.get_mut(id) {
            obj.refcount -= 1;
            if obj.refcount == 0 {
                self.free_object(id);
            }
        }
    }

    /// Get immutable reference to object data
    pub fn get(&self, id: ObjectId) -> Result<&HeapData, HeapError> {
        self.objects
            .get(id)
            .and_then(|slot| slot.as_ref())
            .map(|obj| &obj.data)
            .ok_or(HeapError::InvalidId)
    }

    /// Get mutable reference to object data
    pub fn get_mut(&mut self, id: ObjectId) -> Result<&mut HeapData, HeapError> {
        self.objects
            .get_mut(id)
            .and_then(|slot| slot.as_mut())
            .map(|obj| &mut obj.data)
            .ok_or(HeapError::InvalidId)
    }

    fn free_object(&mut self, id: ObjectId) {
        // Recursively dec_ref any contained Objects
        self.dec_ref_contents(id);

        // Set slot to None, freeing the HeapObject
        self.objects[id] = None;
    }

    fn dec_ref_contents(&mut self, id: ObjectId) {
        // Need to collect IDs first to avoid borrowing issues
        let child_ids: Vec<ObjectId> = if let Some(Some(obj)) = &self.objects.get(id) {
            match &obj.data {
                HeapData::List(items) | HeapData::Tuple(items) => {
                    items.iter()
                        .filter_map(|obj| match obj {
                            Object::Ref(id) => Some(*id),
                            _ => None,
                        })
                        .collect()
                }
                HeapData::Dict(map) => {
                    map.iter()
                        .flat_map(|(k, v)| {
                            let mut ids = Vec::new();
                            if let Object::Ref(id) = k { ids.push(*id); }
                            if let Object::Ref(id) = v { ids.push(*id); }
                            ids
                        })
                        .collect()
                }
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };

        // Now dec_ref all children
        for child_id in child_ids {
            self.dec_ref(child_id);
        }
    }
}
```

2. **Update `src/object.rs`**:
```rust
// Change Object enum
pub enum Object {
    Int(i64),
    Float(f64),
    Bool(bool),
    None,
    Ref(ObjectId),
}

// Update methods to work with heap
impl Object {
    // Operations now take &mut Heap parameter
    pub fn add(&self, other: &Object, heap: &mut Heap) -> Option<Object> {
        match (self, other) {
            (Object::Int(a), Object::Int(b)) => Some(Object::Int(a + b)),
            (Object::Ref(id_a), Object::Ref(id_b)) => {
                // Get data from heap (handle Result)
                let data_a = heap.get(*id_a).ok()?;
                let data_b = heap.get(*id_b).ok()?;

                match (data_a, data_b) {
                    (HeapData::Str(a), HeapData::Str(b)) => {
                        let result = format!("{}{}", a, b);
                        let id = heap.allocate(HeapData::Str(result));
                        Some(Object::Ref(id))
                    }
                    (HeapData::List(a), HeapData::List(b)) => {
                        let mut result = a.clone();
                        result.extend_from_slice(b);
                        // Inc ref for all items
                        for obj in &result {
                            if let Object::Ref(id) = obj {
                                heap.inc_ref(*id);
                            }
                        }
                        let id = heap.allocate(HeapData::List(result));
                        Some(Object::Ref(id))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    // Similar updates for sub, modulus, etc.
}
```

3. **Update `src/lib.rs`**:
```rust
pub struct Executor<'c> {
    initial_namespace: Vec<Object>,
    nodes: Vec<Node<'c>>,
    heap: Heap,  // Add heap
}

impl<'c> Executor<'c> {
    pub fn run(&self, inputs: Vec<Object>) -> RunResult<'c, Exit<'c>> {
        let mut heap = self.heap.clone();  // Start with prepared heap
        // Pass &mut heap through execution
        // ...
    }
}
```

**Testing**: Create tests for basic heap operations (allocate, inc_ref, dec_ref, free)

### Phase 2: Update Evaluation & Execution

**Goal**: Thread heap through all evaluation and execution

**Changes Required**:
- `src/evaluate.rs` - Add heap parameter to all functions
- `src/run.rs` - Pass heap through execution
- `src/prepare.rs` - Pre-allocate constants on heap

**Implementation Steps**:

1. **Update function signatures**:
```rust
// evaluate.rs
pub fn evaluate<'c, 'd>(
    namespace: &'d mut [Object],
    heap: &'d mut Heap,  // Add heap
    expr_loc: &'d ExprLoc<'c>,
) -> RunResult<'c, Cow<'d, Object>>

// run.rs
pub struct RunFrame<'c> {
    namespace: Vec<Object>,
    heap: Heap,  // Add heap to frame
    parent: Option<Box<StackFrame<'c>>>,
    name: Cow<'c, str>,
}
```

2. **Update all call sites**: This is mechanical but extensive - every function that evaluates expressions needs the heap parameter

3. **Update prepare phase** to allocate constants:
```rust
// In prepare.rs
let mut heap = Heap::new();

// When creating constants:
let s = "hello world";
let id = heap.allocate(HeapData::Str(s.to_string()));
namespace[const_id] = Object::Ref(id);
```

**Testing**: All existing tests should still pass (behavior unchanged, just using heap internally)

### Phase 3: Assignment & Cloning Semantics

**Goal**: Implement reference semantics for assignments

**Changes Required**:
- `src/run.rs` - Update assignment handling to inc_ref
- Add Drop implementation for proper cleanup

**Implementation Steps**:

1. **Update assignment**:
```rust
// In handle_assign()
fn handle_assign(frame: &mut RunFrame, target_id: usize, value: Object) {
    // Dec ref on old value (if any)
    if let Object::Ref(old_id) = frame.namespace[target_id] {
        frame.heap.dec_ref(old_id);
    }

    // Inc ref on new value
    if let Object::Ref(new_id) = value {
        frame.heap.inc_ref(new_id);
    }

    // Store new value
    frame.namespace[target_id] = value;
}
```

2. **Implement proper cleanup**:
```rust
impl Drop for RunFrame<'_> {
    fn drop(&mut self) {
        // Dec ref all objects in namespace
        for obj in &self.namespace {
            if let Object::Ref(id) = obj {
                self.heap.dec_ref(*id);
            }
        }
    }
}
```

**Testing**: Test that reference semantics work correctly:
```python
a = [1, 2, 3]
b = a
b.append(4)
assert a == [1, 2, 3, 4]  # Must pass!
```

### Phase 4: Object Identity & `is` Operator

**Goal**: Implement Python's `is` operator

**Changes Required**:
- `src/operators.rs` - Add `Is` and `IsNot` to CmpOperator (already there!)
- `src/evaluate.rs` - Implement `is` comparison
- `src/object.rs` - Add identity comparison method

**Implementation Steps**:

1. **Add identity check**:
```rust
impl Object {
    pub fn is_identical(&self, other: &Object) -> bool {
        match (self, other) {
            // Immediate values: compare by value
            (Object::Int(a), Object::Int(b)) => a == b,
            (Object::Bool(a), Object::Bool(b)) => a == b,
            (Object::None, Object::None) => true,

            // Heap values: compare by ID
            (Object::Ref(id_a), Object::Ref(id_b)) => id_a == id_b,

            // Different types or immediate vs ref
            _ => false,
        }
    }
}
```

2. **Implement in evaluator**:
```rust
// In cmp_op()
CmpOperator::Is => Ok(left_object.is_identical(&right_object)),
CmpOperator::IsNot => Ok(!left_object.is_identical(&right_object)),
```

**Testing**:
```python
a = [1, 2, 3]
b = a
c = [1, 2, 3]

assert a is b        # Same object
assert a is not c    # Different objects
assert a == c        # But equal values
```

### Phase 5: List Methods & Mutation

**Goal**: Fix list methods to work with reference semantics

**Changes Required**:
- `src/evaluate.rs` - Update attr_call to pass heap
- `src/object.rs` - Update list methods to use heap

**Implementation Steps**:

1. **Update attr_call signature**:
```rust
fn attr_call<'c, 'd>(
    namespace: &'d mut [Object],
    heap: &'d mut Heap,  // Add heap
    expr_loc: &'d ExprLoc<'c>,
    object_ident: &Identifier<'c>,
    attr: &Attr,
    args: &'d [ExprLoc<'c>],
) -> RunResult<'c, Cow<'d, Object>>
```

2. **Update list.append()**:
```rust
// In Object::attr_call()
match (self, attr) {
    (Object::Ref(id), Attr::Append) => {
        if let HeapData::List(list) = heap.get_mut(*id) {
            let item = args[0].clone();

            // Inc ref if heap object
            if let Object::Ref(item_id) = item {
                heap.inc_ref(item_id);
            }

            list.push(item);
            Ok(Cow::Owned(Object::None))
        } else {
            Err(AttributeError)
        }
    }
    // ...
}
```

**Testing**: Verify mutation works correctly:
```python
a = [1, 2]
b = a
b.append(3)
assert a == [1, 2, 3]  # Both see the change
```

### Phase 6: Exception Objects on Heap

**Goal**: Move exceptions to heap to support exception instances

**Changes Required**:
- Remove `Object::Exc` variant
- Exceptions always stored as `Object::Ref` with `HeapData::Exception`
- Update exception raising/catching

**Implementation Steps**:

1. **Update exception creation**:
```rust
// In exceptions.rs
impl Exception {
    pub fn to_object(self, heap: &mut Heap) -> Object {
        let id = heap.allocate(HeapData::Exception(self));
        Object::Ref(id)
    }
}
```

2. **Update raise handling**:
```rust
// When raising
let exc_obj = Exception::new(args).to_object(heap);
return Err(ExceptionRaise { exc: exc_obj, frame });
```

**Testing**: Ensure exceptions work and can be passed around as values

### Phase 7: Optimization - Small Integer Caching

**Goal**: Cache small integers like CPython (-5 to 256)

**Implementation**:

```rust
impl Heap {
    pub fn new() -> Self {
        let mut heap = Heap {
            objects: Vec::new(),
            free_list: Vec::new(),
            next_id: 0,
            small_ints: [None; 262], // -5 to 256
        };

        // Pre-allocate small integers
        // (Actually, keep as immediate values - no need to cache!)
        heap
    }
}
```

**Note**: With immediate values, small integer caching is automatic!

### Phase 8: String Interning

**Goal**: Intern commonly used strings

```rust
pub struct Heap {
    objects: Vec<HeapObject>,
    free_list: Vec<ObjectId>,
    next_id: ObjectId,

    /// Interned strings map: content -> ObjectId
    interned_strings: HashMap<String, ObjectId>,
}

impl Heap {
    pub fn intern_string(&mut self, s: String) -> ObjectId {
        if let Some(&id) = self.interned_strings.get(&s) {
            self.inc_ref(id);
            id
        } else {
            let id = self.allocate(HeapData::Str(s.clone()));
            self.interned_strings.insert(s, id);
            id
        }
    }
}
```

**Benefit**: `"hello" is "hello"` returns `True` (same interned string)

## Migration Strategy

### Compatibility Layer

During migration, support both old and new APIs:

```rust
// Old API (deprecated)
impl Object {
    #[deprecated]
    pub fn add_old(&self, other: &Object) -> Option<Object> {
        let mut heap = Heap::new();
        self.add(other, &mut heap)
    }
}
```

### Gradual Migration

1. **Phase 1-2**: Internal only, tests still pass
2. **Phase 3**: Behavior changes (reference semantics)
3. **Phase 4+**: New features enabled

### Testing Strategy

At each phase:
1. All existing tests must pass
2. Add new tests for new functionality
3. Add regression tests for Python semantics

## Examples: Before vs After

### Example 1: Shared Mutable State

**Before (Wrong)**:
```python
a = [1, 2, 3]
b = a        # b is a clone
b.append(4)
print(a)     # [1, 2, 3] - unchanged (WRONG!)
```

**After (Correct)**:
```python
a = [1, 2, 3]
b = a        # b references same list
b.append(4)
print(a)     # [1, 2, 3, 4] - correct!
```

### Example 2: Object Identity

**Before (Impossible)**:
```python
a = [1, 2, 3]
b = [1, 2, 3]
print(a is b)  # Can't implement correctly
```

**After (Correct)**:
```python
a = [1, 2, 3]  # ObjectId(0)
b = [1, 2, 3]  # ObjectId(1)
c = a          # ObjectId(0) - same as a

print(a is b)  # False - different IDs
print(a is c)  # True - same ID
print(a == b)  # True - equal values
```

### Example 3: Default Mutable Arguments

**Before (Wrong)**:
```python
def append_to(item, lst=[]):
    lst.append(item)
    return lst

print(append_to(1))  # [1]
print(append_to(2))  # [2] - WRONG! New list each time
```

**After (Correct)**:
```python
def append_to(item, lst=[]):
    lst.append(item)
    return lst

print(append_to(1))  # [1]
print(append_to(2))  # [1, 2] - correct! Same list
```

## Performance Characteristics

### Memory

**Before**: `size_of::<Object>()` = 32 bytes (largest variant)
**After**: `size_of::<Object>()` = 16 bytes (8-byte discriminant + 8-byte value/ID)

**Improvement**: 50% reduction in Object size

### Allocations

**Before**: Every operation clones
```python
x = [1, 2, 3]  # Allocation
y = x          # Full clone (allocation + copy)
z = x          # Another full clone
```

**After**: Reference counting, no clones
```python
x = [1, 2, 3]  # Allocation
y = x          # Just inc_ref (no allocation)
z = x          # Just inc_ref (no allocation)
```

### Operations

| Operation | Before | After |
|-----------|--------|-------|
| `y = x` (list) | O(n) clone | O(1) inc_ref |
| `list.append()` | O(n) clone + append | O(1) append |
| Function call | O(n) clone all args | O(1) inc_ref args |
| Comparison `==` | O(n) deep compare | O(n) deep compare |
| Identity `is` | Impossible | O(1) ID compare |

## Remaining Limitations

This design solves reference semantics but does NOT solve:

1. **Closures**: Need separate environment capture mechanism
2. **Nested scopes**: Need scope chain (separate from heap)
3. **Global/nonlocal**: Need multi-level namespace lookup
4. **Circular references**: Leak memory without cycle detector
5. **Lifetime 'c**: Still need owned AST for `eval()`

These require additional architectural changes beyond the heap design.

## Future Enhancements

### 1. Cycle Detection

Add mark-and-sweep GC for unreachable cycles:

```rust
impl Heap {
    pub fn collect_garbage(&mut self, roots: &[Object]) {
        // Mark phase
        let mut marked = HashSet::new();
        self.mark_recursive(roots, &mut marked);

        // Sweep phase
        for id in 0..self.objects.len() {
            if let Some(Some(obj)) = self.objects.get(id) {
                if !marked.contains(&id) && obj.refcount > 0 {
                    // Found unreachable cycle
                    self.free_object(id);
                }
            }
        }
    }
}
```

### 2. Compacting GC

When heap becomes fragmented, compact:

```rust
impl Heap {
    pub fn compact(&mut self) -> HashMap<ObjectId, ObjectId> {
        // Move all live objects to front of array
        // Return mapping of old ID -> new ID
        // Update all references
    }
}
```

## Conclusion

The arena hybrid design provides:

✅ **Python-compatible reference semantics**
✅ **Object identity** for `is` operator
✅ **Efficient** immediate values for common cases
✅ **Safe** reference counting with clear ownership
✅ **Simple** no ID reuse eliminates entire class of bugs
✅ **Extensible** foundation for GC, closures, classes
✅ **Debuggable** can inspect entire heap state

The simplified approach (no free list, monotonic IDs) trades some memory efficiency for significant implementation simplicity and safety. For Monty's use case (sandboxed execution), this is an excellent trade-off.

## Next Steps

1. **Proof of concept**: Implement Phase 1-2 in a branch
2. **Benchmark**: Compare performance vs current implementation
3. **Validate**: Ensure existing tests pass
4. **Full migration**: Complete Phase 3-6
5. **Optimize**: Add Phase 7-8 enhancements

Estimated effort: 2-3 weeks for core implementation, 1-2 weeks for optimization and testing.
