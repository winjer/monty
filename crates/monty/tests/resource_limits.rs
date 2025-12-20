/// Tests for resource limits and garbage collection.
///
/// These tests verify that the `ResourceTracker` system correctly enforces
/// allocation limits, time limits, and triggers garbage collection.
use std::time::Duration;

use monty::{Executor, ExecutorIter, PyObject, ResourceLimits, RunError, StdPrint};

/// Test that allocation limits return an error.
#[test]
fn allocation_limit_exceeded() {
    let code = r"
result = []
for i in range(11):
    result.append(str(i))
result
";
    let ex = Executor::new(code, "test.py", &[]).unwrap();

    let limits = ResourceLimits::new().max_allocations(4);
    let result = ex.run_with_limits(vec![], limits);

    // Should fail due to allocation limit
    assert!(result.is_err(), "should exceed allocation limit");
    match result.unwrap_err() {
        RunError::Resource(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("allocation limit exceeded"),
                "expected allocation limit error, got: {msg}"
            );
        }
        other => panic!("expected Resource error, got: {other}"),
    }
}

#[test]
fn allocation_limit_not_exceeded() {
    let code = r"
result = []
for i in range(9):
    result.append(str(i))
result
";
    let ex = Executor::new(code, "test.py", &[]).unwrap();

    let limits = ResourceLimits::new().max_allocations(10);
    let result = ex.run_with_limits(vec![], limits);

    // Should succeed
    assert!(result.is_ok(), "should not exceed allocation limit");
}

#[test]
fn time_limit_exceeded() {
    // Create a long-running loop using for + range (while isn't implemented yet)
    // Use a very large range to ensure it runs long enough to hit the time limit
    let code = r"
x = 0
for i in range(100000000):
    x = x + 1
x
";
    let ex = Executor::new(code, "test.py", &[]).unwrap();

    // Set a short time limit
    let limits = ResourceLimits::new().max_duration(Duration::from_millis(50));
    let result = ex.run_with_limits(vec![], limits);

    // Should fail due to time limit
    assert!(result.is_err(), "should exceed time limit");
    match result.unwrap_err() {
        RunError::Resource(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("time limit exceeded"),
                "expected time limit error, got: {msg}"
            );
        }
        other => panic!("expected Resource error, got: {other}"),
    }
}

#[test]
fn time_limit_not_exceeded() {
    // Simple code that runs quickly
    let code = "x = 1 + 2\nx";
    let ex = Executor::new(code, "test.py", &[]).unwrap();

    // Set a generous time limit
    let limits = ResourceLimits::new().max_duration(Duration::from_secs(5));
    let result = ex.run_with_limits(vec![], limits);

    // Should succeed
    assert!(result.is_ok(), "should not exceed time limit");
}

/// Test that memory limits return an error.
#[test]
fn memory_limit_exceeded() {
    // Create code that builds up memory using lists
    // Each iteration creates a new list that gets appended
    let code = r"
result = []
for i in range(100):
    result.append([1, 2, 3, 4, 5])
result
";
    let ex = Executor::new(code, "test.py", &[]).unwrap();

    // Set a very low memory limit (100 bytes) to trigger on nested list allocation
    let limits = ResourceLimits::new().max_memory(100);
    let result = ex.run_with_limits(vec![], limits);

    // Should fail due to memory limit
    assert!(result.is_err(), "should exceed memory limit");
    match result.unwrap_err() {
        RunError::Resource(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("memory limit exceeded"),
                "expected memory limit error, got: {msg}"
            );
        }
        other => panic!("expected Resource error, got: {other}"),
    }
}

#[test]
fn combined_limits() {
    // Test multiple limits together
    let code = "x = 1 + 2\nx";
    let ex = Executor::new(code, "test.py", &[]).unwrap();

    let limits = ResourceLimits::new()
        .max_allocations(1000)
        .max_duration(Duration::from_secs(5))
        .max_memory(1024 * 1024);

    let result = ex.run_with_limits(vec![], limits);
    assert!(result.is_ok(), "should succeed with generous limits");
}

#[test]
fn run_without_limits_succeeds() {
    // Verify that run() still works (no limits)
    let code = r"
result = []
for i in range(100):
    result.append(str(i))
len(result)
";
    let ex = Executor::new(code, "test.py", &[]).unwrap();

    // Standard run should succeed
    let result = ex.run_no_limits(vec![]);
    assert!(result.is_ok(), "standard run should succeed");
}

#[test]
fn gc_interval_triggers_collection() {
    // This test verifies that GC can run without crashing
    // We can't easily verify that GC actually collected anything without
    // adding more introspection, but we can verify it runs
    let code = r"
result = []
for i in range(100):
    temp = [1, 2, 3]
    result.append(i)
len(result)
";
    let ex = Executor::new(code, "test.py", &[]).unwrap();

    // Set GC to run every 10 allocations
    let limits = ResourceLimits::new().gc_interval(10);
    let result = ex.run_with_limits(vec![], limits);

    assert!(result.is_ok(), "should succeed with GC enabled");
}

#[test]
#[cfg_attr(
    feature = "dec-ref-check",
    ignore = "resource exhaustion doesn't guarantee heap state consistency"
)]
fn executor_iter_resource_limit_on_resume() {
    // Test that resource limits are enforced across function calls
    // First function call succeeds, but resumed execution exceeds limit
    let code = "foo(1)\nx = []\nfor i in range(10):\n    x.append(str(i))\nlen(x)";
    let exec = ExecutorIter::new(code, "test.py", &[], vec!["foo".to_owned()]).unwrap();

    // First function call should succeed with generous limit
    let limits = ResourceLimits::new().max_allocations(5);
    let (name, args, state) = exec
        .run_with_limits(vec![], limits, &mut StdPrint)
        .unwrap()
        .into_function_call()
        .expect("function call");
    assert_eq!(name, "foo");
    assert_eq!(args, vec![PyObject::Int(1)]);

    // Resume - should fail due to allocation limit during the for loop
    let result = state.run(PyObject::None, &mut StdPrint);
    assert!(result.is_err(), "should exceed allocation limit on resume");
    match result.unwrap_err() {
        RunError::Resource(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("allocation limit exceeded"),
                "expected allocation limit error, got: {msg}"
            );
        }
        other => panic!("expected Resource error, got: {other}"),
    }
}

#[test]
#[cfg_attr(
    feature = "dec-ref-check",
    ignore = "resource exhaustion doesn't guarantee heap state consistency"
)]
fn executor_iter_resource_limit_before_function_call() {
    // Test that resource limits are enforced before first function call
    let code = "x = []\nfor i in range(10):\n    x.append(str(i))\nfoo(len(x))\n42";
    let exec = ExecutorIter::new(code, "test.py", &[], vec!["foo".to_owned()]).unwrap();

    // Should fail before reaching the function call
    let limits = ResourceLimits::new().max_allocations(3);
    let result = exec.run_with_limits(vec![], limits, &mut StdPrint);

    assert!(result.is_err(), "should exceed allocation limit before function call");
    match result.unwrap_err() {
        RunError::Resource(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("allocation limit exceeded"),
                "expected allocation limit error, got: {msg}"
            );
        }
        other => panic!("expected Resource error, got: {other}"),
    }
}

#[test]
fn executor_iter_resource_limit_multiple_function_calls() {
    // Test resource limits across multiple function calls
    let code = "foo(1)\nbar(2)\nbaz(3)\n4";
    let exec = ExecutorIter::new(
        code,
        "test.py",
        &[],
        vec!["foo".to_owned(), "bar".to_owned(), "baz".to_owned()],
    )
    .unwrap();

    // Very tight allocation limit - should still work for simple function calls
    let limits = ResourceLimits::new().max_allocations(100);

    let (name, args, state) = exec
        .run_with_limits(vec![], limits, &mut StdPrint)
        .unwrap()
        .into_function_call()
        .expect("first call");
    assert_eq!(name, "foo");
    assert_eq!(args, vec![PyObject::Int(1)]);

    let (name, args, state) = state
        .run(PyObject::None, &mut StdPrint)
        .unwrap()
        .into_function_call()
        .expect("second call");
    assert_eq!(name, "bar");
    assert_eq!(args, vec![PyObject::Int(2)]);

    let (name, args, state) = state
        .run(PyObject::None, &mut StdPrint)
        .unwrap()
        .into_function_call()
        .expect("third call");
    assert_eq!(name, "baz");
    assert_eq!(args, vec![PyObject::Int(3)]);

    let result = state
        .run(PyObject::None, &mut StdPrint)
        .unwrap()
        .into_complete()
        .expect("complete");
    assert_eq!(result, PyObject::Int(4));
}
