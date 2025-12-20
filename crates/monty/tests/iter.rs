use monty::{ExecutorIter, PyObject, StdPrint};

#[test]
fn simple_expression_completes() {
    let exec = ExecutorIter::new("x + 1", "test.py", &["x"], vec![]).unwrap();
    let result = exec.run_no_limits(vec![PyObject::Int(41)], &mut StdPrint).unwrap();
    assert_eq!(result.into_complete().expect("complete"), PyObject::Int(42));
}

#[test]
fn external_function_call_expression_statement() {
    // Calling an undefined function returns a FunctionCall variant
    let exec = ExecutorIter::new("foo(1, 2)", "test.py", &[], vec!["foo".to_string()]).unwrap();
    let progress = exec.run_no_limits(vec![], &mut StdPrint).unwrap();

    let (name, args, state) = progress.into_function_call().expect("function call");
    assert_eq!(name, "foo");
    assert_eq!(args, vec![PyObject::Int(1), PyObject::Int(2)]);

    // Resume with a return value - the value is returned (REPL behavior: last expression is returned)
    let result = state.run(PyObject::Int(100), &mut StdPrint).unwrap();
    assert_eq!(result.into_complete().expect("complete"), PyObject::Int(100));
}

#[test]
fn external_function_call_with_assignment() {
    // Test external function call in assignment: result = foo(1, 2)
    let exec = ExecutorIter::new(
        "
result = foo(1, 2)
result + 10",
        "test.py",
        &[],
        vec!["foo".to_owned()],
    )
    .unwrap();
    let progress = exec.run_no_limits(vec![], &mut StdPrint).unwrap();

    let (name, args, state) = progress.into_function_call().expect("function call");
    assert_eq!(name, "foo");
    assert_eq!(args, vec![PyObject::Int(1), PyObject::Int(2)]);

    // Resume with return value - should be assigned to 'result'
    let result = state.run(PyObject::Int(32), &mut StdPrint).unwrap();
    // result + 10 = 32 + 10 = 42
    assert_eq!(result.into_complete().expect("complete"), PyObject::Int(42));
}

#[test]
fn external_function_call_no_args() {
    // Test external function call with no arguments
    let exec = ExecutorIter::new(
        "
x = get_value()
x",
        "test.py",
        &[],
        vec!["get_value".to_owned()],
    )
    .unwrap();
    let progress = exec.run_no_limits(vec![], &mut StdPrint).unwrap();

    let (name, args, state) = progress.into_function_call().expect("function call");
    assert_eq!(name, "get_value");
    assert!(args.is_empty());

    let result = state.run(PyObject::String("hello".to_string()), &mut StdPrint).unwrap();
    assert_eq!(
        result.into_complete().expect("complete"),
        PyObject::String("hello".to_string())
    );
}

#[test]
fn multiple_external_function_calls() {
    // Test multiple external function calls in sequence
    let code = "
a = foo(1)
b = bar(2)
a + b";
    let exec = ExecutorIter::new(code, "test.py", &[], vec!["foo".to_owned(), "bar".to_owned()]).unwrap();

    // First external call: foo(1)
    let (name, args, state) = exec
        .run_no_limits(vec![], &mut StdPrint)
        .unwrap()
        .into_function_call()
        .expect("first call");
    assert_eq!(name, "foo");
    assert_eq!(args, vec![PyObject::Int(1)]);

    // Resume with foo's return value
    let progress = state.run(PyObject::Int(10), &mut StdPrint).unwrap();

    // Second external call: bar(2)
    let (name, args, state) = progress.into_function_call().expect("second call");
    assert_eq!(name, "bar");
    assert_eq!(args, vec![PyObject::Int(2)]);

    // Resume with bar's return value
    let result = state.run(PyObject::Int(20), &mut StdPrint).unwrap();
    // a + b = 10 + 20 = 30
    assert_eq!(result.into_complete().expect("complete"), PyObject::Int(30));
}

#[test]
fn external_function_call_with_builtin_args() {
    // Test external function call with builtin function results as arguments
    let exec = ExecutorIter::new("foo(len([1, 2, 3]))", "test.py", &[], vec!["foo".to_owned()]).unwrap();
    let progress = exec.run_no_limits(vec![], &mut StdPrint).unwrap();

    let (name, args, _) = progress.into_function_call().expect("function call");
    assert_eq!(name, "foo");
    // len([1, 2, 3]) = 3, so args should be [3]
    assert_eq!(args, vec![PyObject::Int(3)]);
}

#[test]
fn external_function_call_preserves_existing_variables() {
    // Test that external calls don't affect existing variables
    let code = "
x = 10
y = foo(x)
x + y";
    let exec = ExecutorIter::new(code, "test.py", &[], vec!["foo".to_owned()]).unwrap();

    let (_, args, state) = exec
        .run_no_limits(vec![], &mut StdPrint)
        .unwrap()
        .into_function_call()
        .expect("function call");
    // foo receives x=10
    assert_eq!(args, vec![PyObject::Int(10)]);

    // Resume with return value
    let result = state.run(PyObject::Int(5), &mut StdPrint).unwrap();
    // x + y = 10 + 5 = 15
    assert_eq!(result.into_complete().expect("complete"), PyObject::Int(15));
}

#[test]
fn external_function_nested_calls() {
    // Test nested external function calls: foo(bar(42))
    let code = "foo(bar(42))";
    let exec = ExecutorIter::new(code, "test.py", &[], vec!["foo".to_owned(), "bar".to_owned()]).unwrap();

    // First: inner call bar(42)
    let (name, args, state) = exec
        .run_no_limits(vec![], &mut StdPrint)
        .unwrap()
        .into_function_call()
        .expect("function call");

    assert_eq!(name, "bar");
    assert_eq!(args, vec![PyObject::Int(42)]);

    let progress = state.run(PyObject::Int(43), &mut StdPrint).unwrap();

    // Second: outer call foo(43)
    let (name, args, state) = progress.into_function_call().expect("function call");

    assert_eq!(name, "foo");
    assert_eq!(args, vec![PyObject::Int(43)]);

    let result = state.run(PyObject::Int(44), &mut StdPrint).unwrap();
    assert_eq!(result.into_complete().expect("complete"), PyObject::Int(44));
}

#[test]
fn clone_executor_iter() {
    // Test that ExecutorIter can be cloned and both copies work independently
    let exec1 = ExecutorIter::new("foo(42)", "test.py", &[], vec!["foo".to_owned()]).unwrap();
    let exec2 = exec1.clone();

    // Run first executor
    let (name, args, state) = exec1
        .run_no_limits(vec![], &mut StdPrint)
        .unwrap()
        .into_function_call()
        .expect("function call");
    assert_eq!(name, "foo");
    assert_eq!(args, vec![PyObject::Int(42)]);
    let result = state.run(PyObject::Int(100), &mut StdPrint).unwrap();
    assert_eq!(result.into_complete().expect("complete"), PyObject::Int(100));

    // Run second executor (clone) - should work independently
    let (name, args, state) = exec2
        .run_no_limits(vec![], &mut StdPrint)
        .unwrap()
        .into_function_call()
        .expect("function call");
    assert_eq!(name, "foo");
    assert_eq!(args, vec![PyObject::Int(42)]);
    let result = state.run(PyObject::Int(200), &mut StdPrint).unwrap();
    assert_eq!(result.into_complete().expect("complete"), PyObject::Int(200));
}

#[test]
fn external_function_call_in_if_true_branch() {
    // Test function call inside if block when condition is true
    let code = "
x = 1
if x == 1:
    result = foo(10)
else:
    result = bar(20)
result";
    let exec = ExecutorIter::new(code, "test.py", &[], vec!["foo".to_owned(), "bar".to_owned()]).unwrap();

    // Should call foo(10), not bar
    let (name, args, state) = exec
        .run_no_limits(vec![], &mut StdPrint)
        .unwrap()
        .into_function_call()
        .expect("function call");
    assert_eq!(name, "foo");
    assert_eq!(args, vec![PyObject::Int(10)]);

    let result = state.run(PyObject::Int(100), &mut StdPrint).unwrap();
    assert_eq!(result.into_complete().expect("complete"), PyObject::Int(100));
}

#[test]
fn external_function_call_in_if_false_branch() {
    // Test function call inside else block when condition is false
    let code = "
x = 0
if x == 1:
    result = foo(10)
else:
    result = bar(20)
result";
    let exec = ExecutorIter::new(code, "test.py", &[], vec!["foo".to_owned(), "bar".to_owned()]).unwrap();

    // Should call bar(20), not foo
    let (name, args, state) = exec
        .run_no_limits(vec![], &mut StdPrint)
        .unwrap()
        .into_function_call()
        .expect("function call");
    assert_eq!(name, "bar");
    assert_eq!(args, vec![PyObject::Int(20)]);

    let result = state.run(PyObject::Int(200), &mut StdPrint).unwrap();
    assert_eq!(result.into_complete().expect("complete"), PyObject::Int(200));
}

#[test]
fn external_function_call_in_for_loop() {
    // Test function call inside for loop
    let code = "
total = 0
for i in range(3):
    total = total + get_value(i)
total";
    let exec = ExecutorIter::new(code, "test.py", &[], vec!["get_value".to_owned()]).unwrap();

    // First iteration: get_value(0)
    let (name, args, state) = exec
        .run_no_limits(vec![], &mut StdPrint)
        .unwrap()
        .into_function_call()
        .expect("first call");
    assert_eq!(name, "get_value");
    assert_eq!(args, vec![PyObject::Int(0)]);
    let progress = state.run(PyObject::Int(10), &mut StdPrint).unwrap();

    // Second iteration: get_value(1)
    let (name, args, state) = progress.into_function_call().expect("second call");
    assert_eq!(name, "get_value");
    assert_eq!(args, vec![PyObject::Int(1)]);
    let progress = state.run(PyObject::Int(20), &mut StdPrint).unwrap();

    // Third iteration: get_value(2)
    let (name, args, state) = progress.into_function_call().expect("third call");
    assert_eq!(name, "get_value");
    assert_eq!(args, vec![PyObject::Int(2)]);
    let result = state.run(PyObject::Int(30), &mut StdPrint).unwrap();

    // total = 10 + 20 + 30 = 60
    assert_eq!(result.into_complete().expect("complete"), PyObject::Int(60));
}

#[test]
fn external_function_call_state_across_loop() {
    // Test that state persists correctly across loop iterations with function calls
    let code = "
results = []
for i in range(2):
    x = compute(i)
    results.append(x)
results";
    let exec = ExecutorIter::new(code, "test.py", &[], vec!["compute".to_owned()]).unwrap();

    // First iteration: compute(0)
    let (name, args, state) = exec
        .run_no_limits(vec![], &mut StdPrint)
        .unwrap()
        .into_function_call()
        .expect("first call");
    assert_eq!(name, "compute");
    assert_eq!(args, vec![PyObject::Int(0)]);
    let progress = state.run(PyObject::String("a".to_string()), &mut StdPrint).unwrap();

    // Second iteration: compute(1)
    let (name, args, state) = progress.into_function_call().expect("second call");
    assert_eq!(name, "compute");
    assert_eq!(args, vec![PyObject::Int(1)]);
    let result = state.run(PyObject::String("b".to_string()), &mut StdPrint).unwrap();

    // results should be ["a", "b"]
    assert_eq!(
        result.into_complete().expect("complete"),
        PyObject::List(vec![
            PyObject::String("a".to_string()),
            PyObject::String("b".to_string())
        ])
    );
}
