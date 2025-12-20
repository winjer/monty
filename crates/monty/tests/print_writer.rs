use monty::{CollectStringPrint, Executor, NoPrint};

#[test]
fn print_single_string() {
    let ex = Executor::new("print('hello')", "test.py", &[]).unwrap();
    let mut writer = CollectStringPrint::new();
    ex.run_with_writer(vec![], &mut writer).unwrap();
    assert_eq!(writer.output(), "hello\n");
}

#[test]
fn print_multiple_args() {
    let ex = Executor::new("print('hello', 'world')", "test.py", &[]).unwrap();
    let mut writer = CollectStringPrint::new();
    ex.run_with_writer(vec![], &mut writer).unwrap();
    assert_eq!(writer.output(), "hello world\n");
}

#[test]
fn print_multiple_statements() {
    let ex = Executor::new("print('one')\nprint('two')\nprint('three')", "test.py", &[]).unwrap();
    let mut writer = CollectStringPrint::new();
    ex.run_with_writer(vec![], &mut writer).unwrap();
    assert_eq!(writer.output(), "one\ntwo\nthree\n");
}

#[test]
fn print_empty() {
    let ex = Executor::new("print()", "test.py", &[]).unwrap();
    let mut writer = CollectStringPrint::new();
    ex.run_with_writer(vec![], &mut writer).unwrap();
    assert_eq!(writer.output(), "\n");
}

#[test]
fn print_integers() {
    let ex = Executor::new("print(1, 2, 3)", "test.py", &[]).unwrap();
    let mut writer = CollectStringPrint::new();
    ex.run_with_writer(vec![], &mut writer).unwrap();
    assert_eq!(writer.output(), "1 2 3\n");
}

#[test]
fn print_mixed_types() {
    let ex = Executor::new("print('count:', 42, True)", "test.py", &[]).unwrap();
    let mut writer = CollectStringPrint::new();
    ex.run_with_writer(vec![], &mut writer).unwrap();
    assert_eq!(writer.output(), "count: 42 True\n");
}

#[test]
fn print_in_function() {
    let code = "
def greet(name):
    print('Hello', name)

greet('Alice')
greet('Bob')
";
    let ex = Executor::new(code, "test.py", &[]).unwrap();
    let mut writer = CollectStringPrint::new();
    ex.run_with_writer(vec![], &mut writer).unwrap();
    assert_eq!(writer.output(), "Hello Alice\nHello Bob\n");
}

#[test]
fn print_in_loop() {
    let code = "
for i in range(3):
    print(i)
";
    let ex = Executor::new(code, "test.py", &[]).unwrap();
    let mut writer = CollectStringPrint::new();
    ex.run_with_writer(vec![], &mut writer).unwrap();
    assert_eq!(writer.output(), "0\n1\n2\n");
}

#[test]
fn into_output_consumes_writer() {
    let ex = Executor::new("print('test')", "test.py", &[]).unwrap();
    let mut writer = CollectStringPrint::new();
    ex.run_with_writer(vec![], &mut writer).unwrap();
    let output: String = writer.into_output();
    assert_eq!(output, "test\n");
}

#[test]
fn writer_reuse_accumulates() {
    let mut writer = CollectStringPrint::new();

    let ex1 = Executor::new("print('first')", "test.py", &[]).unwrap();
    ex1.run_with_writer(vec![], &mut writer).unwrap();

    let ex2 = Executor::new("print('second')", "test.py", &[]).unwrap();
    ex2.run_with_writer(vec![], &mut writer).unwrap();

    assert_eq!(writer.output(), "first\nsecond\n");
}

#[test]
fn no_print_suppresses_output() {
    let code = "
for i in range(100):
    print('this should be suppressed', i)
";
    let ex = Executor::new(code, "test.py", &[]).unwrap();
    let mut writer = NoPrint;
    // Should complete without error, output is silently discarded
    let result = ex.run_with_writer(vec![], &mut writer);
    assert!(result.is_ok());
}
