use monty::{Executor, Exit};
use std::error::Error;
use std::fs;
use std::path::Path;

/// Represents the expected outcome of a test fixture
#[derive(Debug)]
enum Expectation {
    /// Expect successful execution with specific output format: "type: repr"
    Return(String),
    /// Expect exception with specific message
    Raise(String),
    /// Expect parse error containing message
    ParseError(String),
    /// Expect successful execution, check py_str() output
    ReturnStr(String),
    /// Expect successful execution, check py_repr() output
    ReturnRepr(String),
    /// Expect successful execution, check py_type() output
    ReturnType(String),
    /// Expect successful execution, check is_none() result
    ReturnIsNone(bool),
    /// Expect successful execution, check is_ellipsis() result
    ReturnIsEllipsis(bool),
}

/// Parse a Python fixture file into code and expected outcome
///
/// The file should have a comment line specifying the expectation (first or last line):
/// - `# Return=type: repr` - Standard return format
/// - `# Raise=ExceptionType('message')` - Exception format
/// - `# ParseError=message` - Parse error format
/// - `# Return.str=value` - Check py_str() output
/// - `# Return.repr=value` - Check py_repr() output
/// - `# Return.type=typename` - Check py_type() output
/// - `# Return.is_none=true|false` - Check is_none() result
/// - `# Return.is_ellipsis=true|false` - Check is_ellipsis() result
fn parse_fixture(content: &str) -> (String, Expectation) {
    let lines: Vec<&str> = content.lines().collect();

    // Find the expectation comment line (check first and last lines)
    let (expectation_line, code_lines) = if let Some(first_line) = lines.first() {
        if first_line.starts_with("# Return")
            || first_line.starts_with("# Raise")
            || first_line.starts_with("# ParseError")
        {
            // Expectation is on first line, code is rest
            (first_line, &lines[1..])
        } else if let Some(last_line) = lines.last() {
            // Expectation is on last line, code is everything else
            (last_line, &lines[..lines.len() - 1])
        } else {
            panic!("Empty fixture file");
        }
    } else {
        panic!("Empty fixture file");
    };

    // Parse expectation from comment line
    let expectation = if let Some(expected) = expectation_line.strip_prefix("# Return=") {
        Expectation::Return(expected.to_string())
    } else if let Some(expected) = expectation_line.strip_prefix("# Return.str=") {
        Expectation::ReturnStr(expected.to_string())
    } else if let Some(expected) = expectation_line.strip_prefix("# Return.repr=") {
        Expectation::ReturnRepr(expected.to_string())
    } else if let Some(expected) = expectation_line.strip_prefix("# Return.type=") {
        Expectation::ReturnType(expected.to_string())
    } else if let Some(expected) = expectation_line.strip_prefix("# Return.is_none=") {
        let is_none = expected.trim() == "true";
        Expectation::ReturnIsNone(is_none)
    } else if let Some(expected) = expectation_line.strip_prefix("# Return.is_ellipsis=") {
        let is_ellipsis = expected.trim() == "true";
        Expectation::ReturnIsEllipsis(is_ellipsis)
    } else if let Some(expected) = expectation_line.strip_prefix("# Raise=") {
        Expectation::Raise(expected.to_string())
    } else if let Some(expected) = expectation_line.strip_prefix("# ParseError=") {
        Expectation::ParseError(expected.to_string())
    } else {
        panic!("Invalid expectation format in comment line: {expectation_line}");
    };

    // Code is everything except the expectation comment line
    let code = code_lines.join("\n");

    (code, expectation)
}

/// Run a test with the given code and expectation
///
/// This function executes Python code via the Executor and validates the result
/// against the expected outcome specified in the fixture.
fn run_test(path: &Path, code: &str, expectation: Expectation) {
    let test_name = path.strip_prefix("test_cases/").unwrap_or(path).display().to_string();

    match Executor::new(code, "test.py", &[]) {
        Ok(mut ex) => match ex.run(vec![]) {
            Ok(Exit::Return(obj)) => match expectation {
                Expectation::Return(expected) => {
                    let output = format!("{}: {}", obj.py_type(), obj.py_repr());
                    assert_eq!(output, expected, "[{test_name}] Return value mismatch");
                }
                Expectation::ReturnStr(expected) => {
                    let output = obj.py_str();
                    assert_eq!(output.as_ref(), expected, "[{test_name}] py_str() mismatch");
                }
                Expectation::ReturnRepr(expected) => {
                    let output = obj.py_repr();
                    assert_eq!(output.as_ref(), expected, "[{test_name}] py_repr() mismatch");
                }
                Expectation::ReturnType(expected) => {
                    let output = obj.py_type();
                    assert_eq!(output, expected, "[{test_name}] py_type() mismatch");
                }
                Expectation::ReturnIsNone(expected) => {
                    let output = obj.is_none();
                    assert_eq!(output, expected, "[{test_name}] is_none() mismatch");
                }
                Expectation::ReturnIsEllipsis(expected) => {
                    let output = obj.is_ellipsis();
                    assert_eq!(output, expected, "[{test_name}] is_ellipsis() mismatch");
                }
                _ => panic!("[{test_name}] Expected return, got different expectation type"),
            },
            Ok(Exit::Raise(exc)) => {
                if let Expectation::Raise(expected) = expectation {
                    let output = format!("{}", exc.exc);
                    assert_eq!(output, expected, "[{test_name}] Exception mismatch");
                } else {
                    panic!("[{test_name}] Unexpected exception: {exc:?}");
                }
            }
            Err(e) => panic!("[{test_name}] Runtime error: {e:?}"),
        },
        Err(parse_err) => {
            if let Expectation::ParseError(expected) = expectation {
                let err_msg = parse_err.summary();
                assert_eq!(err_msg, expected, "[{test_name}] Parse error mismatch");
            } else {
                panic!("[{test_name}] Unexpected parse error: {parse_err:?}");
            }
        }
    }
}

/// Test function that runs for each Python fixture file
fn run_fixture_test(path: &Path) -> Result<(), Box<dyn Error>> {
    let content = fs::read_to_string(path)?;

    let (code, expectation) = parse_fixture(&content);
    run_test(path, &code, expectation);
    Ok(())
}

// Generate tests for all fixture files using datatest-stable harness macro
// All fixtures are now in a flat structure with group prefixes (e.g., id__is_test.py)
datatest_stable::harness!(run_fixture_test, "test_cases", r"^.*\.py$");
