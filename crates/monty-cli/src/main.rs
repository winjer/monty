use std::{env, fs, process::ExitCode, time::Instant};

use monty::{MontyObject, MontyRun, NoLimitTracker, RunProgress, StdPrint};
// disabled due to format failing on https://github.com/pydantic/monty/pull/75 where CI and local wanted imports ordered differently
// TODO re-enabled soon!
#[rustfmt::skip]
use monty_type_checking::{SourceFile, type_check};

const EXT_FUNCTIONS: bool = false;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    // Handle subcommands: compile, run-blob
    if args.len() > 1 {
        match args[1].as_str() {
            "compile" => return compile_command(&args[2..]),
            "run-blob" => return run_blob_command(&args[2..]),
            _ => {}
        }
    }

    let file_path = if args.len() > 1 { &args[1] } else { "example.py" };
    run_python_file(file_path)
}

/// Compiles a Python source file to a serialized bytecode blob.
///
/// Usage: monty compile <input.py> -o <output.monty>
fn compile_command(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("usage: monty compile <input.py> -o <output.monty>");
        return ExitCode::FAILURE;
    }

    let input_path = &args[0];

    // Parse -o flag for output path
    let output_path = if args.len() >= 3 && args[1] == "-o" {
        args[2].clone()
    } else {
        // Default: replace .py extension with .monty
        let path = std::path::Path::new(input_path);
        if path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("py")) {
            path.with_extension("monty").to_string_lossy().into_owned()
        } else {
            format!("{input_path}.monty")
        }
    };

    let code = match read_file(input_path) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    let runner = match MontyRun::new(code, input_path, vec![], vec![]) {
        Ok(ex) => ex,
        Err(err) => {
            eprintln!("error:\n{err}");
            return ExitCode::FAILURE;
        }
    };

    let bytes = match runner.dump() {
        Ok(b) => b,
        Err(err) => {
            eprintln!("serialization error: {err}");
            return ExitCode::FAILURE;
        }
    };

    match fs::write(&output_path, &bytes) {
        Ok(()) => {
            eprintln!("compiled {} -> {} ({} bytes)", input_path, output_path, bytes.len());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error writing {output_path}: {err}");
            ExitCode::FAILURE
        }
    }
}

/// Runs a pre-compiled bytecode blob.
///
/// Usage: monty run-blob <input.monty>
fn run_blob_command(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("usage: monty run-blob <input.monty>");
        return ExitCode::FAILURE;
    }

    let input_path = &args[0];
    let bytes = match fs::read(input_path) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("error reading {input_path}: {err}");
            return ExitCode::FAILURE;
        }
    };

    let runner = match MontyRun::load(&bytes) {
        Ok(r) => r,
        Err(err) => {
            eprintln!("deserialization error: {err}");
            return ExitCode::FAILURE;
        }
    };

    let start = Instant::now();
    let value = match runner.run_no_limits(vec![]) {
        Ok(p) => p,
        Err(err) => {
            let elapsed = start.elapsed();
            eprintln!("error after: {elapsed:?}\n{err}");
            return ExitCode::FAILURE;
        }
    };
    let elapsed = start.elapsed();
    eprintln!("success after: {elapsed:?}\n{value}");
    ExitCode::SUCCESS
}

/// Runs a Python source file (the default behavior).
fn run_python_file(file_path: &str) -> ExitCode {
    let code = match read_file(file_path) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    let start = Instant::now();
    if let Some(failure) = type_check(&SourceFile::new(&code, file_path), None).unwrap() {
        eprintln!("type checking failed:\n{failure}");
    } else {
        eprintln!("type checking succeeded");
    }
    let elapsed = start.elapsed();
    println!("time taken to run typing: {elapsed:?}");

    let input_names = vec![];
    let inputs = vec![];
    let ext_functions = vec!["add_ints".to_owned()];

    let runner = match MontyRun::new(code, file_path, input_names, ext_functions) {
        Ok(ex) => ex,
        Err(err) => {
            eprintln!("error:\n{err}");
            return ExitCode::FAILURE;
        }
    };

    if EXT_FUNCTIONS {
        let start = Instant::now();
        let mut progress = match runner.start(inputs, NoLimitTracker, &mut StdPrint) {
            Ok(p) => p,
            Err(err) => {
                let elapsed = start.elapsed();
                eprintln!("error after: {elapsed:?}\n{err}");
                return ExitCode::FAILURE;
            }
        };

        // Handle external function calls in a loop
        loop {
            match progress {
                RunProgress::Complete(value) => {
                    let elapsed = start.elapsed();
                    eprintln!("success after: {elapsed:?}\n{value}");
                    return ExitCode::SUCCESS;
                }
                RunProgress::FunctionCall {
                    function_name,
                    args,
                    state,
                    ..
                } => {
                    let return_value = if function_name == "add_ints" {
                        // Extract two integer arguments and add them
                        if args.len() != 2 {
                            eprintln!("add_ints requires exactly 2 arguments, got {}", args.len());
                            return ExitCode::FAILURE;
                        }
                        if let (MontyObject::Int(a), MontyObject::Int(b)) = (&args[0], &args[1]) {
                            let ret = MontyObject::Int(a + b);
                            eprintln!("Function call: {function_name}({args:?}) -> {ret:?}");
                            ret
                        } else {
                            eprintln!("add_ints requires integer arguments, got {args:?}");
                            return ExitCode::FAILURE;
                        }
                    } else {
                        let elapsed = start.elapsed();
                        eprintln!("{elapsed:?}, unknown external function: {function_name}({args:?})");
                        return ExitCode::FAILURE;
                    };

                    // Resume execution with the return value
                    match state.run(return_value, &mut StdPrint) {
                        Ok(p) => progress = p,
                        Err(err) => {
                            let elapsed = start.elapsed();
                            eprintln!("error after: {elapsed:?}\n{err}");
                            return ExitCode::FAILURE;
                        }
                    }
                }
                RunProgress::ResolveFutures(state) => {
                    let elapsed = start.elapsed();
                    let pending = state.pending_call_ids();
                    eprintln!("{elapsed:?}, async futures not supported in CLI: {pending:?}");
                    return ExitCode::FAILURE;
                }
                RunProgress::OsCall { function, args, .. } => {
                    let elapsed = start.elapsed();
                    eprintln!("{elapsed:?}, OS calls not supported in CLI: {function:?}({args:?})");
                    return ExitCode::FAILURE;
                }
            }
        }
    } else {
        let start = Instant::now();
        let value = match runner.run_no_limits(inputs) {
            Ok(p) => p,
            Err(err) => {
                let elapsed = start.elapsed();
                eprintln!("error after: {elapsed:?}\n{err}");
                return ExitCode::FAILURE;
            }
        };
        let elapsed = start.elapsed();
        eprintln!("success after: {elapsed:?}\n{value}");
        ExitCode::SUCCESS
    }
}

fn read_file(file_path: &str) -> Result<String, String> {
    eprintln!("Reading file: {file_path}");
    match fs::metadata(file_path) {
        Ok(metadata) => {
            if !metadata.is_file() {
                return Err(format!("Error: {file_path} is not a file"));
            }
        }
        Err(err) => {
            return Err(format!("Error reading {file_path}: {err}"));
        }
    }
    match fs::read_to_string(file_path) {
        Ok(contents) => Ok(contents),
        Err(err) => Err(format!("Error reading file: {err}")),
    }
}
