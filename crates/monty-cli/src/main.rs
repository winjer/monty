use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use monty::{ExecProgress, ExecutorIter, StdPrint};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let file_path = if args.len() > 1 { &args[1] } else { "monty.py" };
    let code = match read_file(file_path) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let input_names = vec![];
    let inputs = vec![];
    let ext_functions = vec![];

    let ex = match ExecutorIter::new(&code, file_path, &input_names, ext_functions) {
        Ok(ex) => ex,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    let start = Instant::now();
    match ex.run_no_limits(inputs, &mut StdPrint) {
        Ok(ExecProgress::Complete(value)) => {
            let elapsed = start.elapsed();
            eprintln!("{elapsed:?}, output: {value}");
            ExitCode::SUCCESS
        }
        Ok(ExecProgress::FunctionCall {
            function_name, args, ..
        }) => {
            let elapsed = start.elapsed();
            eprintln!(
                "{elapsed:?}, external function call: {function_name}({args:?}) - no host to provide return value"
            );
            ExitCode::FAILURE
        }
        Err(err) => {
            let elapsed = start.elapsed();
            eprintln!("{elapsed:?}, error: {err}");
            ExitCode::FAILURE
        }
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
