use std::borrow::Cow;

/// Trait for handling output from the `print()` builtin function.
///
/// Implement this trait to capture or redirect print output from sandboxed Python code.
/// The default implementation `StdPrint` writes to stdout.
pub trait PrintWriter {
    /// Called once for each formatted argument passed to `print()`.
    ///
    /// This method is responsible for writing only the given argument's text, and must
    /// not add separators or a trailing newline. Separators (such as spaces) and the
    /// final terminator (such as a newline) are emitted via [`stdout_push`].
    ///
    /// # Arguments
    /// * `output` - The formatted output string for a single argument (without
    ///   separators or trailing newline).
    fn stdout_write(&mut self, output: Cow<'_, str>);

    /// Add a single character to stdout.
    ///
    /// Generally called to add spaces and newlines within print output.
    ///
    /// # Arguments
    /// * `end` - The character to print after the formatted output.
    fn stdout_push(&mut self, end: char);
}

/// Default `PrintWriter` that writes to stdout.
///
/// This is the default writer used when no custom writer is provided.
#[derive(Debug)]
pub struct StdPrint;

impl PrintWriter for StdPrint {
    fn stdout_write(&mut self, output: Cow<'_, str>) {
        print!("{output}");
    }

    fn stdout_push(&mut self, end: char) {
        print!("{end}");
    }
}

/// A `PrintWriter` that collects all output into a string.
///
/// Uses interior mutability via `RefCell` to allow collecting output
/// while being passed as a shared reference through the execution stack.
///
/// Useful for testing or capturing print output programmatically.
#[derive(Debug, Default)]
pub struct CollectStringPrint(String);

impl CollectStringPrint {
    /// Creates a new empty `CollectStringPrint`.
    #[must_use]
    pub fn new() -> Self {
        Self(String::new())
    }

    /// Returns the collected output as a string slice.
    ///
    /// # Panics
    /// Panics if the internal RefCell is currently borrowed mutably.
    #[must_use]
    pub fn output(&self) -> &str {
        self.0.as_str()
    }

    /// Consumes the writer and returns the collected output.
    #[must_use]
    pub fn into_output(self) -> String {
        self.0
    }
}

impl PrintWriter for CollectStringPrint {
    fn stdout_write(&mut self, output: Cow<'_, str>) {
        self.0.push_str(&output);
    }

    fn stdout_push(&mut self, end: char) {
        self.0.push(end);
    }
}

/// `PrintWriter` that ignores all output.
///
/// Useful for suppressing print output during testing or benchmarking.
#[derive(Debug, Default)]
pub struct NoPrint;

impl PrintWriter for NoPrint {
    fn stdout_write(&mut self, _output: Cow<'_, str>) {}

    fn stdout_push(&mut self, _end: char) {}
}
