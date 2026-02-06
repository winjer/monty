//! Parse-related type definitions used throughout the Monty runtime.
//!
//! This module defines types produced by the parser that are also referenced
//! by the runtime (serialized in bytecode, used in error reporting, etc.).
//! The actual ruff-dependent parser logic lives in `parser.rs` behind the
//! `parser` feature flag.

use std::{borrow::Cow, fmt};

use crate::{
    StackFrame,
    exception_private::ExcType,
    exception_public::{CodeLoc, MontyException},
    expressions::ExprLoc,
    intern::StringId,
};

/// A parameter in a function signature with optional default value.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParsedParam {
    /// The parameter name.
    pub name: StringId,
    /// The default value expression (evaluated at definition time).
    pub default: Option<ExprLoc>,
}

/// A parsed function signature with all parameter types.
///
/// This intermediate representation captures the structure of Python function
/// parameters before name resolution. Default value expressions are stored
/// as unevaluated AST and will be evaluated during the prepare phase.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ParsedSignature {
    /// Positional-only parameters (before `/`).
    pub pos_args: Vec<ParsedParam>,
    /// Positional-or-keyword parameters.
    pub args: Vec<ParsedParam>,
    /// Variable positional parameter (`*args`).
    pub var_args: Option<StringId>,
    /// Keyword-only parameters (after `*` or `*args`).
    pub kwargs: Vec<ParsedParam>,
    /// Variable keyword parameter (`**kwargs`).
    pub var_kwargs: Option<StringId>,
}

impl ParsedSignature {
    /// Returns an iterator over all parameter names in the signature.
    ///
    /// Order: pos_args, args, var_args, kwargs, var_kwargs
    pub fn param_names(&self) -> impl Iterator<Item = StringId> + '_ {
        self.pos_args
            .iter()
            .map(|p| p.name)
            .chain(self.args.iter().map(|p| p.name))
            .chain(self.var_args.iter().copied())
            .chain(self.kwargs.iter().map(|p| p.name))
            .chain(self.var_kwargs.iter().copied())
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Try<N> {
    pub body: Vec<N>,
    pub handlers: Vec<ExceptHandler<N>>,
    pub or_else: Vec<N>,
    pub finally: Vec<N>,
}

/// A parsed exception handler (except clause).
///
/// Represents `except ExcType as name:` or bare `except:` clauses.
/// The exception type and variable binding are both optional.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExceptHandler<N> {
    /// Exception type(s) to catch. None = bare except (catches all).
    pub exc_type: Option<ExprLoc>,
    /// Variable name for `except X as e:`. None = no binding.
    pub name: Option<crate::expressions::Identifier>,
    /// Handler body statements.
    pub body: Vec<N>,
}

/// Source code location information for error reporting.
///
/// Contains filename (as StringId), line/column positions, and optionally a line number for
/// extracting the preview line from source during traceback formatting.
///
/// To display the filename, the caller must provide access to the string storage.
#[derive(Clone, Copy, Default, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CodeRange {
    /// Interned filename ID - look up in Interns to get the actual string.
    pub filename: StringId,
    /// Line number (0-indexed) for extracting preview from source. None if range spans multiple lines.
    preview_line: Option<u32>,
    start: CodeLoc,
    end: CodeLoc,
}

/// Custom Debug implementation to make displaying code much less verbose.
impl fmt::Debug for CodeRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CodeRange{{filename: {:?}, start: {:?}, end: {:?}}}",
            self.filename, self.start, self.end
        )
    }
}

impl CodeRange {
    /// Creates a new CodeRange with the given positions.
    pub(crate) fn new(filename: StringId, start: CodeLoc, end: CodeLoc, preview_line: Option<u32>) -> Self {
        Self {
            filename,
            preview_line,
            start,
            end,
        }
    }

    /// Returns the start position.
    #[must_use]
    pub fn start(&self) -> CodeLoc {
        self.start
    }

    /// Returns the end position.
    #[must_use]
    pub fn end(&self) -> CodeLoc {
        self.end
    }

    /// Returns the preview line number (0-indexed) if available.
    #[must_use]
    pub fn preview_line_number(&self) -> Option<u32> {
        self.preview_line
    }
}

/// Errors that can occur during parsing or preparation of Python code.
#[derive(Debug, Clone)]
pub enum ParseError {
    /// Error in syntax
    Syntax {
        msg: Cow<'static, str>,
        position: CodeRange,
    },
    /// Missing feature from Monty, we hope to implement in the future.
    /// Message gets prefixed with "The monty syntax parser does not yet support ".
    NotImplemented {
        msg: Cow<'static, str>,
        position: CodeRange,
    },
    /// Missing feature with a custom full message (no prefix added).
    NotSupported {
        msg: Cow<'static, str>,
        position: CodeRange,
    },
    /// Import error (e.g., relative imports without a package).
    Import {
        msg: Cow<'static, str>,
        position: CodeRange,
    },
}

impl ParseError {
    pub(crate) fn not_implemented(msg: impl Into<Cow<'static, str>>, position: CodeRange) -> Self {
        Self::NotImplemented {
            msg: msg.into(),
            position,
        }
    }

    pub(crate) fn not_supported(msg: impl Into<Cow<'static, str>>, position: CodeRange) -> Self {
        Self::NotSupported {
            msg: msg.into(),
            position,
        }
    }

    pub(crate) fn import_error(msg: impl Into<Cow<'static, str>>, position: CodeRange) -> Self {
        Self::Import {
            msg: msg.into(),
            position,
        }
    }

    pub(crate) fn syntax(msg: impl Into<Cow<'static, str>>, position: CodeRange) -> Self {
        Self::Syntax {
            msg: msg.into(),
            position,
        }
    }
}

impl ParseError {
    pub fn into_python_exc(self, filename: &str, source: &str) -> MontyException {
        match self {
            Self::Syntax { msg, position } => MontyException::new_full(
                ExcType::SyntaxError,
                Some(msg.into_owned()),
                vec![StackFrame::from_position_syntax_error(position, filename, source)],
            ),
            Self::NotImplemented { msg, position } => MontyException::new_full(
                ExcType::NotImplementedError,
                Some(format!("The monty syntax parser does not yet support {msg}")),
                vec![StackFrame::from_position(position, filename, source)],
            ),
            Self::NotSupported { msg, position } => MontyException::new_full(
                ExcType::NotImplementedError,
                Some(msg.into_owned()),
                vec![StackFrame::from_position(position, filename, source)],
            ),
            Self::Import { msg, position } => MontyException::new_full(
                ExcType::ImportError,
                Some(msg.into_owned()),
                vec![StackFrame::from_position_no_caret(position, filename, source)],
            ),
        }
    }
}
