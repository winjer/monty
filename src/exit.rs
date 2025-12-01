use std::{borrow::Cow, fmt};

use crate::heap::HeapData;
use crate::values::PyValue;
use crate::{exceptions::ExceptionRaise, expressions::FrameExit, heap::Heap, object::Object};

#[derive(Debug)]
pub enum Exit<'c, 'e> {
    Return(Value<'e>),
    // Yield(ReturnObject<'e>),
    Raise(ExceptionRaise<'c>),
}

impl fmt::Display for Exit<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Return(v) => write!(f, "{v}"),
            Self::Raise(exc) => write!(f, "{exc}"),
        }
    }
}

impl<'c, 'e> Exit<'c, 'e>
where
    'c: 'e,
{
    pub(crate) fn new(frame_exit: FrameExit<'c, 'e>, heap: Heap<'e>) -> Self {
        match frame_exit {
            FrameExit::Return(object) => Self::Return(Value { object, heap }),
            FrameExit::Raise(exc) => Self::Raise(exc),
        }
    }

    pub fn value(self) -> Result<Value<'e>, ConversionError> {
        match self {
            Self::Return(value) => Ok(value),
            Self::Raise(_) => Err(ConversionError::new("value", "raise")),
        }
    }
}

#[derive(Debug)]
pub struct Value<'e> {
    object: Object<'e>,
    heap: Heap<'e>,
}

impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.py_str())
    }
}

impl<'e> Value<'e> {
    /// User facing representation of the object, should match python's `str(object)`
    #[must_use]
    pub fn py_str(&'e self) -> Cow<'e, str> {
        self.object.py_str(&self.heap)
    }

    /// Debug representation of the object, should match python's `repr(object)`
    #[must_use]
    pub fn py_repr(&'e self) -> Cow<'e, str> {
        self.object.py_repr(&self.heap)
    }

    /// User facing representation of the object type, should roughly match `str(type(object))
    #[must_use]
    pub fn py_type(&self) -> &'static str {
        self.object.py_type(&self.heap)
    }

    /// Checks if the return object is None
    #[must_use]
    pub fn is_none(&self) -> bool {
        matches!(self.object, Object::None)
    }

    /// Checks if the return object is Ellipsis
    #[must_use]
    pub fn is_ellipsis(&self) -> bool {
        matches!(self.object, Object::Ellipsis)
    }
}

/// Conversion error type for failed conversions from ReturnObject
#[derive(Debug)]
pub struct ConversionError {
    pub expected: &'static str,
    pub actual: &'static str,
}

impl ConversionError {
    pub fn new(expected: &'static str, actual: &'static str) -> Self {
        Self { expected, actual }
    }
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "expected {}, got {}", self.expected, self.actual)
    }
}

impl std::error::Error for ConversionError {}

/// Attempts to convert a ReturnObject to an i64 integer.
/// Returns an error if the object is not an Int variant.
impl<'e> TryFrom<&Value<'e>> for i64 {
    type Error = ConversionError;

    fn try_from(value: &Value<'e>) -> Result<Self, Self::Error> {
        match value.object {
            Object::Int(i) => Ok(i),
            _ => Err(ConversionError::new("int", value.py_type())),
        }
    }
}

/// Attempts to convert a ReturnObject to an f64 float.
/// Returns an error if the object is not a Float or Int variant.
/// Int values are automatically converted to f64.
impl<'e> TryFrom<&Value<'e>> for f64 {
    type Error = ConversionError;

    fn try_from(value: &Value<'e>) -> Result<Self, Self::Error> {
        match value.object {
            Object::Float(f) => Ok(f),
            Object::Int(i) => Ok(i as f64),
            _ => Err(ConversionError::new("float", value.py_type())),
        }
    }
}

/// Attempts to convert a ReturnObject to a String.
/// Returns an error if the object is not a heap-allocated Str variant.
impl<'e> TryFrom<&Value<'e>> for String {
    type Error = ConversionError;

    fn try_from(value: &Value<'e>) -> Result<Self, Self::Error> {
        match value.object {
            Object::InternString(s) => return Ok(s.to_owned()),
            Object::Ref(id) => {
                if let HeapData::Str(s) = value.heap.get(id) {
                    return Ok(s.clone().into());
                }
            }
            _ => {}
        }
        Err(ConversionError::new("str", value.py_type()))
    }
}

/// Attempts to convert a ReturnObject to a bool.
/// Returns an error if the object is not a True or False variant.
/// Note: This does NOT use Python's truthiness rules (use Object::bool for that).
impl<'e> TryFrom<&Value<'e>> for bool {
    type Error = ConversionError;

    fn try_from(value: &Value<'e>) -> Result<Self, Self::Error> {
        match value.object {
            Object::Bool(b) => Ok(b),
            _ => Err(ConversionError::new("bool", value.py_type())),
        }
    }
}
