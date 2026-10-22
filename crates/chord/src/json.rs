//! JSON values shared between revisions: an `Arc` copy-on-write tree.
//!
//! Cloning a [`Json`] is O(1). Mutating one copies the shared containers on the mutated path
//! first, so another holder of the value never sees the change. See `docs/design/json.md`.

use std::borrow::Borrow;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use indexmap::IndexMap;
use serde::de::{self, DeserializeOwned, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};

pub use serde_json::Number;

/// An object key.
pub type Key = Arc<str>;

/// The map inside a JSON object. Keeps insertion order.
pub type Map = IndexMap<Key, Json>;

/// A JSON value whose strings and containers are shared until written.
#[derive(Clone, Default)]
pub enum Json {
    #[default]
    Null,
    Bool(bool),
    /// An `i64`, a `u64` or a finite `f64`.
    Number(Number),
    String(Str),
    Array(Array),
    Object(Object),
}

/// The kind of a [`Json`] value, for errors and checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Null,
    Bool,
    Number,
    String,
    Array,
    Object,
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Null => "null",
            Self::Bool => "boolean",
            Self::Number => "number",
            Self::String => "string",
            Self::Array => "array",
            Self::Object => "object",
        })
    }
}

// ─── Shared strings and containers ───────────────────────────────────────────

/// A shared string.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Str(Arc<String>);

impl Str {
    pub fn new(text: impl Into<String>) -> Self {
        Self(Arc::new(text.into()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The string, copied first when another value shares it.
    pub fn make_mut(&mut self) -> &mut String {
        Arc::make_mut(&mut self.0)
    }

    /// Whether both share one allocation.
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        Arc::ptr_eq(&a.0, &b.0)
    }

    pub fn into_string(self) -> String {
        Arc::try_unwrap(self.0).unwrap_or_else(|shared| (*shared).clone())
    }
}

impl Deref for Str {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl Borrow<str> for Str {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Str {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl From<&str> for Str {
    fn from(text: &str) -> Self {
        Self::new(text)
    }
}

impl From<String> for Str {
    fn from(text: String) -> Self {
        Self::new(text)
    }
}

/// A shared array.
#[derive(Clone, Default)]
pub struct Array(Arc<Vec<Json>>);

impl Array {
    pub fn new(items: Vec<Json>) -> Self {
        Self(Arc::new(items))
    }

    /// The elements, copied first (the element handles, not the elements) when another value
    /// shares them.
    pub fn make_mut(&mut self) -> &mut Vec<Json> {
        Arc::make_mut(&mut self.0)
    }

    /// Whether both share one allocation.
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        Arc::ptr_eq(&a.0, &b.0)
    }

    pub fn into_vec(self) -> Vec<Json> {
        Arc::try_unwrap(self.0).unwrap_or_else(|shared| (*shared).clone())
    }
}

impl Deref for Array {
    type Target = [Json];

    fn deref(&self) -> &[Json] {
        &self.0
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl From<Vec<Json>> for Array {
    fn from(items: Vec<Json>) -> Self {
        Self::new(items)
    }
}

impl FromIterator<Json> for Array {
    fn from_iter<I: IntoIterator<Item = Json>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

/// A shared object.
#[derive(Clone, Default)]
pub struct Object(Arc<Map>);

impl Object {
    pub fn new(map: Map) -> Self {
        Self(Arc::new(map))
    }

    /// The entries, copied first (the value handles, not the values) when another value shares
    /// them. Remove keys with `shift_remove` to keep the order of the others.
    pub fn make_mut(&mut self) -> &mut Map {
        Arc::make_mut(&mut self.0)
    }

    /// Whether both share one allocation.
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        Arc::ptr_eq(&a.0, &b.0)
    }

    pub fn into_map(self) -> Map {
        Arc::try_unwrap(self.0).unwrap_or_else(|shared| (*shared).clone())
    }
}

impl Deref for Object {
    type Target = Map;

    fn deref(&self) -> &Map {
        &self.0
    }
}

impl fmt::Debug for Object {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter().map(|(k, v)| (&**k, v))).finish()
    }
}

impl From<Map> for Object {
    fn from(map: Map) -> Self {
        Self::new(map)
    }
}

impl<K: Into<Key>> FromIterator<(K, Json)> for Object {
    fn from_iter<I: IntoIterator<Item = (K, Json)>>(iter: I) -> Self {
        Self::new(iter.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }
}

// ─── Json ────────────────────────────────────────────────────────────────────

impl Json {
    /// A number, or `None` for NaN and infinities.
    pub fn from_f64(value: f64) -> Option<Self> {
        Number::from_f64(value).map(Self::Number)
    }

    pub fn kind(&self) -> Kind {
        match self {
            Self::Null => Kind::Null,
            Self::Bool(_) => Kind::Bool,
            Self::Number(_) => Kind::Number,
            Self::String(_) => Kind::String,
            Self::Array(_) => Kind::Array,
            Self::Object(_) => Kind::Object,
        }
    }

    /// Identity: the same shared string or container, or equal scalars. Unchanged subtrees of
    /// two revisions are the same; this is how comparisons and diffs skip them in O(1).
    pub fn same(a: &Self, b: &Self) -> bool {
        match (a, b) {
            (Self::String(a), Self::String(b)) => Str::ptr_eq(a, b),
            (Self::Array(a), Self::Array(b)) => Array::ptr_eq(a, b),
            (Self::Object(a), Self::Object(b)) => Object::ptr_eq(a, b),
            (Self::Null, Self::Null) => true,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Number(a), Self::Number(b)) => number_eq(a, b),
            _ => false,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_number(&self) -> Option<&Number> {
        match self {
            Self::Number(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        self.as_number().and_then(Number::as_f64)
    }

    pub fn as_i64(&self) -> Option<i64> {
        self.as_number().and_then(Number::as_i64)
    }

    pub fn as_u64(&self) -> Option<u64> {
        self.as_number().and_then(Number::as_u64)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Array> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&Object> {
        match self {
            Self::Object(value) => Some(value),
            _ => None,
        }
    }

    /// The value at `key` of an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object().and_then(|object| object.get(key))
    }

    /// The element at `index` of an array.
    pub fn get_index(&self, index: usize) -> Option<&Json> {
        self.as_array().and_then(|array| array.get(index))
    }

    /// The value at `seg`: a key of an object or an index of an array.
    pub fn child(&self, seg: &Seg) -> Option<&Json> {
        match (self, seg) {
            (Self::Object(object), Seg::Key(key)) => object.get(&**key),
            (Self::Array(array), Seg::Index(index)) => array.get(*index),
            _ => None,
        }
    }

    /// The value at `path`.
    pub fn pointer(&self, path: &[Seg]) -> Option<&Json> {
        path.iter().try_fold(self, |value, seg| value.child(seg))
    }

    /// The value at `seg`, with this container copied first when it is shared.
    pub fn child_mut(&mut self, seg: &Seg) -> Option<&mut Json> {
        match (self, seg) {
            (Self::Object(object), Seg::Key(key)) => {
                if !object.contains_key(&**key) {
                    return None;
                }
                object.make_mut().get_mut(&**key)
            }
            (Self::Array(array), Seg::Index(index)) => {
                if *index >= array.len() {
                    return None;
                }
                array.make_mut().get_mut(*index)
            }
            _ => None,
        }
    }

    /// The value at `path`, with every container on the path copied first when it is shared.
    pub fn pointer_mut(&mut self, path: &[Seg]) -> Option<&mut Json> {
        path.iter().try_fold(self, |value, seg| value.child_mut(seg))
    }

    /// Serialize as compact JSON text.
    pub fn to_json_string(&self) -> String {
        serde_json::to_string(self).expect("Json always serializes")
    }
}

fn number_eq(a: &Number, b: &Number) -> bool {
    if let (Some(a), Some(b)) = (a.as_i64(), b.as_i64()) {
        return a == b;
    }
    if let (Some(a), Some(b)) = (a.as_u64(), b.as_u64()) {
        return a == b;
    }
    a.as_f64() == b.as_f64()
}

/// Deep equality. Numbers compare by value (`1 == 1.0`), objects ignore key order, and shared
/// subtrees compare equal at once.
impl PartialEq for Json {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::String(a), Self::String(b)) => Str::ptr_eq(a, b) || a.as_str() == b.as_str(),
            (Self::Array(a), Self::Array(b)) => Array::ptr_eq(a, b) || a[..] == b[..],
            (Self::Object(a), Self::Object(b)) => {
                Object::ptr_eq(a, b)
                    || (a.len() == b.len()
                        && a.iter()
                            .all(|(key, value)| b.get(key).is_some_and(|other| value == other)))
            }
            _ => Self::same(self, other),
        }
    }
}

impl fmt::Debug for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => f.write_str("null"),
            Self::Bool(value) => fmt::Debug::fmt(value, f),
            Self::Number(value) => fmt::Display::fmt(value, f),
            Self::String(value) => fmt::Debug::fmt(value, f),
            Self::Array(value) => fmt::Debug::fmt(value, f),
            Self::Object(value) => fmt::Debug::fmt(value, f),
        }
    }
}

/// Compact JSON text.
impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_json_string())
    }
}

// ─── Conversions ─────────────────────────────────────────────────────────────

impl From<bool> for Json {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

macro_rules! from_integer {
    ($($t:ty),*) => {$(
        impl From<$t> for Json {
            fn from(value: $t) -> Self {
                Self::Number(Number::from(value))
            }
        }
    )*};
}

from_integer!(i8, i16, i32, i64, u8, u16, u32, u64, usize, isize);

impl From<&str> for Json {
    fn from(value: &str) -> Self {
        Self::String(Str::new(value))
    }
}

impl From<String> for Json {
    fn from(value: String) -> Self {
        Self::String(Str::new(value))
    }
}

impl From<Str> for Json {
    fn from(value: Str) -> Self {
        Self::String(value)
    }
}

impl From<Array> for Json {
    fn from(value: Array) -> Self {
        Self::Array(value)
    }
}

impl From<Vec<Json>> for Json {
    fn from(value: Vec<Json>) -> Self {
        Self::Array(Array::new(value))
    }
}

impl From<Object> for Json {
    fn from(value: Object) -> Self {
        Self::Object(value)
    }
}

impl From<Map> for Json {
    fn from(value: Map) -> Self {
        Self::Object(Object::new(value))
    }
}

impl From<serde_json::Value> for Json {
    fn from(value: serde_json::Value) -> Self {
        use serde_json::Value;
        match value {
            Value::Null => Self::Null,
            Value::Bool(value) => Self::Bool(value),
            Value::Number(value) => Self::Number(value),
            Value::String(value) => Self::String(Str::new(value)),
            Value::Array(items) => Self::Array(items.into_iter().map(Self::from).collect()),
            Value::Object(map) => Self::Object(map.into_iter().map(|(k, v)| (Key::from(k), Self::from(v))).collect()),
        }
    }
}

impl From<&Json> for serde_json::Value {
    fn from(value: &Json) -> Self {
        use serde_json::Value;
        match value {
            Json::Null => Value::Null,
            Json::Bool(value) => Value::Bool(*value),
            Json::Number(value) => Value::Number(value.clone()),
            Json::String(value) => Value::String(value.as_str().to_owned()),
            Json::Array(items) => Value::Array(items.iter().map(Self::from).collect()),
            Json::Object(map) => Value::Object(map.iter().map(|(k, v)| (k.to_string(), Self::from(v))).collect()),
        }
    }
}

impl From<Json> for serde_json::Value {
    fn from(value: Json) -> Self {
        Self::from(&value)
    }
}

/// Serialize a typed value into a [`Json`].
pub fn to_json<T: Serialize + ?Sized>(value: &T) -> Result<Json, serde_json::Error> {
    serde_json::to_value(value).map(Json::from)
}

/// Deserialize a typed value from a [`Json`].
pub fn from_json<T: DeserializeOwned>(value: &Json) -> Result<T, serde_json::Error> {
    serde_json::from_value(serde_json::Value::from(value))
}

/// Parse JSON text.
pub fn from_str(text: &str) -> Result<Json, serde_json::Error> {
    serde_json::from_str(text)
}

/// Build a [`Json`] with `serde_json::json!` syntax.
#[macro_export]
macro_rules! json {
    ($($json:tt)+) => {
        $crate::json::Json::from($crate::__private::serde_json::json!($($json)+))
    };
}

// ─── Serde ───────────────────────────────────────────────────────────────────

impl Serialize for Json {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Null => serializer.serialize_unit(),
            Self::Bool(value) => serializer.serialize_bool(*value),
            Self::Number(value) => value.serialize(serializer),
            Self::String(value) => serializer.serialize_str(value),
            Self::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items.iter() {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Self::Object(map) => {
                let mut out = serializer.serialize_map(Some(map.len()))?;
                for (key, value) in map.iter() {
                    out.serialize_entry(&**key, value)?;
                }
                out.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(JsonVisitor)
    }
}

struct JsonVisitor;

impl<'de> Visitor<'de> for JsonVisitor {
    type Value = Json;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Json, E> {
        Ok(Json::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Json, E> {
        Ok(Json::from(value))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Json, E> {
        Ok(Json::from(value))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Json, E> {
        Json::from_f64(value).ok_or_else(|| E::custom("non-finite number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Json, E> {
        Ok(Json::from(value))
    }

    fn visit_string<E>(self, value: String) -> Result<Json, E> {
        Ok(Json::from(value))
    }

    fn visit_unit<E>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_none<E>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Json, D::Error> {
        Json::deserialize(deserializer)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
        let mut items = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(4096));
        while let Some(item) = seq.next_element()? {
            items.push(item);
        }
        Ok(Json::from(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Json, A::Error> {
        let mut map = Map::with_capacity(access.size_hint().unwrap_or(0).min(4096));
        while let Some(key) = access.next_key::<String>()? {
            let value = access.next_value()?;
            map.insert(Key::from(key), value);
        }
        Ok(Json::from(map))
    }
}

// ─── Paths ───────────────────────────────────────────────────────────────────

/// One step of a [`Path`]: an object key or an array index.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Seg {
    /// Serialized as a JSON string, even when it looks numeric.
    Key(Key),
    /// Serialized as a JSON number.
    Index(usize),
}

impl Seg {
    pub fn as_key(&self) -> Option<&str> {
        match self {
            Self::Key(key) => Some(key),
            Self::Index(_) => None,
        }
    }

    pub fn as_index(&self) -> Option<usize> {
        match self {
            Self::Key(_) => None,
            Self::Index(index) => Some(*index),
        }
    }
}

impl fmt::Debug for Seg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Key(key) => fmt::Debug::fmt(&**key, f),
            Self::Index(index) => fmt::Debug::fmt(index, f),
        }
    }
}

impl From<&str> for Seg {
    fn from(key: &str) -> Self {
        Self::Key(Key::from(key))
    }
}

impl From<String> for Seg {
    fn from(key: String) -> Self {
        Self::Key(Key::from(key))
    }
}

impl From<Key> for Seg {
    fn from(key: Key) -> Self {
        Self::Key(key)
    }
}

impl From<usize> for Seg {
    fn from(index: usize) -> Self {
        Self::Index(index)
    }
}

impl Serialize for Seg {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Key(key) => serializer.serialize_str(key),
            Self::Index(index) => serializer.serialize_u64(*index as u64),
        }
    }
}

impl<'de> Deserialize<'de> for Seg {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct SegVisitor;

        impl Visitor<'_> for SegVisitor {
            type Value = Seg;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an object key or a non-negative integer index")
            }

            fn visit_str<E>(self, value: &str) -> Result<Seg, E> {
                Ok(Seg::from(value))
            }

            fn visit_string<E>(self, value: String) -> Result<Seg, E> {
                Ok(Seg::from(value))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Seg, E> {
                usize::try_from(value)
                    .map(Seg::Index)
                    .map_err(|_| E::custom("index out of range"))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Seg, E> {
                usize::try_from(value)
                    .map(Seg::Index)
                    .map_err(|_| E::custom("negative index"))
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Seg, E> {
                if value >= 0.0 && value.fract() == 0.0 && value <= 9_007_199_254_740_991.0 {
                    Ok(Seg::Index(value as usize))
                } else {
                    Err(E::custom("index is not a non-negative integer"))
                }
            }
        }

        deserializer.deserialize_any(SegVisitor)
    }
}

/// A location in a JSON value. The empty path is the value itself.
#[derive(Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Path(Vec<Seg>);

impl Path {
    pub fn root() -> Self {
        Self(Vec::new())
    }

    pub fn push(&mut self, seg: impl Into<Seg>) {
        self.0.push(seg.into());
    }

    pub fn pop(&mut self) -> Option<Seg> {
        self.0.pop()
    }

    /// This path extended by `seg`.
    #[must_use]
    pub fn join(&self, seg: impl Into<Seg>) -> Self {
        let mut path = Vec::with_capacity(self.0.len() + 1);
        path.extend_from_slice(&self.0);
        path.push(seg.into());
        Self(path)
    }

    /// This path extended by `suffix`.
    #[must_use]
    pub fn concat(&self, suffix: &[Seg]) -> Self {
        let mut path = Vec::with_capacity(self.0.len() + suffix.len());
        path.extend_from_slice(&self.0);
        path.extend_from_slice(suffix);
        Self(path)
    }

    /// Whether `self` is `other` or below it.
    pub fn starts_with(&self, other: &[Seg]) -> bool {
        self.0.starts_with(other)
    }

    pub fn into_vec(self) -> Vec<Seg> {
        self.0
    }
}

impl Deref for Path {
    type Target = [Seg];

    fn deref(&self) -> &[Seg] {
        &self.0
    }
}

impl From<Vec<Seg>> for Path {
    fn from(segs: Vec<Seg>) -> Self {
        Self(segs)
    }
}

impl From<&[Seg]> for Path {
    fn from(segs: &[Seg]) -> Self {
        Self(segs.to_vec())
    }
}

impl<S: Into<Seg>> FromIterator<S> for Path {
    fn from_iter<I: IntoIterator<Item = S>>(iter: I) -> Self {
        Self(iter.into_iter().map(Into::into).collect())
    }
}

impl fmt::Debug for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.0.iter()).finish()
    }
}

/// The path as a JSON array, for errors and logs.
impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&serde_json::to_string(self).expect("paths always serialize"))
    }
}

#[doc(hidden)]
pub trait IntoSeg {
    fn into_seg(self) -> Seg;
}

impl IntoSeg for &str {
    fn into_seg(self) -> Seg {
        Seg::from(self)
    }
}

impl IntoSeg for String {
    fn into_seg(self) -> Seg {
        Seg::from(self)
    }
}

impl IntoSeg for usize {
    fn into_seg(self) -> Seg {
        Seg::Index(self)
    }
}

/// Integer literals in [`path!`] default to `i32`.
impl IntoSeg for i32 {
    fn into_seg(self) -> Seg {
        Seg::Index(usize::try_from(self).expect("path index must not be negative"))
    }
}

/// Build a [`Path`] from keys and indices: `path!["entries", 0, "text"]`.
#[macro_export]
macro_rules! path {
    () => { $crate::json::Path::root() };
    ($($seg:expr),+ $(,)?) => {
        $crate::json::Path::from(::std::vec![$($crate::json::IntoSeg::into_seg($seg)),+])
    };
}
