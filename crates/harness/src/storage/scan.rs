//! Scan positions, cursors and pages shared by the built-in backends.

use delta_chord::json::{Json, Map};

use super::error::StorageError;
use crate::records::{Cursor, Page, ScanOrder};

/// Where a scan starts: its order and the last id a previous page returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScanStart {
    pub(crate) order: ScanOrder,
    pub(crate) after: Option<u64>,
    /// `limit + 1`: a page reads one item more to know whether another page follows.
    pub(crate) fetch: usize,
}

impl ScanStart {
    /// Resolves a scan's order and position. A cursor continues in the order it was created
    /// with, whether the query repeats that order or omits it; a different order is refused.
    /// A cursor without an order continues in the scan's default order (`fallback`).
    pub(crate) fn new(
        requested: Option<ScanOrder>,
        cursor: Option<&Cursor>,
        fallback: ScanOrder,
        limit: usize,
    ) -> Result<Self, StorageError> {
        if limit == 0 {
            return Err(StorageError::Request("Scan limit must be positive".to_owned()));
        }
        let fetch = limit.saturating_add(1);
        let Some(Cursor(cursor)) = cursor else {
            return Ok(Self {
                order: requested.unwrap_or(fallback),
                after: None,
                fetch,
            });
        };
        let invalid = || StorageError::Request("Invalid storage cursor".to_owned());
        let after = cursor.get("after").and_then(Json::as_u64).ok_or_else(invalid)?;
        let order = match cursor.get("order").map(Json::as_str) {
            None => fallback,
            Some(Some("ascending")) => ScanOrder::Ascending,
            Some(Some("descending")) => ScanOrder::Descending,
            Some(_) => return Err(invalid()),
        };
        if let Some(requested) = requested
            && requested != order
        {
            return Err(StorageError::Request(format!(
                "The cursor continues a {} scan; the query asks for {}",
                order.as_str(),
                requested.as_str()
            )));
        }
        Ok(Self {
            order,
            after: Some(after),
            fetch,
        })
    }

    pub(crate) fn limit(&self) -> usize {
        self.fetch - 1
    }

    /// The page of `items` (at most `fetch` of them, in scan order).
    pub(crate) fn page<T>(&self, mut items: Vec<T>, id: impl Fn(&T) -> u64) -> Page<T> {
        let limit = self.limit();
        if items.len() <= limit {
            return Page { items, next: None };
        }
        items.truncate(limit);
        let next = items.last().map(|last| cursor(id(last), self.order));
        Page { items, next }
    }
}

/// The continuation of a scan whose last returned item has `id`.
fn cursor(after: u64, order: ScanOrder) -> Cursor {
    let mut map = Map::new();
    map.insert("after".into(), Json::from(after));
    map.insert("order".into(), Json::from(order.as_str()));
    Cursor(Json::from(map))
}

#[cfg(test)]
mod tests {
    use delta_chord::json;

    use super::*;

    #[test]
    fn cursors_carry_their_order() {
        let start = ScanStart::new(None, None, ScanOrder::Descending, 2).unwrap();
        let page = start.page(vec![5_u64, 4, 3], |id| *id);
        assert_eq!(page.items, vec![5, 4]);
        let next = page.next.unwrap();
        assert_eq!(next.0, json!({ "after": 4, "order": "descending" }));

        let resumed = ScanStart::new(None, Some(&next), ScanOrder::Ascending, 2).unwrap();
        assert_eq!(resumed.order, ScanOrder::Descending);
        assert_eq!(resumed.after, Some(4));
        let error = ScanStart::new(Some(ScanOrder::Ascending), Some(&next), ScanOrder::Ascending, 2).unwrap_err();
        assert_eq!(
            error.to_string(),
            "The cursor continues a descending scan; the query asks for ascending"
        );
    }

    #[test]
    fn malformed_cursors_and_limits_are_requests() {
        for cursor in [json!({}), json!({ "after": "1" }), json!({ "after": 1, "order": "up" })] {
            let error = ScanStart::new(None, Some(&Cursor(cursor)), ScanOrder::Ascending, 1).unwrap_err();
            assert!(error.is_request(), "{error}");
        }
        assert!(
            ScanStart::new(None, None, ScanOrder::Ascending, 0)
                .unwrap_err()
                .is_request()
        );
        let legacy = ScanStart::new(None, Some(&Cursor(json!({ "after": 3 }))), ScanOrder::Descending, 1).unwrap();
        assert_eq!(legacy.order, ScanOrder::Descending);
    }
}
