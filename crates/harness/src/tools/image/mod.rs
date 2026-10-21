//! Images in `read`: detection, header dimensions, the model's limits, and the
//! [`ImageProcessor`] seam that resizes and converts what does not fit.

mod detect;
mod dimensions;

use std::fmt;

use delta_ai::model::Model;
use delta_chord::context::Context;
use futures::future::BoxFuture;

pub(super) use detect::detect;
pub use detect::detect_image_type;
pub use dimensions::image_dimensions;

/// Formats every provider takes inline; others (BMP) need converting.
pub const INLINE_IMAGE_TYPES: [&str; 4] = ["image/png", "image/jpeg", "image/gif", "image/webp"];

/// Whether providers take `mime` inline.
pub fn is_inline(mime: &str) -> bool {
    INLINE_IMAGE_TYPES.contains(&mime)
}

/// An image's size in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dimensions {
    pub width: u32,
    pub height: u32,
}

/// What an image the model sees must keep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageLimits {
    pub max_width: u32,
    pub max_height: u32,
    /// Bytes of base64.
    pub max_bytes: u64,
}

/// The coding agent's limits: 2000 × 2000 pixels, 4.5 MiB of base64 (below Anthropic's 5 MB).
pub const DEFAULT_IMAGE_LIMITS: ImageLimits = ImageLimits {
    max_width: 2000,
    max_height: 2000,
    max_bytes: 4_718_592,
};

impl ImageLimits {
    /// The model's `input_limits.images.resize`, field by field over the defaults; the
    /// defaults without a model.
    pub fn for_model(model: Option<&Model>) -> Self {
        let resize = model
            .and_then(|model| model.input_limits)
            .and_then(|limits| limits.images)
            .and_then(|images| images.resize)
            .unwrap_or_default();
        Self {
            max_width: resize.max_width.unwrap_or(DEFAULT_IMAGE_LIMITS.max_width),
            max_height: resize.max_height.unwrap_or(DEFAULT_IMAGE_LIMITS.max_height),
            max_bytes: resize.max_bytes.unwrap_or(DEFAULT_IMAGE_LIMITS.max_bytes),
        }
    }
}

/// The length of the base64 of `bytes` bytes.
pub fn base64_len(bytes: u64) -> u64 {
    bytes.div_ceil(3) * 4
}

/// An image the model can take: a format providers take inline, within the limits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedImage {
    /// The encoded image; `read` sends its base64.
    pub data: Vec<u8>,
    pub mime: String,
    /// The size before and after, when the image was resized, so the model can map
    /// coordinates back.
    pub resized: Option<(Dimensions, Dimensions)>,
    /// The original format, when it differs from `mime`.
    pub converted_from: Option<String>,
}

/// Decodes, orients, resizes and re-encodes images so the model can take them. `read` uses
/// one when given; without one, it passes inline images within the limits through as they
/// are.
pub trait ImageProcessor: Send + Sync + fmt::Debug {
    /// The image prepared to fit `limits`, or `None` when it cannot be decoded or made to fit.
    fn prepare<'a>(
        &'a self,
        bytes: &'a [u8],
        mime: &'a str,
        limits: ImageLimits,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Option<PreparedImage>>;
}

#[cfg(test)]
mod tests {
    use delta_ai::model::{ModelImageInputLimits, ModelImageResizeOptions, ModelInputLimits};

    use super::*;

    #[test]
    fn base64_lengths_round_up() {
        assert_eq!(base64_len(0), 0);
        assert_eq!(base64_len(1), 4);
        assert_eq!(base64_len(3), 4);
        assert_eq!(base64_len(4), 8);
    }

    #[test]
    fn model_limits_override_field_by_field() {
        assert_eq!(ImageLimits::for_model(None), DEFAULT_IMAGE_LIMITS);
        let mut model =
            delta_ai::providers::faux::FauxProvider::new(delta_ai::providers::faux::FauxOptions::default()).model();
        model.input_limits = Some(ModelInputLimits {
            images: Some(ModelImageInputLimits {
                resize: Some(ModelImageResizeOptions {
                    max_width: Some(800),
                    max_bytes: Some(1000),
                    ..ModelImageResizeOptions::default()
                }),
                ..ModelImageInputLimits::default()
            }),
            ..ModelInputLimits::default()
        });
        let limits = ImageLimits::for_model(Some(&model));
        assert_eq!(
            (limits.max_width, limits.max_height, limits.max_bytes),
            (800, 2000, 1000)
        );
    }
}
