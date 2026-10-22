//! [`ImageCrateProcessor`]: an [`ImageProcessor`] on the `image` crate (feature `images`).

use std::io::Cursor;

use delta_chord::context::Context;
use futures::future::BoxFuture;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};

use super::{Dimensions, ImageLimits, ImageProcessor, PreparedImage, base64_len, is_inline};

/// JPEG qualities tried in order.
const JPEG_QUALITIES: [u8; 4] = [80, 70, 55, 40];

/// Prepares images with the `image` crate's PNG, JPEG, GIF, WebP and BMP codecs, on tokio's
/// blocking pool.
///
/// An image already upright, in a format providers take inline and within the limits is
/// returned as it is, once it decodes. Otherwise it is turned upright by its EXIF
/// orientation, scaled to the maximum width and height, and encoded as PNG or JPEG at falling
/// qualities (JPEG first for JPEG sources), then at 3/4 of the size at a time until its
/// base64 fits. BMP always becomes PNG or JPEG. Decoding holds the whole image in memory
/// (4 bytes per pixel for most images).
#[derive(Clone, Copy, Debug, Default)]
pub struct ImageCrateProcessor;

impl ImageProcessor for ImageCrateProcessor {
    fn prepare<'a>(
        &'a self,
        bytes: &'a [u8],
        mime: &'a str,
        limits: ImageLimits,
        ctx: &'a Context,
    ) -> BoxFuture<'a, Option<PreparedImage>> {
        let (bytes, mime) = (bytes.to_vec(), mime.to_owned());
        Box::pin(async move {
            let work = tokio::task::spawn_blocking(move || prepare(&bytes, &mime, limits));
            ctx.run(work).await.ok()?.ok()?
        })
    }
}

fn prepare(bytes: &[u8], mime: &str, limits: ImageLimits) -> Option<PreparedImage> {
    let format = ImageFormat::from_mime_type(mime)?;
    let mut decoder = ImageReader::with_format(Cursor::new(bytes), format)
        .into_decoder()
        .ok()?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    // Decoded even when it is returned as it is: a damaged file is no image, not one the
    // provider rejects.
    let mut image = DynamicImage::from_decoder(decoder).ok()?;
    if is_inline(mime)
        && orientation == Orientation::NoTransforms
        && image.width() <= limits.max_width
        && image.height() <= limits.max_height
        && base64_len(bytes.len() as u64) <= limits.max_bytes
    {
        return Some(PreparedImage {
            data: bytes.to_vec(),
            mime: mime.to_owned(),
            resized: None,
            converted_from: None,
        });
    }
    image.apply_orientation(orientation);
    let original = Dimensions {
        width: image.width(),
        height: image.height(),
    };
    let mut target = fit(original, limits);
    loop {
        if let Some((data, encoded)) = encode_within(&image, target, limits.max_bytes, mime == "image/jpeg") {
            return Some(PreparedImage {
                data,
                converted_from: (encoded != mime).then(|| mime.to_owned()),
                mime: encoded.to_owned(),
                resized: (target != original).then_some((original, target)),
            });
        }
        if target.width == 1 && target.height == 1 {
            return None;
        }
        target = Dimensions {
            width: (target.width * 3 / 4).max(1),
            height: (target.height * 3 / 4).max(1),
        };
    }
}

/// `size` scaled down, keeping its proportions, to fit the limits' width and height.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn fit(size: Dimensions, limits: ImageLimits) -> Dimensions {
    let scale = 1f64
        .min(f64::from(limits.max_width) / f64::from(size.width))
        .min(f64::from(limits.max_height) / f64::from(size.height));
    let scaled = |side: u32| ((f64::from(side) * scale).round() as u32).max(1);
    Dimensions {
        width: scaled(size.width),
        height: scaled(size.height),
    }
}

/// `image` at `size` in the first encoding whose base64 fits `max_bytes`: PNG, then JPEG at
/// falling qualities, or the JPEGs first when `jpeg_first`.
fn encode_within(
    image: &DynamicImage,
    size: Dimensions,
    max_bytes: u64,
    jpeg_first: bool,
) -> Option<(Vec<u8>, &'static str)> {
    let scaled;
    let image = if size.width == image.width() && size.height == image.height() {
        image
    } else {
        scaled = image.resize_exact(size.width, size.height, FilterType::Lanczos3);
        &scaled
    };
    let png = || {
        let mut data = Vec::new();
        image.write_to(&mut Cursor::new(&mut data), ImageFormat::Png).ok()?;
        Some((data, "image/png"))
    };
    let jpeg = |quality: u8| {
        let mut data = Vec::new();
        let rgb = DynamicImage::ImageRgb8(image.to_rgb8());
        rgb.write_with_encoder(JpegEncoder::new_with_quality(&mut data, quality))
            .ok()?;
        Some((data, "image/jpeg"))
    };
    let fits = |encoded: &(Vec<u8>, &str)| base64_len(encoded.0.len() as u64) <= max_bytes;
    if !jpeg_first && let Some(encoded) = png().filter(fits) {
        return Some(encoded);
    }
    for quality in JPEG_QUALITIES {
        if let Some(encoded) = jpeg(quality).filter(fits) {
            return Some(encoded);
        }
    }
    if jpeg_first {
        return png().filter(fits);
    }
    None
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::tools::image::DEFAULT_IMAGE_LIMITS;

    /// A `width` × `height` gradient encoded as `format`.
    fn encoded(width: u32, height: u32, format: ImageFormat) -> Vec<u8> {
        let image = RgbImage::from_fn(width, height, |x, y| {
            Rgb([(x * 7 % 256) as u8, (y * 13 % 256) as u8, ((x + y) % 256) as u8])
        });
        let mut data = Vec::new();
        DynamicImage::ImageRgb8(image)
            .write_to(&mut Cursor::new(&mut data), format)
            .unwrap();
        data
    }

    fn decoded(prepared: &PreparedImage) -> DynamicImage {
        image::load_from_memory(&prepared.data).unwrap()
    }

    fn limits(max_width: u32, max_height: u32, max_bytes: u64) -> ImageLimits {
        ImageLimits {
            max_width,
            max_height,
            max_bytes,
        }
    }

    #[test]
    fn an_inline_image_within_the_limits_passes_through() {
        let bytes = encoded(10, 10, ImageFormat::Png);
        let prepared = prepare(&bytes, "image/png", DEFAULT_IMAGE_LIMITS).unwrap();
        assert_eq!(prepared.data, bytes);
        assert_eq!(
            (prepared.mime.as_str(), prepared.resized, prepared.converted_from),
            ("image/png", None, None)
        );
    }

    #[test]
    fn an_oversized_image_is_scaled_to_fit() {
        let bytes = encoded(400, 200, ImageFormat::Png);
        let prepared = prepare(&bytes, "image/png", limits(100, 100, 1 << 20)).unwrap();
        let size = |width, height| Dimensions { width, height };
        assert_eq!(prepared.resized, Some((size(400, 200), size(100, 50))));
        let image = decoded(&prepared);
        assert_eq!((image.width(), image.height()), (100, 50));
    }

    #[test]
    fn a_bmp_is_converted() {
        let bytes = encoded(4, 4, ImageFormat::Bmp);
        let prepared = prepare(&bytes, "image/bmp", DEFAULT_IMAGE_LIMITS).unwrap();
        assert_eq!(prepared.mime, "image/png");
        assert_eq!(prepared.converted_from.as_deref(), Some("image/bmp"));
        assert_eq!(prepared.resized, None);
    }

    #[test]
    fn too_many_bytes_fall_back_to_jpeg_then_smaller_sizes() {
        let bytes = encoded(300, 300, ImageFormat::Png);
        let budget = base64_len(bytes.len() as u64) / 4;
        let prepared = prepare(&bytes, "image/png", limits(2000, 2000, budget)).unwrap();
        assert!(base64_len(prepared.data.len() as u64) <= budget);
        assert_eq!(prepared.mime, "image/jpeg");
        assert_eq!(prepared.converted_from.as_deref(), Some("image/png"));
    }

    /// `jpeg` with an APP1 Exif segment holding orientation `orientation`, after SOI.
    fn with_orientation(jpeg: &[u8], orientation: u8) -> Vec<u8> {
        let mut tiff = b"MM\x00\x2a\x00\x00\x00\x08".to_vec();
        tiff.extend_from_slice(&[0x00, 0x01]); // one entry
        tiff.extend_from_slice(&[
            0x01,
            0x12,
            0x00,
            0x03,
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            orientation,
            0x00,
            0x00,
        ]);
        tiff.extend_from_slice(&[0; 4]); // no next directory
        let mut payload = b"Exif\x00\x00".to_vec();
        payload.extend_from_slice(&tiff);
        let length = u16::try_from(payload.len() + 2).unwrap();
        let mut bytes = jpeg[..2].to_vec();
        bytes.extend_from_slice(&[0xff, 0xe1]);
        bytes.extend_from_slice(&length.to_be_bytes());
        bytes.extend_from_slice(&payload);
        bytes.extend_from_slice(&jpeg[2..]);
        bytes
    }

    #[test]
    fn the_exif_orientation_is_applied() {
        let bytes = with_orientation(&encoded(40, 20, ImageFormat::Jpeg), 6);
        let prepared = prepare(&bytes, "image/jpeg", DEFAULT_IMAGE_LIMITS).unwrap();
        assert_eq!(prepared.mime, "image/jpeg");
        let image = decoded(&prepared);
        assert_eq!((image.width(), image.height()), (20, 40));
    }

    #[test]
    fn what_cannot_be_decoded_or_fit_is_none() {
        assert_eq!(
            prepare(b"\x89PNG\r\n\x1a\nnot really", "image/png", DEFAULT_IMAGE_LIMITS),
            None
        );
        let bytes = encoded(8, 8, ImageFormat::Png);
        assert_eq!(prepare(&bytes, "image/png", limits(2000, 2000, 1)), None);
    }

    #[tokio::test]
    async fn it_runs_on_the_blocking_pool() {
        let bytes = encoded(4, 4, ImageFormat::Bmp);
        let prepared = ImageCrateProcessor
            .prepare(&bytes, "image/bmp", DEFAULT_IMAGE_LIMITS, &Context::background())
            .await
            .unwrap();
        assert_eq!(prepared.mime, "image/png");
    }
}
