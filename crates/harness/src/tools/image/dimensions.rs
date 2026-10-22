//! An image's width and height from its header, without decoding it: upstream's
//! `imageDimensions`.

use super::Dimensions;
use super::detect::{byte, u16_be, u16_le, u32_be, u32_le};

/// The dimensions an image of `mime` declares: PNG's IHDR, GIF's logical screen, WebP's
/// `VP8 `, `VP8L` or `VP8X` chunk, JPEG's first SOF segment. `None` when the header does not
/// say.
pub fn image_dimensions(bytes: &[u8], mime: &str) -> Option<Dimensions> {
    let dimensions = |width, height| Some(Dimensions { width, height });
    match mime {
        "image/png" if bytes.len() >= 24 => dimensions(u32_be(bytes, 16), u32_be(bytes, 20)),
        "image/gif" if bytes.len() >= 10 => dimensions(u32::from(u16_le(bytes, 6)), u32::from(u16_le(bytes, 8))),
        "image/webp" if bytes.len() >= 30 => match bytes.get(12..16)? {
            b"VP8 " => dimensions(
                u32::from(u16_le(bytes, 26) & 0x3fff),
                u32::from(u16_le(bytes, 28) & 0x3fff),
            ),
            b"VP8L" => {
                let bits = u32_le(bytes, 21);
                dimensions((bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1)
            }
            b"VP8X" => {
                let u24 = |at: usize| u32::from(u16_le(bytes, at)) + (u32::from(byte(bytes, at + 2)) << 16);
                dimensions(u24(24) + 1, u24(27) + 1)
            }
            _ => None,
        },
        "image/jpeg" => jpeg_dimensions(bytes),
        _ => None,
    }
}

/// The first SOF segment's dimensions, walking the markers from offset 2.
fn jpeg_dimensions(bytes: &[u8]) -> Option<Dimensions> {
    let mut offset = 2;
    while offset + 9 <= bytes.len() {
        if bytes[offset] != 0xff {
            return None;
        }
        let marker = bytes[offset + 1];
        if marker == 0xff {
            // A fill byte.
            offset += 1;
            continue;
        }
        // SOF0 to SOF15, except DHT (C4), JPG (C8) and DAC (CC), which share the range.
        if (0xc0..=0xcf).contains(&marker) && !matches!(marker, 0xc4 | 0xc8 | 0xcc) {
            return Some(Dimensions {
                width: u32::from(u16_be(bytes, offset + 7)),
                height: u32::from(u16_be(bytes, offset + 5)),
            });
        }
        offset += 2 + usize::from(u16_be(bytes, offset + 2));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::super::detect::tests::png;
    use super::*;

    #[allow(clippy::unnecessary_wraps)] // Compared with `image_dimensions`' result.
    fn size(width: u32, height: u32) -> Option<Dimensions> {
        Some(Dimensions { width, height })
    }

    #[test]
    fn png_and_gif() {
        assert_eq!(image_dimensions(&png(300, 20, &[]), "image/png"), size(300, 20));
        assert_eq!(image_dimensions(b"GIF89a\x2c\x01\x14\x00", "image/gif"), size(300, 20));
        assert_eq!(image_dimensions(b"GIF89a", "image/gif"), None);
    }

    fn webp(kind: [u8; 4], payload: &[(usize, &[u8])]) -> Vec<u8> {
        let mut bytes = b"RIFF\0\0\0\0WEBP".to_vec();
        bytes.extend_from_slice(&kind);
        bytes.resize(32, 0);
        for (at, data) in payload {
            bytes[*at..*at + data.len()].copy_from_slice(data);
        }
        bytes
    }

    #[test]
    fn webp_chunks() {
        let lossy = webp(
            *b"VP8 ",
            &[(26, &(0xc000 | 0x0280_u16).to_le_bytes()), (28, &480u16.to_le_bytes())],
        );
        assert_eq!(image_dimensions(&lossy, "image/webp"), size(640, 480));
        let bits: u32 = 0x027f | (0x01df << 14);
        let lossless = webp(*b"VP8L", &[(21, &bits.to_le_bytes())]);
        assert_eq!(image_dimensions(&lossless, "image/webp"), size(640, 480));
        let extended = webp(*b"VP8X", &[(24, &[0x7f, 0x02, 0x00]), (27, &[0xdf, 0x01, 0x00])]);
        assert_eq!(image_dimensions(&extended, "image/webp"), size(640, 480));
        assert_eq!(image_dimensions(&webp(*b"ALPH", &[]), "image/webp"), None);
    }

    #[test]
    fn jpeg_skips_fill_and_tables() {
        let mut bytes = vec![0xff, 0xd8];
        // An APP0 segment of 4 bytes, a fill byte, a DHT (C4) segment, then SOF2.
        bytes.extend_from_slice(&[0xff, 0xe0, 0x00, 0x04, 0x00, 0x00]);
        bytes.extend_from_slice(&[0xff]);
        bytes.extend_from_slice(&[0xff, 0xc4, 0x00, 0x03, 0x00]);
        bytes.extend_from_slice(&[0xff, 0xc2, 0x00, 0x11, 0x08, 0x01, 0xe0, 0x02, 0x80, 0x03]);
        assert_eq!(image_dimensions(&bytes, "image/jpeg"), size(640, 480));
        assert_eq!(
            image_dimensions(&[0xff, 0xd8, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0], "image/jpeg"),
            None
        );
    }
}
