//! Which image a file is, from its first bytes: upstream's `detectSupportedImageMimeTypeOf`.

use delta_chord::context::Context;

use crate::env::{BinaryReader, FileResult};

/// Bytes every check reads, except the PNG chunk walk: BMP reads up to offset 29.
const HEADER_BYTES: usize = 32;
/// Bytes read at a time while walking PNG chunks.
const BLOCK_BYTES: usize = 64 * 1024;
const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/// The image format of an opened file of `size` bytes, reading its header and, for a PNG,
/// its chunk headers up to the first `acTL` or `IDAT`. `None`: not an image `read` takes,
/// so the file is read as text.
pub(in crate::tools) async fn detect(
    reader: &dyn BinaryReader,
    size: u64,
    ctx: &Context,
) -> FileResult<Option<&'static str>> {
    let header = reader.read(0, HEADER_BYTES, ctx).await?;
    if !header.starts_with(&PNG_SIGNATURE) {
        return Ok(detect_image_type(&header));
    }
    Ok((is_png(&header) && !is_animated_png_of(reader, size, ctx).await?).then_some("image/png"))
}

/// The image format `bytes` (a file's start) announce: JPEG (not JPEG-LS), PNG with an IHDR
/// chunk and no animation within `bytes`, GIF, WebP, or a structurally plausible BMP.
pub fn detect_image_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return (bytes.get(3) != Some(&0xf7)).then_some("image/jpeg");
    }
    if bytes.starts_with(&PNG_SIGNATURE) {
        return (is_png(bytes) && !is_animated_png(bytes)).then_some("image/png");
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        return Some("image/webp");
    }
    if bytes.starts_with(b"BM") && is_bmp(bytes) {
        return Some("image/bmp");
    }
    None
}

fn is_png(bytes: &[u8]) -> bool {
    bytes.len() >= 16 && u32_be(bytes, 8) == 13 && &bytes[12..16] == b"IHDR"
}

/// What a chunk header (length and type) says about animation.
enum Chunk {
    Animated,
    Still,
    /// The next chunk header's offset.
    Next(u64),
}

fn chunk(header: &[u8], offset: u64, size: u64) -> Chunk {
    match &header[4..8] {
        b"acTL" => Chunk::Animated,
        b"IDAT" => Chunk::Still,
        _ => {
            let next = offset + 8 + u64::from(u32_be(header, 0)) + 4;
            if next > size { Chunk::Still } else { Chunk::Next(next) }
        }
    }
}

fn is_animated_png(bytes: &[u8]) -> bool {
    let size = bytes.len() as u64;
    let mut offset = PNG_SIGNATURE.len() as u64;
    while offset + 8 <= size {
        let at = usize::try_from(offset).unwrap_or(usize::MAX);
        match chunk(&bytes[at..at + 8], offset, size) {
            Chunk::Animated => return true,
            Chunk::Still => return false,
            Chunk::Next(next) => offset = next,
        }
    }
    false
}

/// [`is_animated_png`] over a file read in blocks.
async fn is_animated_png_of(reader: &dyn BinaryReader, size: u64, ctx: &Context) -> FileResult<bool> {
    let mut block: Vec<u8> = Vec::new();
    let mut block_start = 0;
    let mut offset = PNG_SIGNATURE.len() as u64;
    while offset + 8 <= size {
        if offset < block_start || offset + 8 > block_start + block.len() as u64 {
            block_start = offset;
            block = reader.read(offset, BLOCK_BYTES, ctx).await?;
            if block.len() < 8 {
                return Ok(false);
            }
        }
        let at = usize::try_from(offset - block_start).unwrap_or(usize::MAX);
        match chunk(&block[at..at + 8], offset, size) {
            Chunk::Animated => return Ok(true),
            Chunk::Still => return Ok(false),
            Chunk::Next(next) => offset = next,
        }
    }
    Ok(false)
}

fn is_bmp(bytes: &[u8]) -> bool {
    if bytes.len() < 26 {
        return false;
    }
    let declared = u32_le(bytes, 2);
    let pixel_offset = u64::from(u32_le(bytes, 10));
    let dib = u32_le(bytes, 14);
    if declared != 0 && declared < 26 {
        return false;
    }
    if pixel_offset < 14 + u64::from(dib) {
        return false;
    }
    if declared != 0 && pixel_offset >= u64::from(declared) {
        return false;
    }
    let (planes, bits) = if dib == 12 {
        (u16_le(bytes, 22), u16_le(bytes, 24))
    } else if (40..=124).contains(&dib) && bytes.len() >= 30 {
        (u16_le(bytes, 26), u16_le(bytes, 28))
    } else {
        return false;
    };
    planes == 1 && matches!(bits, 1 | 4 | 8 | 16 | 24 | 32)
}

pub(super) fn u16_le(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([byte(bytes, at), byte(bytes, at + 1)])
}

pub(super) fn u16_be(bytes: &[u8], at: usize) -> u16 {
    u16::from_be_bytes([byte(bytes, at), byte(bytes, at + 1)])
}

pub(super) fn u32_le(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([
        byte(bytes, at),
        byte(bytes, at + 1),
        byte(bytes, at + 2),
        byte(bytes, at + 3),
    ])
}

pub(super) fn u32_be(bytes: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([
        byte(bytes, at),
        byte(bytes, at + 1),
        byte(bytes, at + 2),
        byte(bytes, at + 3),
    ])
}

/// The byte at `at`, 0 past the end (as upstream reads `undefined ?? 0`).
pub(super) fn byte(bytes: &[u8], at: usize) -> u8 {
    bytes.get(at).copied().unwrap_or(0)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// A PNG header for `width` × `height` with an IHDR chunk, followed by `chunks`.
    pub(in crate::tools) fn png(width: u32, height: u32, chunks: &[&[u8; 4]]) -> Vec<u8> {
        let mut bytes = PNG_SIGNATURE.to_vec();
        bytes.extend_from_slice(&13u32.to_be_bytes());
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[8, 6, 0, 0, 0]);
        bytes.extend_from_slice(&[0; 4]); // CRC
        for kind in chunks {
            bytes.extend_from_slice(&0u32.to_be_bytes());
            bytes.extend_from_slice(*kind);
            bytes.extend_from_slice(&[0; 4]);
        }
        bytes
    }

    /// A BMP header with a 40-byte DIB header.
    pub(in crate::tools) fn bmp(declared: u32, pixel_offset: u32, planes: u16, bits: u16) -> Vec<u8> {
        let mut bytes = b"BM".to_vec();
        bytes.extend_from_slice(&declared.to_le_bytes());
        bytes.extend_from_slice(&[0; 4]);
        bytes.extend_from_slice(&pixel_offset.to_le_bytes());
        bytes.extend_from_slice(&40u32.to_le_bytes());
        bytes.extend_from_slice(&4i32.to_le_bytes());
        bytes.extend_from_slice(&4i32.to_le_bytes());
        bytes.extend_from_slice(&planes.to_le_bytes());
        bytes.extend_from_slice(&bits.to_le_bytes());
        bytes
    }

    #[test]
    fn jpeg_but_not_jpeg_ls() {
        assert_eq!(detect_image_type(&[0xff, 0xd8, 0xff, 0xe0]), Some("image/jpeg"));
        assert_eq!(detect_image_type(&[0xff, 0xd8, 0xff, 0xf7]), None);
    }

    #[test]
    fn png_needs_ihdr_and_no_animation() {
        assert_eq!(detect_image_type(&png(1, 1, &[b"IDAT"])), Some("image/png"));
        assert_eq!(detect_image_type(&png(1, 1, &[b"acTL", b"IDAT"])), None);
        assert_eq!(detect_image_type(&png(1, 1, &[b"IDAT", b"acTL"])), Some("image/png"));
        let mut broken = png(1, 1, &[]);
        broken[12..16].copy_from_slice(b"XXXX");
        assert_eq!(detect_image_type(&broken), None);
    }

    #[test]
    fn gif_and_webp() {
        assert_eq!(detect_image_type(b"GIF89a\x01\x00"), Some("image/gif"));
        assert_eq!(detect_image_type(b"RIFF\0\0\0\0WEBPVP8 "), Some("image/webp"));
        assert_eq!(detect_image_type(b"RIFF\0\0\0\0WAVE"), None);
    }

    #[test]
    fn bmp_structure_is_checked() {
        assert_eq!(detect_image_type(&bmp(1000, 54, 1, 24)), Some("image/bmp"));
        assert_eq!(
            detect_image_type(&bmp(0, 54, 1, 24)),
            Some("image/bmp"),
            "an unknown size"
        );
        assert_eq!(detect_image_type(&bmp(20, 54, 1, 24)), None, "a declared size below 26");
        assert_eq!(
            detect_image_type(&bmp(1000, 40, 1, 24)),
            None,
            "pixels inside the headers"
        );
        assert_eq!(detect_image_type(&bmp(54, 54, 1, 24)), None, "pixels past the end");
        assert_eq!(detect_image_type(&bmp(1000, 54, 2, 24)), None, "two planes");
        assert_eq!(detect_image_type(&bmp(1000, 54, 1, 7)), None, "seven bits");
        assert_eq!(detect_image_type(b"BMW is a car maker and more text"), None);
    }
}
