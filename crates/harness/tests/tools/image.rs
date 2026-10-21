//! `read` on images: pass-through within the model's limits, the reasons an image cannot be
//! sent without a processor, and what a processor's preparation adds.

use std::sync::Arc;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use delta_ai::types::UserBlock;
use delta_chord::context::Context;
use delta_harness::tool::{HarnessTool, Severity};
use delta_harness::tools::{
    BashOptions, CodingToolsOptions, Dimensions, ImageLimits, ImageProcessor, PreparedImage, ReadOptions, ReadTool,
};
use futures::future::BoxFuture;

use crate::support::fixture::MODEL_ID;
use crate::{Backend, Called, Workspace};

cases!(
    an_image_within_the_limits_is_attached,
    a_model_without_vision_gets_a_note,
    an_animated_png_is_read_as_text,
    reasons_an_image_cannot_be_sent,
    model_limits_replace_the_defaults,
    a_processor_prepares_what_does_not_fit,
);

/// A PNG header for `width` × `height`: the signature, IHDR, and an empty IDAT.
fn png(width: u32, height: u32, extra: &[&[u8; 4]]) -> Vec<u8> {
    let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
    bytes.extend_from_slice(&13u32.to_be_bytes());
    bytes.extend_from_slice(b"IHDR");
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&[8, 6, 0, 0, 0, 0, 0, 0, 0]);
    for kind in extra.iter().copied().chain([b"IDAT"]) {
        bytes.extend_from_slice(&0u32.to_be_bytes());
        bytes.extend_from_slice(kind);
        bytes.extend_from_slice(&[0; 4]);
    }
    bytes
}

/// A plausible BMP header with a 40-byte DIB header.
fn bmp() -> Vec<u8> {
    let mut bytes = b"BM".to_vec();
    bytes.extend_from_slice(&70u32.to_le_bytes());
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(&54u32.to_le_bytes());
    bytes.extend_from_slice(&40u32.to_le_bytes());
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&24u16.to_le_bytes());
    bytes.resize(70, 0);
    bytes
}

/// Makes the fake model take images, with `limits` (its `inputLimits.images.resize`).
fn vision(workspace: &Workspace, limits: &serde_json::Value) {
    workspace.fixture.model.patch_model(
        MODEL_ID,
        serde_json::json!({ "input": ["text", "image"], "inputLimits": { "images": { "resize": limits } } }),
    );
}

fn image_error(called: &Called) -> &str {
    assert!(called.message.is_error);
    assert!(called.output().is_empty());
    &called.diagnostic("unsupported_image").message
}

async fn an_image_within_the_limits_is_attached(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    vision(&workspace, &serde_json::json!({}));
    let bytes = png(4, 3, &[]);
    workspace.write("shot.png", &bytes);
    let called = workspace.call("read", serde_json::json!({ "path": "shot.png" })).await;
    assert!(!called.message.is_error, "{:?}", called.diagnostics);
    let [UserBlock::Image(image), ..] = called.message.content.as_slice() else {
        panic!("an image item: {:?}", called.message.content)
    };
    assert_eq!(image.mime_type, "image/png");
    assert_eq!(image.data, STANDARD.encode(&bytes));
    let note = called.diagnostic("image");
    assert_eq!(note.severity, Severity::Info);
    assert_eq!(note.message, "Read image file [image/png].");
    workspace.close().await;
}

async fn a_model_without_vision_gets_a_note(backend: Backend) {
    // The fake model takes text only.
    let workspace = Workspace::new(backend).await;
    workspace.write("shot.png", png(4, 3, &[]));
    let called = workspace.call("read", serde_json::json!({ "path": "shot.png" })).await;
    assert!(!called.message.is_error);
    assert_eq!(
        called.diagnostic("image").message,
        "Read image file [image/png]. The current model does not support images; it sees a placeholder instead."
    );
    workspace.close().await;
}

async fn an_animated_png_is_read_as_text(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    vision(&workspace, &serde_json::json!({}));
    workspace.write("anim.png", png(4, 3, &[b"acTL"]));
    let called = workspace.call("read", serde_json::json!({ "path": "anim.png" })).await;
    assert!(!called.message.is_error);
    assert!(matches!(called.message.content.first(), Some(UserBlock::Text(_))));
    assert!(
        called
            .diagnostics
            .iter()
            .all(|diagnostic| diagnostic.code.as_deref() != Some("image"))
    );
    workspace.close().await;
}

async fn reasons_an_image_cannot_be_sent(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    vision(&workspace, &serde_json::json!({ "maxBytes": 100 }));

    workspace.write("pic.bmp", bmp());
    let called = workspace.call("read", serde_json::json!({ "path": "pic.bmp" })).await;
    assert_eq!(
        image_error(&called),
        "pic.bmp is an image (image/bmp) that needs converting, and no image processor is configured"
    );

    let mut big = png(4, 3, &[]);
    big.resize(200, 0);
    workspace.write("big.png", &big);
    let called = workspace.call("read", serde_json::json!({ "path": "big.png" })).await;
    assert_eq!(
        image_error(&called),
        "big.png is an image (image/png) of 200B, too large to send, and no image processor is configured to shrink it"
    );

    workspace.write("wide.png", png(3000, 10, &[]));
    let called = workspace.call("read", serde_json::json!({ "path": "wide.png" })).await;
    assert_eq!(
        image_error(&called),
        "wide.png is an image (image/png) of 3000x10, larger than 2000x2000, and no image processor is configured to shrink it"
    );

    // A JPEG without a frame header.
    workspace.write(
        "odd.jpg",
        [0xff, 0xd8, 0xff, 0xe0, 0x00, 0x04, 0x00, 0x00, 0x12, 0x34, 0x56],
    );
    let called = workspace.call("read", serde_json::json!({ "path": "odd.jpg" })).await;
    assert_eq!(
        image_error(&called),
        "odd.jpg is an image (image/jpeg) whose size cannot be read"
    );
    workspace.close().await;
}

async fn model_limits_replace_the_defaults(backend: Backend) {
    let workspace = Workspace::new(backend).await;
    vision(&workspace, &serde_json::json!({ "maxWidth": 100 }));
    workspace.write("shot.png", png(200, 10, &[]));
    let called = workspace.call("read", serde_json::json!({ "path": "shot.png" })).await;
    assert_eq!(
        image_error(&called),
        "shot.png is an image (image/png) of 200x10, larger than 100x2000, and no image processor is configured to shrink it"
    );
    workspace.close().await;
}

/// A processor that answers every image with `prepared`, recording the limits it got.
#[derive(Debug)]
struct Scripted {
    prepared: Option<PreparedImage>,
    limits: std::sync::Mutex<Vec<ImageLimits>>,
}

impl ImageProcessor for Scripted {
    fn prepare<'a>(
        &'a self,
        _bytes: &'a [u8],
        _mime: &'a str,
        limits: ImageLimits,
        _ctx: &'a Context,
    ) -> BoxFuture<'a, Option<PreparedImage>> {
        self.limits.lock().unwrap().push(limits);
        let prepared = self.prepared.clone();
        Box::pin(async move { prepared })
    }
}

fn scripted(prepared: Option<PreparedImage>) -> Arc<Scripted> {
    Arc::new(Scripted {
        prepared,
        limits: std::sync::Mutex::default(),
    })
}

fn with(processor: Arc<Scripted>) -> CodingToolsOptions {
    CodingToolsOptions {
        bash: BashOptions::default(),
        read: ReadOptions {
            image_processor: Some(processor),
        },
    }
}

async fn a_processor_prepares_what_does_not_fit(backend: Backend) {
    let prepared = PreparedImage {
        data: b"converted".to_vec(),
        mime: "image/png".to_owned(),
        resized: Some((
            Dimensions {
                width: 4000,
                height: 3000,
            },
            Dimensions {
                width: 2000,
                height: 1500,
            },
        )),
        converted_from: Some("image/bmp".to_owned()),
    };
    let processor = scripted(Some(prepared));
    let workspace = Workspace::open(backend, with(processor.clone()), None).await;
    vision(&workspace, &serde_json::json!({ "maxHeight": 900 }));
    workspace.write("pic.bmp", bmp());
    let called = workspace.call("read", serde_json::json!({ "path": "pic.bmp" })).await;
    assert!(!called.message.is_error, "{:?}", called.diagnostics);
    let [UserBlock::Image(image), ..] = called.message.content.as_slice() else {
        panic!("an image item")
    };
    assert_eq!(
        (image.mime_type.as_str(), image.data.as_str()),
        ("image/png", STANDARD.encode(b"converted").as_str())
    );
    assert_eq!(
        called.diagnostic("image").message,
        "Read image file [image/png]. Converted from image/bmp to image/png. Resized from 4000x3000 to 2000x1500. Multiply coordinates by 2.00 to map them to the original."
    );
    assert_eq!(processor.limits.lock().unwrap()[0].max_height, 900);
    workspace.close().await;

    let workspace = Workspace::open(backend, with(scripted(None)), None).await;
    workspace.write("pic.bmp", bmp());
    let called = workspace.call("read", serde_json::json!({ "path": "pic.bmp" })).await;
    assert_eq!(
        image_error(&called),
        "pic.bmp is an image (image/bmp) that cannot be prepared for the model"
    );
    workspace.close().await;
}

#[test]
fn the_description_names_bmp_only_with_a_processor() {
    let plain = ReadTool::new();
    assert!(plain.definition().description.contains("(jpg, png, gif, webp)"));
    let options = ReadOptions {
        image_processor: Some(scripted(None)),
    };
    let processed = ReadTool::with_options(options);
    assert!(
        processed
            .definition()
            .description
            .contains("(jpg, png, gif, webp, bmp)")
    );
}
