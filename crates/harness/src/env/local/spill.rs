//! The spill of one command: the complete raw output in a temporary file once it crossed the
//! [`SpillOptions`] thresholds.
//!
//! Until then the output is counted and kept as the spill's prefix; afterwards every chunk is
//! written before the next one is read, so a slow disk pauses the command's pipes instead of
//! buffering without bound.

use delta_chord::context::Context;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncWriteExt, BufWriter};

use super::fs::create_temp_file;
use crate::env::{ExecError, ExecErrorCode, SpillOptions, TempFileOptions};

/// Buffer of the spill writer.
const SPILL_BUFFER: usize = 64 * 1024;

#[derive(Debug)]
pub(super) struct Spill {
    state: State,
}

#[derive(Debug)]
enum State {
    /// No spill requested.
    Off,
    /// Below the thresholds: the output so far.
    Counting {
        thresholds: SpillOptions,
        prefix: Vec<u8>,
        newlines: u64,
    },
    Writing {
        path: String,
        writer: BufWriter<File>,
    },
    /// A write failed; the command is being killed.
    Failed,
}

impl Spill {
    pub(super) fn new(options: Option<SpillOptions>) -> Self {
        let state = match options {
            Some(thresholds) => State::Counting {
                thresholds,
                prefix: Vec::new(),
                newlines: 0,
            },
            None => State::Off,
        };
        Self { state }
    }

    /// Account one raw chunk of either stream. An error means the complete output can no longer
    /// be preserved; the caller kills the command.
    pub(super) async fn push(&mut self, chunk: &[u8], ctx: &Context) -> Result<(), ExecError> {
        if chunk.is_empty() {
            return Ok(());
        }
        if let State::Counting {
            thresholds,
            prefix,
            newlines,
        } = &mut self.state
        {
            prefix.extend_from_slice(chunk);
            *newlines += count_newlines(chunk);
            // Complete or partial lines.
            let lines = *newlines + u64::from(chunk.last() != Some(&b'\n'));
            if prefix.len() as u64 <= thresholds.after_bytes && lines <= thresholds.after_lines {
                return Ok(());
            }
            let prefix = std::mem::take(prefix);
            return self.start(&prefix, ctx).await;
        }
        if let State::Writing { writer, .. } = &mut self.state
            && let Err(error) = writer.write_all(chunk).await
        {
            self.state = State::Failed;
            return Err(failure(error));
        }
        Ok(())
    }

    /// Create the spill file and write the output seen so far.
    async fn start(&mut self, prefix: &[u8], ctx: &Context) -> Result<(), ExecError> {
        let options = TempFileOptions {
            prefix: Some("delta-output-".to_owned()),
            suffix: Some(".log".to_owned()),
        };
        // Created even when the command is being aborted: the output so far is preserved.
        let created = match create_temp_file(&options, &ctx.without_cancel()).await {
            Ok(path) => path,
            Err(error) => {
                self.state = State::Failed;
                return Err(ExecError::new(
                    ExecErrorCode::Unknown,
                    format!("Failed to preserve complete shell output: {error}"),
                ));
            }
        };
        let opened = OpenOptions::new().append(true).open(&created).await;
        let mut writer = match opened {
            Ok(file) => BufWriter::with_capacity(SPILL_BUFFER, file),
            Err(error) => {
                self.state = State::Failed;
                return Err(failure(error));
            }
        };
        if let Err(error) = writer.write_all(prefix).await {
            self.state = State::Failed;
            return Err(failure(error));
        }
        self.state = State::Writing { path: created, writer };
        Ok(())
    }

    /// Flush the spill; returns its path when there is one.
    pub(super) async fn finish(&mut self) -> Result<Option<String>, ExecError> {
        match &mut self.state {
            State::Writing { path, writer } => match writer.flush().await {
                Ok(()) => Ok(Some(path.clone())),
                Err(error) => {
                    self.state = State::Failed;
                    Err(failure(error))
                }
            },
            _ => Ok(None),
        }
    }

    /// The spill file, once there is one.
    pub(super) fn path(&self) -> Option<&str> {
        match &self.state {
            State::Writing { path, .. } => Some(path),
            _ => None,
        }
    }
}

fn failure(error: std::io::Error) -> ExecError {
    let message = format!("Failed to preserve complete shell output: {error}");
    ExecError::io(ExecErrorCode::Unknown, message, error)
}

#[allow(clippy::naive_bytecount)] // No `bytecount` dependency for chunk-sized counts.
fn count_newlines(bytes: &[u8]) -> u64 {
    bytes.iter().filter(|&&byte| byte == b'\n').count() as u64
}
