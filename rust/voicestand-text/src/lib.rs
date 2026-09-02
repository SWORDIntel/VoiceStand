//! Focused-application text output backends.

use std::path::{Path, PathBuf};
use std::process::Command;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TextSinkError {
    #[error("text output backend is unavailable: {0}")]
    Unavailable(String),
    #[error("text output failed: {0}")]
    Commit(String),
}

pub type Result<T> = std::result::Result<T, TextSinkError>;

/// Lifecycle for text destined for the application that currently owns focus.
pub trait TextSink: Send {
    fn name(&self) -> &'static str;
    fn begin(&mut self) -> Result<()>;
    fn partial(&mut self, text: &str) -> Result<()>;
    fn commit(&mut self, text: &str) -> Result<()>;
    fn cancel(&mut self) -> Result<()>;
}

/// X11/XWayland compatibility sink using xdotool's focused-window typing.
///
/// Partials are intentionally not inserted: replacing an unstable partial in
/// arbitrary applications is unsafe without a real input-method protocol.
pub struct XdotoolTextSink {
    executable: PathBuf,
    active: bool,
}

impl XdotoolTextSink {
    pub fn new() -> Result<Self> {
        Self::with_executable("xdotool")
    }

    pub fn with_executable(executable: impl AsRef<Path>) -> Result<Self> {
        if std::env::var_os("DISPLAY").is_none() {
            return Err(TextSinkError::Unavailable("DISPLAY is not set".into()));
        }
        let executable = executable.as_ref().to_path_buf();
        let status = Command::new(&executable)
            .arg("--version")
            .status()
            .map_err(|error| TextSinkError::Unavailable(error.to_string()))?;
        if !status.success() {
            return Err(TextSinkError::Unavailable(format!(
                "{} --version failed",
                executable.display()
            )));
        }
        Ok(Self {
            executable,
            active: false,
        })
    }

    fn safe_text(text: &str) -> String {
        text.chars()
            .filter_map(|character| match character {
                '\n' | '\r' | '\t' => Some(' '),
                character if character.is_control() => None,
                character => Some(character),
            })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl TextSink for XdotoolTextSink {
    fn name(&self) -> &'static str {
        "xdotool-x11"
    }

    fn begin(&mut self) -> Result<()> {
        self.active = true;
        Ok(())
    }

    fn partial(&mut self, _text: &str) -> Result<()> {
        Ok(())
    }

    fn commit(&mut self, text: &str) -> Result<()> {
        if !self.active {
            self.begin()?;
        }
        let text = Self::safe_text(text);
        if text.is_empty() {
            self.active = false;
            return Ok(());
        }
        let status = Command::new(&self.executable)
            .args(["type", "--clearmodifiers", "--delay", "1", "--"])
            .arg(text)
            .status()
            .map_err(|error| TextSinkError::Commit(error.to_string()))?;
        self.active = false;
        if status.success() {
            Ok(())
        } else {
            Err(TextSinkError::Commit(format!(
                "xdotool exited with {status}"
            )))
        }
    }

    fn cancel(&mut self) -> Result<()> {
        self.active = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::XdotoolTextSink;

    #[test]
    fn removes_command_triggering_controls() {
        assert_eq!(
            XdotoolTextSink::safe_text("hello\nrm -rf nope\r\n"),
            "hello rm -rf nope"
        );
        assert_eq!(XdotoolTextSink::safe_text("a\t b\u{7} c"), "a b c");
    }
}
