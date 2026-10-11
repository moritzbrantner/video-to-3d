//! Cooperative I/O around the existing Serde JSON authority. Private buffers are
//! discarded on failure; no partial document is returned to artifact callers.
use serde::{de::DeserializeOwned, Serialize};
use std::io::{self, BufReader, Read, Write};

const READ_CHUNK: usize = 8 * 1024;
const WRITE_CHUNK: usize = 64 * 1024;

pub(crate) fn from_str<T: DeserializeOwned>(
    document: &str,
    poll: &mut dyn FnMut() -> Result<(), String>,
) -> Result<T, String> {
    let reader = Reader {
        remaining: document.as_bytes(),
        poll,
    };
    // Refill boundaries bound even Serde's version-header skip traversal. The
    // buffer avoids invoking the cancellation callback for each individual byte.
    serde_json::from_reader(BufReader::with_capacity(READ_CHUNK, reader))
        .map_err(|error| error.to_string())
}

struct Reader<'a, 'p> {
    remaining: &'a [u8],
    poll: &'p mut dyn FnMut() -> Result<(), String>,
}

impl Read for Reader<'_, '_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        (self.poll)().map_err(io::Error::other)?;
        let count = buffer.len().min(self.remaining.len()).min(READ_CHUNK);
        buffer[..count].copy_from_slice(&self.remaining[..count]);
        self.remaining = &self.remaining[count..];
        Ok(count)
    }
}

pub(crate) fn to_string<T: Serialize>(
    value: &T,
    pretty: bool,
    newline: bool,
    poll: &mut dyn FnMut() -> Result<(), String>,
) -> Result<String, String> {
    let mut output = StringOutput(String::new());
    to_writer(value, &mut output, pretty, poll)?;
    if newline {
        Writer {
            output: &mut output,
            poll,
        }
        .write_all(b"\n")
        .map_err(|error| error.to_string())?;
    }
    poll()?;
    Ok(output.0)
}

// Serde's JSON formatter emits UTF-8 fragments. Checking each bounded fragment
// avoids a final uncancelable UTF-8 scan of the complete owned document.
struct StringOutput(String);
impl Write for StringOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let text = std::str::from_utf8(bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        self.0.push_str(text);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(crate) fn to_writer<T: Serialize>(
    value: &T,
    output: &mut impl Write,
    pretty: bool,
    poll: &mut dyn FnMut() -> Result<(), String>,
) -> Result<(), String> {
    let writer = Writer { output, poll };
    if pretty {
        serde_json::to_writer_pretty(writer, value)
    } else {
        serde_json::to_writer(writer, value)
    }
    .map_err(|error| error.to_string())
}

struct Writer<'a, W> {
    output: &'a mut W,
    poll: &'a mut dyn FnMut() -> Result<(), String>,
}

impl<W: Write> Write for Writer<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let mut offset = 0;
        while offset < bytes.len() {
            let mut end = (offset + WRITE_CHUNK).min(bytes.len());
            // Keep the formatter's UTF-8 fragment boundaries valid when a
            // single long string write is split. At most three continuation
            // bytes belong to the next chunk; this does not parse JSON.
            while end > offset && end < bytes.len() && bytes[end] & 0xc0 == 0x80 {
                end -= 1;
            }
            if end == offset {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid JSON UTF-8 fragment",
                ));
            }
            (self.poll)().map_err(io::Error::other)?;
            self.output.write_all(&bytes[offset..end])?;
            offset = end;
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        (self.poll)().map_err(io::Error::other)?;
        self.output.flush()
    }
}
