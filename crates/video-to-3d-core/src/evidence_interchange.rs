//! Owned, versioned interchange of reconstruction evidence across a process boundary.
//!
//! Heavy local providers (for example a learned multi-view model in a Tauri/native
//! worker) run outside WASM and hand their evidence back through this stream. The
//! receiving side decodes it into [`OwnedReconstructionEvidence`] and only ever uses
//! it through [`OwnedReconstructionEvidence::view`], so it crosses the same
//! [`ReconstructionEvidenceView`] validation as in-process evidence before any
//! fusion. In-process providers keep borrowing their buffers; nothing here copies
//! them except [`OwnedReconstructionEvidence::from_view`], for interchange.
//!
//! Stream layout (all integers and floats little-endian):
//!
//! | bytes      | content                                                          |
//! |------------|------------------------------------------------------------------|
//! | 8          | [`EVIDENCE_INTERCHANGE_MAGIC`]                                   |
//! | 4          | format version ([`EVIDENCE_INTERCHANGE_FORMAT_VERSION`]), u32    |
//! | 4          | header length `h`, u32                                           |
//! | h          | JSON header: provider, scale, cameras, regions, buffer counts    |
//! | 19 · n     | points: x, y, z, confidence (f32); r, g, b (u8)                  |
//! | 16 · t     | triangles: a, b, c (u32); confidence (f32)                       |
//! | 8 · o      | per-observed-point attributes (f32 pair), when the header says so |
//! | 16 · s     | triangle sources: triangle, a, b, c (u32), sorted by triangle    |
//! | 44 · d     | derived vertices: point, source triangle, a, b, c (u32);         |
//! |            | barycentric weights (3 × f64), in point order                    |
//!
//! `o = n − d` observed points precede the `d` derived ones; the derived-vertex
//! count is omitted from the header (and the section is empty) without them.
//!
//! Geometry buffers are written straight from the borrowed slices and read into
//! one owned buffer each. Every declared count is checked against the caller's
//! [`InterchangeLimits`] before anything is allocated, and the stream must end
//! exactly after the last buffer.

use crate::{
    DerivedVertexSource, EvidenceCamera, EvidencePointAttributes, EvidenceScale, MeshTriangle,
    Point3, ReconstructionEvidenceView, ReconstructionProviderDescriptor, SurfaceEvidenceRegion,
    TriangleSourceCorners,
};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

pub const EVIDENCE_INTERCHANGE_FORMAT_VERSION: u32 = 1;
/// 8-byte magic at the start of every encoded stream.
pub const EVIDENCE_INTERCHANGE_MAGIC: [u8; 8] = *b"V3DEVID\0";

const POINT_BYTES: usize = 19;
const TRIANGLE_BYTES: usize = 16;
const ATTRIBUTE_BYTES: usize = 8;
const SOURCE_BYTES: usize = 16;
const DERIVED_BYTES: usize = 44;
/// Points or triangles decoded per read, so a large buffer is streamed through a
/// bounded scratch block instead of a second full-size byte copy.
const CHUNK_ITEMS: usize = 4096;

/// Upper bounds a receiver accepts from one stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterchangeLimits {
    pub max_header_bytes: usize,
    pub max_cameras: usize,
    pub max_regions: usize,
    pub max_points: usize,
    pub max_triangles: usize,
}

impl Default for InterchangeLimits {
    /// Generous bounds for local providers: 64 MiB of header, 100 000 cameras and
    /// regions, 64 Mi points and 128 Mi triangles (several GiB of geometry).
    fn default() -> Self {
        Self {
            max_header_bytes: 64 << 20,
            max_cameras: 100_000,
            max_regions: 100_000,
            max_points: 64 << 20,
            max_triangles: 128 << 20,
        }
    }
}

/// Owned evidence, e.g. received from a native worker.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnedReconstructionEvidence {
    pub provider: ReconstructionProviderDescriptor,
    pub scale: EvidenceScale,
    pub cameras: Vec<EvidenceCamera>,
    pub regions: Vec<SurfaceEvidenceRegion>,
    pub points: Vec<Point3>,
    pub triangles: Vec<MeshTriangle>,
    pub point_attributes: Option<Vec<EvidencePointAttributes>>,
    /// Sparse pre-canonicalization corners of merged triangles (empty when none).
    pub triangle_sources: Vec<TriangleSourceCorners>,
    /// Lineage of the derived points at the end of `points` (empty when none).
    pub derived_vertices: Vec<DerivedVertexSource>,
}

impl OwnedReconstructionEvidence {
    /// Copy a borrowed view (the one place buffers are copied).
    pub fn from_view(view: &ReconstructionEvidenceView<'_>) -> Self {
        Self {
            provider: view.provider.clone(),
            scale: view.scale,
            cameras: view.cameras.clone(),
            regions: view.regions.clone(),
            points: view.points.to_vec(),
            triangles: view.triangles.to_vec(),
            point_attributes: view.point_attributes.map(<[_]>::to_vec),
            triangle_sources: view.triangle_sources.map(<[_]>::to_vec).unwrap_or_default(),
            derived_vertices: view.derived_vertices.to_vec(),
        }
    }

    /// Borrowed view over the owned buffers, validated by the shared rules.
    pub fn view(&self) -> Result<ReconstructionEvidenceView<'_>, String> {
        let view = ReconstructionEvidenceView::new_with_derived_vertices(
            self.provider.clone(),
            self.scale,
            self.cameras.clone(),
            self.regions.clone(),
            &self.points,
            &self.triangles,
            &self.derived_vertices,
        )?;
        let view = match &self.point_attributes {
            Some(attributes) => view.with_point_attributes(attributes)?,
            None => view,
        };
        view.with_triangle_sources(&self.triangle_sources)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    schema_version: u32,
    provider: ReconstructionProviderDescriptor,
    scale: EvidenceScale,
    cameras: Vec<EvidenceCamera>,
    regions: Vec<SurfaceEvidenceRegion>,
    points: usize,
    triangles: usize,
    point_attributes: bool,
    triangle_sources: usize,
    #[serde(default)]
    derived_vertices: usize,
}

/// Header fields borrowed for encoding, so cameras and regions are not cloned.
#[derive(Serialize)]
struct HeaderRef<'a> {
    schema_version: u32,
    provider: &'a ReconstructionProviderDescriptor,
    scale: EvidenceScale,
    cameras: &'a [EvidenceCamera],
    regions: &'a [SurfaceEvidenceRegion],
    points: usize,
    triangles: usize,
    point_attributes: bool,
    triangle_sources: usize,
    #[serde(skip_serializing_if = "is_zero")]
    derived_vertices: usize,
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

/// Lengths of the header's collections, counted without materializing them, so
/// caller limits apply before the header is deserialized.
#[derive(Deserialize)]
struct HeaderCounts {
    #[serde(default)]
    cameras: SequenceLength,
    #[serde(default)]
    regions: SequenceLength,
}

#[derive(Default)]
struct SequenceLength(usize);

impl<'de> Deserialize<'de> for SequenceLength {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = SequenceLength;
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a sequence")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<SequenceLength, A::Error> {
                let mut length = 0;
                while sequence.next_element::<serde::de::IgnoredAny>()?.is_some() {
                    length += 1;
                }
                Ok(SequenceLength(length))
            }
        }
        deserializer.deserialize_seq(Visitor)
    }
}

fn io_error(error: std::io::Error) -> String {
    format!("evidence interchange write failed: {error}")
}

/// Stream a validated view: magic, format version, header length, JSON header, then
/// the raw little-endian buffers written straight from the borrowed slices.
pub fn encode_evidence(
    view: &ReconstructionEvidenceView<'_>,
    out: &mut impl Write,
) -> Result<(), String> {
    view.validate()?;
    let header = serde_json::to_vec(&HeaderRef {
        schema_version: view.schema_version,
        provider: &view.provider,
        scale: view.scale,
        cameras: &view.cameras,
        regions: &view.regions,
        points: view.points.len(),
        triangles: view.triangles.len(),
        point_attributes: view.point_attributes.is_some(),
        triangle_sources: view.triangle_sources.map_or(0, <[_]>::len),
        derived_vertices: view.derived_vertices.len(),
    })
    .map_err(|error| format!("evidence interchange header: {error}"))?;
    let header_length = u32::try_from(header.len())
        .map_err(|_| "evidence interchange header exceeds the u32 length limit".to_owned())?;
    out.write_all(&EVIDENCE_INTERCHANGE_MAGIC)
        .map_err(io_error)?;
    out.write_all(&EVIDENCE_INTERCHANGE_FORMAT_VERSION.to_le_bytes())
        .map_err(io_error)?;
    out.write_all(&header_length.to_le_bytes())
        .map_err(io_error)?;
    out.write_all(&header).map_err(io_error)?;

    let mut block = Vec::with_capacity(CHUNK_ITEMS * POINT_BYTES);
    for chunk in view.points.chunks(CHUNK_ITEMS) {
        block.clear();
        for point in chunk {
            for value in [point.x, point.y, point.z, point.confidence] {
                block.extend_from_slice(&value.to_le_bytes());
            }
            block.extend_from_slice(&[point.r, point.g, point.b]);
        }
        out.write_all(&block).map_err(io_error)?;
    }
    for chunk in view.triangles.chunks(CHUNK_ITEMS) {
        block.clear();
        for triangle in chunk {
            for index in [triangle.a, triangle.b, triangle.c] {
                let index = u32::try_from(index).map_err(|_| {
                    "evidence interchange triangle index exceeds the u32 range".to_owned()
                })?;
                block.extend_from_slice(&index.to_le_bytes());
            }
            block.extend_from_slice(&triangle.confidence.to_le_bytes());
        }
        out.write_all(&block).map_err(io_error)?;
    }
    if let Some(attributes) = view.point_attributes {
        for chunk in attributes.chunks(CHUNK_ITEMS) {
            block.clear();
            for attribute in chunk {
                block.extend_from_slice(&attribute.reciprocal_consistency.to_le_bytes());
                block.extend_from_slice(&attribute.depth_margin.to_le_bytes());
            }
            out.write_all(&block).map_err(io_error)?;
        }
    }
    for chunk in view
        .triangle_sources
        .unwrap_or_default()
        .chunks(CHUNK_ITEMS)
    {
        block.clear();
        for source in chunk {
            for value in [source.triangle].into_iter().chain(source.corners) {
                let value = u32::try_from(value).map_err(|_| {
                    "evidence interchange triangle source exceeds the u32 range".to_owned()
                })?;
                block.extend_from_slice(&value.to_le_bytes());
            }
        }
        out.write_all(&block).map_err(io_error)?;
    }
    for chunk in view.derived_vertices.chunks(CHUNK_ITEMS) {
        block.clear();
        for derived in chunk {
            for value in [derived.point_index, derived.source_triangle]
                .into_iter()
                .chain(derived.source_corners)
            {
                let value = u32::try_from(value).map_err(|_| {
                    "evidence interchange derived vertex exceeds the u32 range".to_owned()
                })?;
                block.extend_from_slice(&value.to_le_bytes());
            }
            for weight in derived.barycentric {
                block.extend_from_slice(&weight.to_le_bytes());
            }
        }
        out.write_all(&block).map_err(io_error)?;
    }
    out.flush().map_err(io_error)
}

fn f64_at(bytes: &[u8], offset: usize) -> f64 {
    f64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("eight bytes"))
}

/// Read exactly `buffer.len()` bytes; a short read is a truncated stream.
fn read_exact(input: &mut impl Read, buffer: &mut [u8], what: &str) -> Result<(), String> {
    input.read_exact(buffer).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            format!("evidence interchange stream is truncated in the {what}")
        } else {
            format!("evidence interchange read failed in the {what}: {error}")
        }
    })
}

fn f32_at(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes"))
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes"))
}

/// Decode `count` fixed-size records through a bounded scratch block.
fn read_records<T>(
    input: &mut impl Read,
    count: usize,
    record_bytes: usize,
    what: &str,
    decode: impl Fn(&[u8]) -> T,
) -> Result<Vec<T>, String> {
    // Grow with the bytes actually received: a header that declares many records
    // over a short stream fails as truncated before it can reserve them all.
    let mut records = Vec::with_capacity(count.min(CHUNK_ITEMS));
    let mut block = vec![0_u8; CHUNK_ITEMS.min(count) * record_bytes];
    let mut remaining = count;
    while remaining > 0 {
        let items = remaining.min(CHUNK_ITEMS);
        let bytes = &mut block[..items * record_bytes];
        read_exact(input, bytes, what)?;
        records.try_reserve(items).map_err(|_| {
            format!("evidence interchange {what} exceeds the receiver's allocation limit")
        })?;
        records.extend(bytes.chunks_exact(record_bytes).map(&decode));
        remaining -= items;
    }
    Ok(records)
}

fn bounded(name: &str, declared: usize, limit: usize) -> Result<(), String> {
    if declared > limit {
        return Err(format!(
            "evidence interchange declares {declared} {name}, above the receiver limit of {limit}"
        ));
    }
    Ok(())
}

/// Decode with bounded allocation. Truncated input, trailing bytes, a wrong magic,
/// an unknown format version, unknown header fields, or evidence failing
/// [`ReconstructionEvidenceView::validate`] are errors.
pub fn decode_evidence(
    input: &mut impl Read,
    limits: InterchangeLimits,
) -> Result<OwnedReconstructionEvidence, String> {
    let mut prefix = [0_u8; 16];
    read_exact(input, &mut prefix, "prefix")?;
    if prefix[..8] != EVIDENCE_INTERCHANGE_MAGIC {
        return Err("evidence interchange stream does not start with the expected magic".into());
    }
    let version = u32_at(&prefix, 8);
    if version != EVIDENCE_INTERCHANGE_FORMAT_VERSION {
        return Err(format!(
            "unsupported evidence interchange format version {version}; this build reads version {EVIDENCE_INTERCHANGE_FORMAT_VERSION}"
        ));
    }
    let header_length = u32_at(&prefix, 12) as usize;
    bounded("header bytes", header_length, limits.max_header_bytes)?;
    // Grow with the bytes actually received, like the geometry buffers.
    let header = read_records(input, header_length, 1, "header", |bytes| bytes[0])?;
    // Collection lengths are counted first; nested data is then bounded by the
    // header byte limit.
    let counts: HeaderCounts = serde_json::from_slice(&header)
        .map_err(|error| format!("evidence interchange header: {error}"))?;
    bounded("cameras", counts.cameras.0, limits.max_cameras)?;
    bounded("regions", counts.regions.0, limits.max_regions)?;
    let header: Header = serde_json::from_slice(&header)
        .map_err(|error| format!("evidence interchange header: {error}"))?;
    bounded("points", header.points, limits.max_points)?;
    bounded("triangles", header.triangles, limits.max_triangles)?;
    // At most one source per triangle.
    bounded(
        "triangle sources",
        header.triangle_sources,
        header.triangles,
    )?;
    // Derived points are among the declared points.
    bounded("derived vertices", header.derived_vertices, header.points)?;

    let points = read_records(input, header.points, POINT_BYTES, "point buffer", |bytes| {
        Point3 {
            x: f32_at(bytes, 0),
            y: f32_at(bytes, 4),
            z: f32_at(bytes, 8),
            confidence: f32_at(bytes, 12),
            r: bytes[16],
            g: bytes[17],
            b: bytes[18],
        }
    })?;
    let triangles = read_records(
        input,
        header.triangles,
        TRIANGLE_BYTES,
        "triangle buffer",
        |bytes| MeshTriangle {
            a: u32_at(bytes, 0) as usize,
            b: u32_at(bytes, 4) as usize,
            c: u32_at(bytes, 8) as usize,
            confidence: f32_at(bytes, 12),
        },
    )?;
    let point_attributes = if header.point_attributes {
        Some(read_records(
            input,
            header.points - header.derived_vertices,
            ATTRIBUTE_BYTES,
            "attribute buffer",
            |bytes| EvidencePointAttributes {
                reciprocal_consistency: f32_at(bytes, 0),
                depth_margin: f32_at(bytes, 4),
            },
        )?)
    } else {
        None
    };
    let triangle_sources = read_records(
        input,
        header.triangle_sources,
        SOURCE_BYTES,
        "triangle source buffer",
        |bytes| TriangleSourceCorners {
            triangle: u32_at(bytes, 0) as usize,
            corners: [1, 2, 3].map(|slot| u32_at(bytes, slot * 4) as usize),
        },
    )?;
    let derived_vertices = read_records(
        input,
        header.derived_vertices,
        DERIVED_BYTES,
        "derived vertex buffer",
        |bytes| DerivedVertexSource {
            point_index: u32_at(bytes, 0) as usize,
            source_triangle: u32_at(bytes, 4) as usize,
            source_corners: [2, 3, 4].map(|slot| u32_at(bytes, slot * 4) as usize),
            barycentric: [0, 1, 2].map(|slot| f64_at(bytes, 20 + slot * 8)),
        },
    )?;
    let mut probe = [0_u8; 1];
    loop {
        match input.read(&mut probe) {
            Ok(0) => break,
            Ok(_) => {
                return Err(
                    "evidence interchange stream has trailing bytes after its last buffer".into(),
                )
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(format!("evidence interchange read failed: {error}")),
        }
    }

    let evidence = OwnedReconstructionEvidence {
        provider: header.provider,
        scale: header.scale,
        cameras: header.cameras,
        regions: header.regions,
        points,
        triangles,
        point_attributes,
        triangle_sources,
        derived_vertices,
    };
    let view = evidence.view()?;
    if view.schema_version != header.schema_version {
        return Err(format!(
            "unsupported reconstruction evidence schema version {} in the interchange header",
            header.schema_version
        ));
    }
    Ok(evidence)
}

#[cfg(test)]
#[path = "evidence_interchange_tests.rs"]
mod tests;

#[cfg(test)]
mod implementation_tests {
    use super::*;
    use crate::{EvidenceOrigin, EvidenceRange, ReconstructionProviderClass};

    fn point(x: f32) -> Point3 {
        Point3 {
            x,
            y: 0.0,
            z: 1.0,
            confidence: 0.9,
            r: 1,
            g: 2,
            b: 3,
        }
    }

    /// Points 1 and 3 coincide; triangle 0 was re-indexed from corner 3 to corner 1.
    fn fixture() -> (Vec<Point3>, Vec<MeshTriangle>, Vec<TriangleSourceCorners>) {
        let points = vec![
            point(0.0),
            point(1.0),
            Point3 {
                y: 1.0,
                ..point(0.0)
            },
            point(1.0),
        ];
        let triangles = vec![MeshTriangle {
            a: 0,
            b: 1,
            c: 2,
            confidence: 0.8,
        }];
        let sources = vec![TriangleSourceCorners {
            triangle: 0,
            corners: [0, 3, 2],
        }];
        (points, triangles, sources)
    }

    fn view<'a>(
        points: &'a [Point3],
        triangles: &'a [MeshTriangle],
        sources: &'a [TriangleSourceCorners],
    ) -> ReconstructionEvidenceView<'a> {
        ReconstructionEvidenceView::new(
            ReconstructionProviderDescriptor::new(
                "test/completion",
                ReconstructionProviderClass::GenerativeCompletion,
                None,
            )
            .unwrap(),
            EvidenceScale::ProviderLocal,
            Vec::new(),
            vec![SurfaceEvidenceRegion {
                origin: EvidenceOrigin::GenerativeCompletion,
                reference_frame: None,
                source_frames: Vec::new(),
                points: EvidenceRange::new(0, points.len()),
            }],
            points,
            triangles,
        )
        .unwrap()
        .with_triangle_sources(sources)
        .unwrap()
    }

    #[test]
    fn triangle_sources_cross_the_interchange() {
        let (points, triangles, sources) = fixture();
        let mut bytes = Vec::new();
        encode_evidence(&view(&points, &triangles, &sources), &mut bytes).unwrap();
        let decoded = decode_evidence(&mut bytes.as_slice(), InterchangeLimits::default()).unwrap();
        assert_eq!(decoded.triangle_sources, sources);
        assert_eq!(decoded.view().unwrap().triangle_sources, Some(&sources[..]));
        // A truncated source buffer is refused.
        let cut = &bytes[..bytes.len() - 1];
        let error = decode_evidence(&mut &cut[..], InterchangeLimits::default()).unwrap_err();
        assert!(error.contains("truncated"), "{error}");
    }

    #[test]
    fn interrupted_reads_are_retried() {
        struct Interrupting<'a> {
            bytes: &'a [u8],
            interrupt: bool,
        }
        impl Read for Interrupting<'_> {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                self.interrupt = !self.interrupt;
                if self.interrupt {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                self.bytes.read(buffer)
            }
        }
        let (points, triangles, sources) = fixture();
        let mut bytes = Vec::new();
        encode_evidence(&view(&points, &triangles, &sources), &mut bytes).unwrap();
        let mut reader = Interrupting {
            bytes: &bytes,
            interrupt: false,
        };
        assert!(decode_evidence(&mut reader, InterchangeLimits::default()).is_ok());
    }
}
