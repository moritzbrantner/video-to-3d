//! Reusable sparse COLMAP interchange and text parsing.
//!
//! `video-to-3d-core` owns these scene-data contracts because they describe reconstruction data
//! shared across consumers. This module parses COLMAP's sparse text interchange only; COLMAP
//! execution remains reference/evidence tooling rather than product authority.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum ColmapError {
    Io(std::io::Error),
    Parse {
        path: PathBuf,
        line: usize,
        message: String,
    },
}

impl std::fmt::Display for ColmapError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Parse {
                path,
                line,
                message,
            } => write!(
                formatter,
                "parse error in {}:{line}: {message}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for ColmapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Parse { .. } => None,
        }
    }
}

impl From<std::io::Error> for ColmapError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub type ColmapResult<T> = std::result::Result<T, ColmapError>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColmapDataset {
    pub cameras: Vec<ColmapCamera>,
    pub images: Vec<ColmapImage>,
    pub points: Vec<ColmapPoint3d>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColmapCamera {
    pub id: u32,
    pub raw_model: String,
    pub width: u32,
    pub height: u32,
    pub params: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColmapImage {
    pub id: u32,
    pub qw: f32,
    pub qx: f32,
    pub qy: f32,
    pub qz: f32,
    pub tx: f32,
    pub ty: f32,
    pub tz: f32,
    pub camera_id: u32,
    pub name: String,
    pub points2d: Vec<ColmapPoint2d>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColmapPoint2d {
    pub xy: Vec2,
    pub point3d_id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColmapPoint3d {
    pub id: u64,
    pub xyz: Vec3,
    pub color: [u8; 3],
    pub error: f32,
    pub track: Vec<ColmapTrackElement>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColmapTrackElement {
    pub image_id: u32,
    pub point2d_index: usize,
}

/// Reads `cameras.txt`, `images.txt`, and `points3D.txt` from a COLMAP sparse directory.
pub fn read_colmap_text_dir(path: impl AsRef<Path>) -> ColmapResult<ColmapDataset> {
    let path = path.as_ref();
    Ok(ColmapDataset {
        cameras: read_cameras(path.join("cameras.txt"))?,
        images: read_images(path.join("images.txt"))?,
        points: read_points(path.join("points3D.txt"))?,
    })
}

fn read_cameras(path: PathBuf) -> ColmapResult<Vec<ColmapCamera>> {
    data_lines(&path)?
        .into_iter()
        .map(|(line_number, line)| {
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() < 5 {
                return parse_error(&path, line_number, "camera line requires at least 5 fields");
            }
            Ok(ColmapCamera {
                id: parse(&path, line_number, parts[0], "camera id")?,
                raw_model: parts[1].to_string(),
                width: parse(&path, line_number, parts[2], "camera width")?,
                height: parse(&path, line_number, parts[3], "camera height")?,
                params: parts[4..]
                    .iter()
                    .map(|value| parse(&path, line_number, value, "camera parameter"))
                    .collect::<ColmapResult<Vec<_>>>()?,
            })
        })
        .collect()
}

fn read_images(path: PathBuf) -> ColmapResult<Vec<ColmapImage>> {
    let file = File::open(&path)?;
    let mut lines = BufReader::new(file).lines().enumerate();
    let mut images = Vec::new();

    while let Some((metadata_index, line)) = lines.next() {
        let metadata = line?;
        let metadata = metadata.trim();
        if metadata.is_empty() || metadata.starts_with('#') {
            continue;
        }

        let metadata_line = metadata_index + 1;
        let parts = metadata.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 10 {
            return parse_error(
                &path,
                metadata_line,
                "image metadata line requires at least 10 fields",
            );
        }

        let mut points_line = None;
        for (points_index, candidate) in lines.by_ref() {
            let candidate = candidate?;
            if candidate.trim_start().starts_with('#') {
                continue;
            }
            points_line = Some((points_index + 1, candidate));
            break;
        }
        let (points_line_number, points) =
            points_line.unwrap_or((metadata_line + 1, String::new()));

        images.push(ColmapImage {
            id: parse(&path, metadata_line, parts[0], "image id")?,
            qw: parse(&path, metadata_line, parts[1], "qw")?,
            qx: parse(&path, metadata_line, parts[2], "qx")?,
            qy: parse(&path, metadata_line, parts[3], "qy")?,
            qz: parse(&path, metadata_line, parts[4], "qz")?,
            tx: parse(&path, metadata_line, parts[5], "tx")?,
            ty: parse(&path, metadata_line, parts[6], "ty")?,
            tz: parse(&path, metadata_line, parts[7], "tz")?,
            camera_id: parse(&path, metadata_line, parts[8], "camera id")?,
            name: parts[9..].join(" "),
            points2d: parse_points2d(&path, points_line_number, &points)?,
        });
    }

    Ok(images)
}

fn parse_points2d(
    path: &Path,
    line_number: usize,
    line: &str,
) -> ColmapResult<Vec<ColmapPoint2d>> {
    let parts = line.split_whitespace().collect::<Vec<_>>();
    if parts.is_empty() {
        return Ok(Vec::new());
    }
    if parts.len() % 3 != 0 {
        return parse_error(
            path,
            line_number,
            "point2D line must contain x y point3D-id triples",
        );
    }

    parts
        .as_chunks::<3>()
        .0
        .iter()
        .map(|chunk| {
            let point3d_id = if chunk[2] == "-1" {
                None
            } else {
                Some(parse(path, line_number, chunk[2], "point3D id")?)
            };
            Ok(ColmapPoint2d {
                xy: Vec2 {
                    x: parse(path, line_number, chunk[0], "point2D x")?,
                    y: parse(path, line_number, chunk[1], "point2D y")?,
                },
                point3d_id,
            })
        })
        .collect()
}

fn read_points(path: PathBuf) -> ColmapResult<Vec<ColmapPoint3d>> {
    data_lines(&path)?
        .into_iter()
        .map(|(line_number, line)| {
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() < 8 {
                return parse_error(
                    &path,
                    line_number,
                    "point3D line requires at least 8 fields",
                );
            }
            if (parts.len() - 8) % 2 != 0 {
                return parse_error(
                    &path,
                    line_number,
                    "point3D track must contain image-id/point2D-index pairs",
                );
            }

            let track = parts[8..]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| {
                    Ok(ColmapTrackElement {
                        image_id: parse(&path, line_number, chunk[0], "track image id")?,
                        point2d_index: parse(&path, line_number, chunk[1], "track point2D index")?,
                    })
                })
                .collect::<ColmapResult<Vec<_>>>()?;

            Ok(ColmapPoint3d {
                id: parse(&path, line_number, parts[0], "point3D id")?,
                xyz: Vec3 {
                    x: parse(&path, line_number, parts[1], "point3D x")?,
                    y: parse(&path, line_number, parts[2], "point3D y")?,
                    z: parse(&path, line_number, parts[3], "point3D z")?,
                },
                color: [
                    parse(&path, line_number, parts[4], "red")?,
                    parse(&path, line_number, parts[5], "green")?,
                    parse(&path, line_number, parts[6], "blue")?,
                ],
                error: parse(&path, line_number, parts[7], "reprojection error")?,
                track,
            })
        })
        .collect()
}

fn data_lines(path: &Path) -> ColmapResult<Vec<(usize, String)>> {
    let file = File::open(path)?;
    BufReader::new(file)
        .lines()
        .enumerate()
        .filter_map(|(index, line)| match line {
            Ok(line) if line.trim().is_empty() || line.trim_start().starts_with('#') => None,
            Ok(line) => Some(Ok((index + 1, line))),
            Err(error) => Some(Err(ColmapError::Io(error))),
        })
        .collect()
}

fn parse<T>(path: &Path, line: usize, value: &str, field: &str) -> ColmapResult<T>
where
    T: std::str::FromStr,
{
    value.parse().map_err(|_| ColmapError::Parse {
        path: path.to_path_buf(),
        line,
        message: format!("invalid {field}: {value}"),
    })
}

fn parse_error<T>(path: &Path, line: usize, message: impl Into<String>) -> ColmapResult<T> {
    Err(ColmapError::Parse {
        path: path.to_path_buf(),
        line,
        message: message.into(),
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::*;

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("video-to-3d-colmap-{}-{name}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("create test directory");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn write_empty_sparse_files(directory: &TestDir, images: &str) {
        fs::write(
            directory.path().join("cameras.txt"),
            "1 PINHOLE 640 480 500 501 320 240\n",
        )
        .expect("camera fixture");
        fs::write(directory.path().join("images.txt"), images).expect("image fixture");
        fs::write(directory.path().join("points3D.txt"), "").expect("point fixture");
    }

    #[test]
    fn reads_sparse_colmap_text_directory() {
        let directory = TestDir::new("reads-sparse");
        fs::write(
            directory.path().join("cameras.txt"),
            "# cameras\n1 PINHOLE 640 480 500 501 320 240\n",
        )
        .expect("camera fixture");
        fs::write(
            directory.path().join("images.txt"),
            "# images\n1 1 0 0 0 0 0 0 1 frame_000001.png\n10 20 7 30 40 -1\n2 1 0 0 0 1 0 0 1 frame with spaces.png\n\n",
        )
        .expect("image fixture");
        fs::write(
            directory.path().join("points3D.txt"),
            "# points\n7 1 2 3 255 128 0 0.25 1 0 2 0\n",
        )
        .expect("point fixture");

        let dataset = read_colmap_text_dir(directory.path()).expect("COLMAP fixture parses");

        assert_eq!(dataset.cameras.len(), 1);
        assert_eq!(dataset.cameras[0].raw_model, "PINHOLE");
        assert_eq!(dataset.images.len(), 2);
        assert_eq!(dataset.images[0].points2d.len(), 2);
        assert_eq!(dataset.images[0].points2d[1].point3d_id, None);
        assert_eq!(dataset.images[1].name, "frame with spaces.png");
        assert!(dataset.images[1].points2d.is_empty());
        assert_eq!(dataset.points.len(), 1);
        assert_eq!(
            dataset.points[0].xyz,
            Vec3 {
                x: 1.0,
                y: 2.0,
                z: 3.0
            }
        );
        assert_eq!(dataset.points[0].track.len(), 2);
    }

    #[test]
    fn preserves_full_u64_point_ids() {
        let directory = TestDir::new("u64-point-id");
        write_empty_sparse_files(
            &directory,
            "1 1 0 0 0 0 0 0 1 frame.png\n10 20 18446744073709551615\n",
        );

        let dataset = read_colmap_text_dir(directory.path()).expect("u64 point id parses");
        assert_eq!(dataset.images[0].points2d[0].point3d_id, Some(u64::MAX));
    }

    #[test]
    fn rejects_negative_point_ids_other_than_the_colmap_sentinel() {
        let directory = TestDir::new("negative-point-id");
        write_empty_sparse_files(&directory, "1 1 0 0 0 0 0 0 1 frame.png\n10 20 -2\n");

        let error = read_colmap_text_dir(directory.path()).expect_err("-2 is not a sentinel");
        assert!(error.to_string().contains("point3D id"));
    }

    #[test]
    fn rejects_malformed_point_track() {
        let directory = TestDir::new("malformed-track");
        fs::write(
            directory.path().join("cameras.txt"),
            "1 PINHOLE 1 1 1 1 0 0\n",
        )
        .expect("camera fixture");
        fs::write(directory.path().join("images.txt"), "").expect("image fixture");
        fs::write(
            directory.path().join("points3D.txt"),
            "7 1 2 3 255 128 0 0.25 1\n",
        )
        .expect("point fixture");

        let error = read_colmap_text_dir(directory.path()).expect_err("track must be rejected");
        assert!(error.to_string().contains("track"));
    }
}
