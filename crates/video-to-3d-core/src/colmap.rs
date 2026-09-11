//! Reusable sparse COLMAP interchange contracts.
//!
//! `video-to-3d-core` owns these scene-data types because they describe reconstruction inputs and
//! outputs shared across adapters. Format-specific parsing and writing stay outside the core.

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
