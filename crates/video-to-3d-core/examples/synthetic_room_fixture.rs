//! Synthetic whole-scene control fixture with exact truth (issue #75).
//!
//! A closed, procedurally textured room (floor, ceiling, four walls, a door and a window)
//! holds four separate, independently placed objects. A short controlled camera path
//! (translation with a look-at, at the browser's sampling plan) renders the frames the
//! pipeline consumes, and the generator exports what an end-to-end scene check needs
//! without screenshots: intrinsics and poses, per-frame depth, per-frame instance masks,
//! every scene element's local mesh, world transform, pivot and semantic label, and the
//! expected editable inventory (architectural elements and objects).
//!
//! The scene is analytic: boxes, an upright cylinder and a sphere, intersected exactly.
//! The exported meshes tessellate the same shapes; their deviation from the analytic
//! surface is bounded and recorded per element. Everything is generated from fixed
//! parameters and a seed; nothing is committed.
//!
//! Usage: `synthetic_room_fixture <fixture-dir|-> [controlled-room]`. The fixture
//! directory receives `images/`, `truth/` (`scene.json`, `cameras.json`, depth PFMs,
//! instance PGMs, element PLYs) and `stage-report.json`, which compares the browser
//! pipeline's registered cameras against the truth. The report records evidence only;
//! it is not a quality gate. The last stdout line is a one-line summary.

mod common;

use std::{env, fs, path::Path};

use nalgebra::{Matrix3, Rotation3, Vector3};
use serde_json::{json, Value};
use video_to_3d_core::{
    reconstruct_browser, FrameInput, ReconstructionOptions, ReconstructionRequest,
};

const SCHEMA: &str = "video-to-3d/synthetic-scene-truth/v1";
const CASE: &str = "controlled-room";
/// Generator seed for every procedural texture.
const SEED: u64 = 0x0075_5ce9_e0ac_1e00;

// The notional source clip; the analysis size follows `buildVideoSamplingPlan` in
// `apps/web/src/videoSampling.ts` (see the reference fixtures).
const SOURCE_WIDTH: u32 = 1920;
const SOURCE_HEIGHT: u32 = 1080;
const DURATION_SECONDS: f64 = 16.0;
const FRAMES_PER_SECOND: f64 = 1.25;
const MAX_FRAMES: usize = 18;
const ANALYSIS_MAX_WIDTH: u32 = 360;
/// Sub-pixel rays per axis averaged into each analysis pixel.
const SUPERSAMPLE: u32 = 3;
/// The pipeline's default focal assumption, `0.86 × max(width, height)`.
const FOCAL_FACTOR: f64 = 0.86;

/// Tessellation of curved elements in the exported meshes.
const CYLINDER_SEGMENTS: usize = 64;
const SPHERE_SEGMENTS: usize = 48;
const SPHERE_RINGS: usize = 24;

/// Camera path: the center moves along a straight segment while looking at one target.
const PATH_START: [f64; 3] = [-0.9, -1.5, 0.2];
const PATH_END: [f64; 3] = [0.9, -1.45, 0.5];
const LOOK_AT: [f64; 3] = [0.2, -0.6, 2.6];

/// Direction toward the key light (world, y down); radiance =
/// albedo × (AMBIENT + (1 − AMBIENT) × max(0, n · l)).
const LIGHT: [f64; 3] = [-0.35, -1.0, -0.45];
const AMBIENT: f64 = 0.45;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Shape {
    /// Axis-aligned box in local coordinates, centered at the local origin.
    Box {
        half: [f64; 3],
    },
    /// Upright cylinder along local y, centered at the local origin.
    Cylinder {
        radius: f64,
        half_height: f64,
    },
    Sphere {
        radius: f64,
    },
}

impl Shape {
    /// Lowest point along local +y (world down for upright elements): the resting pivot.
    fn base(self) -> Vector3<f64> {
        let down = match self {
            Self::Box { half } => half[1],
            Self::Cylinder { half_height, .. } => half_height,
            Self::Sphere { radius } => radius,
        };
        Vector3::new(0.0, down, 0.0)
    }

    /// Whether a local point lies inside (or on) the shape.
    fn contains(self, point: &Vector3<f64>) -> bool {
        match self {
            Self::Box { half } => (0..3).all(|axis| point[axis].abs() <= half[axis]),
            Self::Cylinder {
                radius,
                half_height,
            } => {
                point.y.abs() <= half_height
                    && point.x * point.x + point.z * point.z <= radius * radius
            }
            Self::Sphere { radius } => point.norm_squared() <= radius * radius,
        }
    }

    /// Nearest entry hit of a local ray with `t > 0`, with the local outward normal.
    /// Exact for origins outside the shape, the only case the fixture renders (every
    /// camera center is checked against every element before rendering); an interior
    /// origin reports no hit.
    fn intersect(
        self,
        origin: &Vector3<f64>,
        direction: &Vector3<f64>,
    ) -> Option<(f64, Vector3<f64>)> {
        match self {
            Self::Box { half } => {
                let (mut near, mut far) = (f64::NEG_INFINITY, f64::INFINITY);
                let mut axis = 0;
                for i in 0..3 {
                    if direction[i].abs() < 1e-15 {
                        if origin[i].abs() > half[i] {
                            return None;
                        }
                        continue;
                    }
                    let a = (-half[i] - origin[i]) / direction[i];
                    let b = (half[i] - origin[i]) / direction[i];
                    let (low, high) = if a < b { (a, b) } else { (b, a) };
                    if low > near {
                        near = low;
                        axis = i;
                    }
                    far = far.min(high);
                }
                if near > far || near <= 1e-9 {
                    return None;
                }
                let mut normal = Vector3::zeros();
                normal[axis] = -direction[axis].signum();
                Some((near, normal))
            }
            Self::Cylinder {
                radius,
                half_height,
            } => {
                let mut best: Option<(f64, Vector3<f64>)> = None;
                let a = direction.x * direction.x + direction.z * direction.z;
                if a > 1e-15 {
                    let b = origin.x * direction.x + origin.z * direction.z;
                    let c = origin.x * origin.x + origin.z * origin.z - radius * radius;
                    let disc = b * b - a * c;
                    if disc >= 0.0 {
                        let t = (-b - disc.sqrt()) / a;
                        let y = origin.y + t * direction.y;
                        if t > 1e-9 && y.abs() <= half_height {
                            let hit = origin + direction * t;
                            best = Some((t, Vector3::new(hit.x, 0.0, hit.z) / radius));
                        }
                    }
                }
                if direction.y.abs() > 1e-15 {
                    for cap in [-half_height, half_height] {
                        let t = (cap - origin.y) / direction.y;
                        let hit = origin + direction * t;
                        if t > 1e-9
                            && hit.x * hit.x + hit.z * hit.z <= radius * radius
                            && best.is_none_or(|(current, _)| t < current)
                        {
                            best = Some((t, Vector3::new(0.0, cap.signum(), 0.0)));
                        }
                    }
                }
                best
            }
            Self::Sphere { radius } => {
                let b = origin.dot(direction);
                let c = origin.norm_squared() - radius * radius;
                let a = direction.norm_squared();
                let disc = b * b - a * c;
                if disc < 0.0 {
                    return None;
                }
                let t = (-b - disc.sqrt()) / a;
                (t > 1e-9).then(|| (t, (origin + direction * t) / radius))
            }
        }
    }

    /// Distance from a local point to the analytic surface (independent of `intersect`).
    #[cfg(test)]
    fn surface_distance(self, point: &Vector3<f64>) -> f64 {
        match self {
            Self::Box { half } => {
                let q = Vector3::new(
                    point.x.abs() - half[0],
                    point.y.abs() - half[1],
                    point.z.abs() - half[2],
                );
                let outside = Vector3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).norm();
                (outside + q.x.max(q.y).max(q.z).min(0.0)).abs()
            }
            Self::Cylinder {
                radius,
                half_height,
            } => {
                let radial = (point.x * point.x + point.z * point.z).sqrt() - radius;
                let axial = point.y.abs() - half_height;
                let outside = (radial.max(0.0).powi(2) + axial.max(0.0).powi(2)).sqrt();
                (outside + radial.max(axial).min(0.0)).abs()
            }
            Self::Sphere { radius } => (point.norm() - radius).abs(),
        }
    }

    /// Largest distance between the exported tessellation and the analytic surface.
    fn tessellation_bound(self) -> f64 {
        match self {
            Self::Box { .. } => 0.0,
            Self::Cylinder { radius, .. } => {
                radius * (1.0 - (std::f64::consts::PI / CYLINDER_SEGMENTS as f64).cos())
            }
            Self::Sphere { radius } => {
                let ring = std::f64::consts::PI / SPHERE_RINGS as f64;
                let segment = std::f64::consts::PI / SPHERE_SEGMENTS as f64;
                radius * (1.0 - (ring * 0.5).cos() * segment.cos())
            }
        }
    }

    /// Closed, outward-wound local mesh.
    fn mesh(self) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        match self {
            Self::Box { half } => {
                let vertices = (0..8)
                    .map(|corner| {
                        [0, 1, 2].map(|axis| {
                            if corner >> axis & 1 == 1 {
                                half[axis]
                            } else {
                                -half[axis]
                            }
                        })
                    })
                    .collect();
                // Corner index bits: x = 1, y = 2, z = 4.
                let faces: [[u32; 4]; 6] = [
                    [0, 4, 6, 2], // -x
                    [1, 3, 7, 5], // +x
                    [0, 1, 5, 4], // -y
                    [2, 6, 7, 3], // +y
                    [0, 2, 3, 1], // -z
                    [4, 5, 7, 6], // +z
                ];
                let triangles = faces
                    .iter()
                    .flat_map(|&[a, b, c, d]| [[a, b, c], [a, c, d]])
                    .collect();
                (vertices, triangles)
            }
            Self::Cylinder {
                radius,
                half_height,
            } => {
                let n = CYLINDER_SEGMENTS as u32;
                let mut vertices = Vec::new();
                for y in [-half_height, half_height] {
                    for segment in 0..CYLINDER_SEGMENTS {
                        let angle =
                            std::f64::consts::TAU * segment as f64 / CYLINDER_SEGMENTS as f64;
                        vertices.push([radius * angle.cos(), y, radius * angle.sin()]);
                    }
                }
                vertices.push([0.0, -half_height, 0.0]);
                vertices.push([0.0, half_height, 0.0]);
                let (top_center, bottom_center) = (2 * n, 2 * n + 1);
                let mut triangles = Vec::new();
                for i in 0..n {
                    let j = (i + 1) % n;
                    triangles.push([i, n + i, n + j]);
                    triangles.push([i, n + j, j]);
                    triangles.push([top_center, i, j]);
                    triangles.push([bottom_center, n + j, n + i]);
                }
                (vertices, triangles)
            }
            Self::Sphere { radius } => {
                let mut vertices = vec![[0.0, -radius, 0.0]];
                for ring in 1..SPHERE_RINGS {
                    let polar = std::f64::consts::PI * ring as f64 / SPHERE_RINGS as f64;
                    for segment in 0..SPHERE_SEGMENTS {
                        let azimuth =
                            std::f64::consts::TAU * segment as f64 / SPHERE_SEGMENTS as f64;
                        vertices.push([
                            radius * polar.sin() * azimuth.cos(),
                            -radius * polar.cos(),
                            radius * polar.sin() * azimuth.sin(),
                        ]);
                    }
                }
                vertices.push([0.0, radius, 0.0]);
                let s = SPHERE_SEGMENTS as u32;
                let ring_start = |ring: u32| 1 + (ring - 1) * s;
                let bottom = vertices.len() as u32 - 1;
                let mut triangles = Vec::new();
                for i in 0..s {
                    let j = (i + 1) % s;
                    triangles.push([0, ring_start(1) + i, ring_start(1) + j]);
                    let last = ring_start(SPHERE_RINGS as u32 - 1);
                    triangles.push([bottom, last + j, last + i]);
                }
                for ring in 1..SPHERE_RINGS as u32 - 1 {
                    let (upper, lower) = (ring_start(ring), ring_start(ring + 1));
                    for i in 0..s {
                        let j = (i + 1) % s;
                        triangles.push([upper + i, lower + j, upper + j]);
                        triangles.push([upper + i, lower + i, lower + j]);
                    }
                }
                (vertices, triangles)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Architecture,
    Object,
}

/// Procedural surface pattern of an element, evaluated in local surface coordinates.
#[derive(Clone, Copy, Debug)]
enum Pattern {
    Plaster,
    Planks,
    Panels,
    Glass,
    Grain,
    Checker,
}

#[derive(Clone, Copy, Debug)]
struct Element {
    id: &'static str,
    label: &'static str,
    kind: Kind,
    shape: Shape,
    /// World position of the local origin.
    center: [f64; 3],
    /// Rotation about world y (down), radians.
    yaw: f64,
    albedo: [f64; 3],
    pattern: Pattern,
}

impl Element {
    fn rotation(&self) -> Matrix3<f64> {
        *Rotation3::from_axis_angle(&Vector3::y_axis(), self.yaw).matrix()
    }

    fn center(&self) -> Vector3<f64> {
        Vector3::from(self.center)
    }

    fn local_point(&self, world: &Vector3<f64>) -> Vector3<f64> {
        self.rotation().transpose() * (world - self.center())
    }

    fn world_point(&self, local: &Vector3<f64>) -> Vector3<f64> {
        self.rotation() * local + self.center()
    }

    fn instance_id(index: usize) -> u8 {
        u8::try_from(index + 1).expect("fewer than 255 elements")
    }

    fn salt(&self) -> u64 {
        self.id.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    /// Albedo at a local surface point with local normal `normal`.
    fn albedo_at(&self, local: &Vector3<f64>, normal: &Vector3<f64>) -> [f64; 3] {
        // Surface coordinates: the two local axes orthogonal to the dominant normal axis.
        let axis = (0..3)
            .max_by(|a, b| normal[*a].abs().total_cmp(&normal[*b].abs()))
            .expect("three axes");
        let (u, v) = match axis {
            0 => (local.z, local.y),
            1 => (local.x, local.z),
            _ => (local.x, local.y),
        };
        let salt = self.salt() ^ axis as u64;
        let coarse = value_noise(u, v, 0.23, salt);
        let fine = value_noise(u, v, 0.06, salt ^ 0x5bd1);
        let factor = match self.pattern {
            Pattern::Plaster => 0.55 + 0.3 * coarse + 0.25 * fine,
            Pattern::Planks => {
                let plank = (u / 0.18).floor();
                let seam = smoothstep(0.0, 0.012, (u - plank * 0.18).min((plank + 1.0) * 0.18 - u));
                (0.5 + 0.25 * hash(plank as i64, 7, salt)
                    + 0.3 * value_noise(u * 4.0, v * 0.5, 0.2, salt))
                    * (0.55 + 0.45 * seam)
            }
            Pattern::Panels => {
                let edge = |x: f64, period: f64| {
                    let r = x.rem_euclid(period);
                    smoothstep(0.0, 0.02, r.min(period - r))
                };
                (0.6 + 0.35 * fine) * (0.6 + 0.4 * edge(u, 0.45) * edge(v, 0.5))
            }
            Pattern::Glass => 0.75 + 0.25 * coarse + 0.15 * fine,
            Pattern::Grain => 0.45 + 0.35 * value_noise(u * 6.0, v, 0.25, salt) + 0.25 * fine,
            Pattern::Checker => {
                let cell = ((u / 0.09).floor() + (v / 0.09).floor()) as i64;
                0.5 + 0.3 * (cell.rem_euclid(2) as f64) + 0.25 * fine
            }
        };
        self.albedo
            .map(|channel| (channel * factor).clamp(0.0, 1.0))
    }
}

const ROOM_HALF_WIDTH: f64 = 2.5;
const ROOM_HEIGHT: f64 = 2.6;
const ROOM_FRONT: f64 = -1.0;
const ROOM_BACK: f64 = 4.0;
const SHELL: f64 = 0.1;

/// Every scene element. World axes follow the camera convention: x right, y down,
/// z forward; the floor is the plane y = 0 and the ceiling y = -ROOM_HEIGHT.
fn scene() -> Vec<Element> {
    let depth = ROOM_BACK - ROOM_FRONT;
    let mid_z = (ROOM_BACK + ROOM_FRONT) * 0.5;
    let wall = |id, center: [f64; 3], half: [f64; 3], albedo| Element {
        id,
        label: "wall",
        kind: Kind::Architecture,
        shape: Shape::Box { half },
        center,
        yaw: 0.0,
        albedo,
        pattern: Pattern::Plaster,
    };
    let half_height = ROOM_HEIGHT * 0.5 + SHELL;
    vec![
        Element {
            id: "floor",
            label: "floor",
            kind: Kind::Architecture,
            shape: Shape::Box {
                half: [ROOM_HALF_WIDTH + SHELL, SHELL * 0.5, depth * 0.5 + SHELL],
            },
            center: [0.0, SHELL * 0.5, mid_z],
            yaw: 0.0,
            albedo: [0.72, 0.52, 0.34],
            pattern: Pattern::Planks,
        },
        Element {
            id: "ceiling",
            label: "ceiling",
            kind: Kind::Architecture,
            shape: Shape::Box {
                half: [ROOM_HALF_WIDTH + SHELL, SHELL * 0.5, depth * 0.5 + SHELL],
            },
            center: [0.0, -ROOM_HEIGHT - SHELL * 0.5, mid_z],
            yaw: 0.0,
            albedo: [0.9, 0.9, 0.88],
            pattern: Pattern::Plaster,
        },
        wall(
            "wall-back",
            [0.0, -ROOM_HEIGHT * 0.5, ROOM_BACK + SHELL * 0.5],
            [ROOM_HALF_WIDTH + SHELL, half_height, SHELL * 0.5],
            [0.78, 0.8, 0.74],
        ),
        wall(
            "wall-front",
            [0.0, -ROOM_HEIGHT * 0.5, ROOM_FRONT - SHELL * 0.5],
            [ROOM_HALF_WIDTH + SHELL, half_height, SHELL * 0.5],
            [0.74, 0.76, 0.8],
        ),
        wall(
            "wall-left",
            [-ROOM_HALF_WIDTH - SHELL * 0.5, -ROOM_HEIGHT * 0.5, mid_z],
            [SHELL * 0.5, half_height, depth * 0.5 + SHELL],
            [0.82, 0.76, 0.7],
        ),
        wall(
            "wall-right",
            [ROOM_HALF_WIDTH + SHELL * 0.5, -ROOM_HEIGHT * 0.5, mid_z],
            [SHELL * 0.5, half_height, depth * 0.5 + SHELL],
            [0.7, 0.78, 0.76],
        ),
        Element {
            id: "door",
            label: "door",
            kind: Kind::Architecture,
            shape: Shape::Box {
                half: [0.45, 1.0, 0.03],
            },
            center: [-1.45, -1.0, ROOM_BACK - 0.01],
            yaw: 0.0,
            albedo: [0.55, 0.36, 0.22],
            pattern: Pattern::Panels,
        },
        Element {
            id: "window",
            label: "window",
            kind: Kind::Architecture,
            shape: Shape::Box {
                half: [0.7, 0.5, 0.03],
            },
            center: [0.4, -1.6, ROOM_BACK - 0.01],
            yaw: 0.0,
            albedo: [0.62, 0.78, 0.92],
            pattern: Pattern::Glass,
        },
        Element {
            id: "cabinet",
            label: "cabinet",
            kind: Kind::Object,
            shape: Shape::Box {
                half: [0.5, 0.38, 0.32],
            },
            center: [-0.8, -0.38, 2.5],
            yaw: 0.0,
            albedo: [0.42, 0.55, 0.78],
            pattern: Pattern::Grain,
        },
        Element {
            id: "crate",
            label: "crate",
            kind: Kind::Object,
            shape: Shape::Box {
                half: [0.25, 0.25, 0.25],
            },
            center: [0.75, -0.25, 2.3],
            yaw: 25.0_f64.to_radians(),
            albedo: [0.85, 0.62, 0.3],
            pattern: Pattern::Checker,
        },
        Element {
            id: "stool",
            label: "stool",
            kind: Kind::Object,
            shape: Shape::Cylinder {
                radius: 0.22,
                half_height: 0.25,
            },
            center: [0.15, -0.25, 3.2],
            yaw: 0.0,
            albedo: [0.75, 0.32, 0.32],
            pattern: Pattern::Grain,
        },
        Element {
            id: "ball",
            label: "ball",
            kind: Kind::Object,
            shape: Shape::Sphere { radius: 0.3 },
            center: [-0.2, -0.3, 1.6],
            yaw: 0.0,
            albedo: [0.4, 0.75, 0.4],
            pattern: Pattern::Checker,
        },
    ]
}

fn smoothstep(edge0: f64, edge1: f64, value: f64) -> f64 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Deterministic hash of a lattice cell to `[0, 1)` (SplitMix64 finalizer).
fn hash(a: i64, b: i64, salt: u64) -> f64 {
    let mut z = SEED
        ^ (a as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (b as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
        ^ salt.wrapping_mul(0x1656_67b1_9e37_79f9);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 11) as f64 / (1_u64 << 53) as f64
}

fn value_noise(x: f64, y: f64, cell: f64, salt: u64) -> f64 {
    let gx = x / cell;
    let gy = y / cell;
    let ix = gx.floor();
    let iy = gy.floor();
    let fx = smoothstep(0.0, 1.0, gx - ix);
    let fy = smoothstep(0.0, 1.0, gy - iy);
    let (ix, iy) = (ix as i64, iy as i64);
    let top = hash(ix, iy, salt) * (1.0 - fx) + hash(ix + 1, iy, salt) * fx;
    let bottom = hash(ix, iy + 1, salt) * (1.0 - fx) + hash(ix + 1, iy + 1, salt) * fx;
    top * (1.0 - fy) + bottom * fy
}

/// The browser sampling plan, mirrored from `buildVideoSamplingPlan`.
struct SamplingPlan {
    width: u32,
    height: u32,
    times: Vec<f64>,
}

fn sampling_plan() -> SamplingPlan {
    let target = ((DURATION_SECONDS * FRAMES_PER_SECOND).ceil() as usize).max(4);
    let count = MAX_FRAMES.min(target);
    let width = ANALYSIS_MAX_WIDTH.min(SOURCE_WIDTH);
    let height = ((width as f64 / SOURCE_WIDTH as f64) * SOURCE_HEIGHT as f64)
        .round()
        .max(1.0) as u32;
    let times = (0..count)
        .map(|index| {
            let fraction = 0.05 + (0.9 * index as f64) / (count - 1) as f64;
            (DURATION_SECONDS * fraction).clamp(0.0, DURATION_SECONDS - 0.001)
        })
        .collect();
    SamplingPlan {
        width,
        height,
        times,
    }
}

/// Pinhole intrinsics; pixel centers sit at half-integer coordinates.
#[derive(Clone, Copy, Debug)]
struct Intrinsics {
    width: u32,
    height: u32,
    focal: f64,
    cx: f64,
    cy: f64,
}

impl Intrinsics {
    fn for_plan(plan: &SamplingPlan) -> Self {
        Self {
            width: plan.width,
            height: plan.height,
            focal: FOCAL_FACTOR * plan.width.max(plan.height) as f64,
            cx: plan.width as f64 * 0.5,
            cy: plan.height as f64 * 0.5,
        }
    }

    fn ray(&self, u: f64, v: f64) -> Vector3<f64> {
        Vector3::new((u - self.cx) / self.focal, (v - self.cy) / self.focal, 1.0)
    }
}

/// World-to-camera pose: `x_camera = rotation * (x_world - center)`; camera x right,
/// y down, z forward.
#[derive(Clone, Copy, Debug)]
struct Pose {
    rotation: Matrix3<f64>,
    center: Vector3<f64>,
}

impl Pose {
    fn at(time: f64) -> Self {
        let progress =
            ((time - DURATION_SECONDS * 0.05) / (DURATION_SECONDS * 0.9)).clamp(0.0, 1.0);
        let start = Vector3::from(PATH_START);
        let center = start + (Vector3::from(PATH_END) - start) * progress;
        let forward = (Vector3::from(LOOK_AT) - center).normalize();
        let right = Vector3::y().cross(&forward).normalize();
        let down = forward.cross(&right);
        Self {
            rotation: Matrix3::from_rows(&[
                right.transpose(),
                down.transpose(),
                forward.transpose(),
            ]),
            center,
        }
    }

    fn translation(&self) -> Vector3<f64> {
        -(self.rotation * self.center)
    }

    fn world_direction(&self, camera_direction: &Vector3<f64>) -> Vector3<f64> {
        self.rotation.transpose() * camera_direction
    }
}

/// Nearest element hit by a world ray: element index, ray parameter, world normal.
fn trace(
    elements: &[Element],
    origin: &Vector3<f64>,
    direction: &Vector3<f64>,
) -> Option<(usize, f64, Vector3<f64>)> {
    let mut best: Option<(usize, f64, Vector3<f64>)> = None;
    for (index, element) in elements.iter().enumerate() {
        let rotation = element.rotation();
        let local_origin = element.local_point(origin);
        let local_direction = rotation.transpose() * direction;
        if let Some((t, normal)) = element.shape.intersect(&local_origin, &local_direction) {
            if best.is_none_or(|(_, current, _)| t < current) {
                best = Some((index, t, rotation * normal));
            }
        }
    }
    best
}

fn radiance(element: &Element, world: &Vector3<f64>, world_normal: &Vector3<f64>) -> [f64; 3] {
    let local = element.local_point(world);
    let local_normal = element.rotation().transpose() * world_normal;
    let light = Vector3::from(LIGHT).normalize();
    let shade = AMBIENT + (1.0 - AMBIENT) * world_normal.dot(&light).max(0.0);
    element
        .albedo_at(&local, &local_normal)
        .map(|channel| channel * shade)
}

struct View {
    pose: Pose,
    frame: FrameInput,
    /// Camera-space z of the surface through each pixel center, row-major.
    depth: Vec<f32>,
    /// Instance id (element index + 1) through each pixel center, row-major.
    instances: Vec<u8>,
}

/// Render one view with `supersample`² area-averaged rays per pixel for the colors.
/// Depth and instance masks always come from the pixel-center ray, so they do not
/// depend on `supersample`.
fn render(elements: &[Element], intrinsics: &Intrinsics, pose: Pose, supersample: u32) -> View {
    let (width, height) = (intrinsics.width, intrinsics.height);
    let mut rgba = vec![255_u8; (width * height * 4) as usize];
    let mut depth = vec![0.0_f32; (width * height) as usize];
    let mut instances = vec![0_u8; (width * height) as usize];
    let samples = f64::from(supersample * supersample);
    let hit = |u: f64, v: f64| {
        let camera_direction = intrinsics.ray(u, v);
        let direction = pose.world_direction(&camera_direction);
        let (index, t, normal) = trace(elements, &pose.center, &direction)
            .expect("the room is closed: every ray hits an element");
        (index, t, normal, direction, camera_direction)
    };
    for py in 0..height {
        for px in 0..width {
            let mut sum = [0.0_f64; 3];
            for sy in 0..supersample {
                for sx in 0..supersample {
                    let u = f64::from(px) + (f64::from(sx) + 0.5) / f64::from(supersample);
                    let v = f64::from(py) + (f64::from(sy) + 0.5) / f64::from(supersample);
                    let (index, t, normal, direction, _) = hit(u, v);
                    let color = radiance(&elements[index], &(pose.center + direction * t), &normal);
                    for channel in 0..3 {
                        sum[channel] += color[channel];
                    }
                }
            }
            let offset = ((py * width + px) * 4) as usize;
            for channel in 0..3 {
                rgba[offset + channel] = (sum[channel] / samples * 255.0).round() as u8;
            }
            let (index, t, _, _, camera_direction) = hit(f64::from(px) + 0.5, f64::from(py) + 0.5);
            let pixel = (py * width + px) as usize;
            // The camera ray has unit z, so its z at parameter t is t.
            depth[pixel] = (camera_direction.z * t) as f32;
            instances[pixel] = Element::instance_id(index);
        }
    }
    View {
        pose,
        frame: FrameInput {
            width,
            height,
            rgba,
        },
        depth,
        instances,
    }
}

struct Fixture {
    elements: Vec<Element>,
    plan: SamplingPlan,
    intrinsics: Intrinsics,
    views: Vec<View>,
}

fn generate(supersample: u32) -> Fixture {
    let elements = scene();
    let plan = sampling_plan();
    let intrinsics = Intrinsics::for_plan(&plan);
    for &time in &plan.times {
        let center = Pose::at(time).center;
        for element in &elements {
            assert!(
                !element.shape.contains(&element.local_point(&center)),
                "camera center at {time} s lies inside {}",
                element.id
            );
        }
    }
    let views = plan
        .times
        .iter()
        .map(|&time| render(&elements, &intrinsics, Pose::at(time), supersample))
        .collect();
    Fixture {
        elements,
        plan,
        intrinsics,
        views,
    }
}

fn round6(value: f64) -> f64 {
    (value * 1e6).round() / 1e6
}

fn row_major(matrix: &Matrix3<f64>) -> Vec<f64> {
    matrix
        .transpose()
        .as_slice()
        .iter()
        .map(|v| round6(*v))
        .collect()
}

/// Pixels of each element (index) in one view.
fn pixel_counts(view: &View, elements: usize) -> Vec<usize> {
    let mut counts = vec![0; elements];
    for &id in &view.instances {
        if id > 0 {
            counts[usize::from(id) - 1] += 1;
        }
    }
    counts
}

/// Pixels an element must cover in a frame to count as visible there.
const VISIBLE_PIXELS: usize = 200;

fn world_bounds(element: &Element) -> [[f64; 3]; 2] {
    let (vertices, _) = element.shape.mesh();
    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for vertex in vertices {
        let world = element.world_point(&Vector3::from(vertex));
        for axis in 0..3 {
            low[axis] = low[axis].min(world[axis]);
            high[axis] = high[axis].max(world[axis]);
        }
    }
    [low.map(round6), high.map(round6)]
}

fn shape_document(shape: Shape) -> Value {
    match shape {
        Shape::Box { half } => json!({ "primitive": "box", "half_extents_m": half }),
        Shape::Cylinder {
            radius,
            half_height,
        } => {
            json!({ "primitive": "cylinder", "axis": "local y", "radius_m": radius, "half_height_m": half_height, "segments": CYLINDER_SEGMENTS })
        }
        Shape::Sphere { radius } => {
            json!({ "primitive": "sphere", "radius_m": radius, "segments": SPHERE_SEGMENTS, "rings": SPHERE_RINGS })
        }
    }
}

fn scene_document(fixture: &Fixture) -> Value {
    let elements = &fixture.elements;
    let counts: Vec<Vec<usize>> = fixture
        .views
        .iter()
        .map(|view| pixel_counts(view, elements.len()))
        .collect();
    let nodes: Vec<Value> = elements
        .iter()
        .enumerate()
        .map(|(index, element)| {
            let (vertices, triangles) = element.shape.mesh();
            let rotation = element.rotation();
            let center = element.center();
            let mut world_from_local = Vec::with_capacity(16);
            for row in 0..3 {
                world_from_local.extend((0..3).map(|col| round6(rotation[(row, col)])));
                world_from_local.push(round6(center[row]));
            }
            world_from_local.extend([0.0, 0.0, 0.0, 1.0]);
            let visible_frames: Vec<usize> = counts
                .iter()
                .enumerate()
                .filter(|(_, frame)| frame[index] >= VISIBLE_PIXELS)
                .map(|(frame, _)| frame)
                .collect();
            json!({
                "id": element.id,
                "instance_id": Element::instance_id(index),
                "label": element.label,
                "kind": match element.kind { Kind::Architecture => "architecture", Kind::Object => "object" },
                "independently_editable": true,
                "provenance": "synthetic ground truth",
                "shape": shape_document(element.shape),
                "world_from_local": world_from_local,
                "translation_m": element.center.map(round6),
                "yaw_degrees_about_world_y": round6(element.yaw.to_degrees()),
                "pivot_local_m": element.shape.base().iter().map(|v| round6(*v)).collect::<Vec<_>>(),
                "pivot_world_m": element.world_point(&element.shape.base()).iter().map(|v| round6(*v)).collect::<Vec<_>>(),
                "world_bounds_m": world_bounds(element),
                "mesh": format!("truth/meshes/{}.ply", element.id),
                "mesh_vertices": vertices.len(),
                "mesh_triangles": triangles.len(),
                "tessellation_bound_m": round6(element.shape.tessellation_bound()),
                "albedo": element.albedo,
                "pixels_per_frame": counts.iter().map(|frame| frame[index]).collect::<Vec<_>>(),
                "visible_frames": visible_frames,
            })
        })
        .collect();
    let objects: Vec<&str> = elements
        .iter()
        .filter(|element| element.kind == Kind::Object)
        .map(|element| element.id)
        .collect();
    let architecture: Vec<&str> = elements
        .iter()
        .filter(|element| element.kind == Kind::Architecture)
        .map(|element| element.id)
        .collect();
    json!({
        "schema": SCHEMA,
        "case": CASE,
        "intent": "deterministic synthetic control scene: a closed room with separable architectural elements and four independently placed objects, seen along a controlled translating look-at path",
        "generator": {
            "example": "crates/video-to-3d-core/examples/synthetic_room_fixture.rs",
            "seed": format!("{SEED:#018x}"),
            "room_m": { "half_width": ROOM_HALF_WIDTH, "height": ROOM_HEIGHT, "front_z": ROOM_FRONT, "back_z": ROOM_BACK, "shell_thickness": SHELL },
            "shading": format!("radiance = albedo * ({AMBIENT} + {} * max(0, n . l)), l = normalize({:?})", 1.0 - AMBIENT, LIGHT),
        },
        "coordinate_convention": {
            "world": "metres; x right, y down, z forward (the camera convention at identity); floor y = 0, ceiling y = -height",
            "local": "each element's local origin is its shape center; world = world_from_local * local (row-major 4x4)",
            "pivot": "lowest point of the shape along world down (its resting point), in local and world coordinates",
            "gltf_from_world": [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "gltf_note": "glTF/Blender import (+Y up, -Z forward) is a 180-degree rotation about x of this world",
        },
        "masks": {
            "format": "binary PGM (P5, maxval 255), one byte per pixel, rows top to bottom",
            "value": "instance_id of the element hit through each pixel center; 0 never occurs (the room is closed)",
            "visible_pixels_threshold": VISIBLE_PIXELS,
        },
        "nodes": nodes,
        "expected_inventory": {
            "objects": objects,
            "architecture": architecture,
            "min_independent_object_nodes": 3,
            "note": "ground truth of this synthetic scene: every node is a separate element with its own mesh and transform; disconnected parts of one mesh are not separate nodes",
        },
    })
}

fn cameras_document(fixture: &Fixture) -> Value {
    let intrinsics = &fixture.intrinsics;
    let frames: Vec<Value> = fixture
        .views
        .iter()
        .zip(&fixture.plan.times)
        .enumerate()
        .map(|(index, (view, time))| {
            let (near, far) = view
                .depth
                .iter()
                .fold((f32::INFINITY, 0.0_f32), |(near, far), &depth| (near.min(depth), far.max(depth)));
            json!({
                "index": index,
                "time_seconds": round6(*time),
                "image": format!("images/frame-{index:02}.ppm"),
                "depth": format!("truth/depth-{index:02}.pfm"),
                "instances": format!("truth/instances-{index:02}.pgm"),
                "rotation_world_to_camera": row_major(&view.pose.rotation),
                "translation": view.pose.translation().iter().map(|v| round6(*v)).collect::<Vec<_>>(),
                "center": view.pose.center.iter().map(|v| round6(*v)).collect::<Vec<_>>(),
                "depth_range": [near, far],
            })
        })
        .collect();
    json!({
        "schema": SCHEMA,
        "case": CASE,
        "sampling": {
            "contract": "apps/web/src/videoSampling.ts buildVideoSamplingPlan",
            "source_width": SOURCE_WIDTH,
            "source_height": SOURCE_HEIGHT,
            "duration_seconds": DURATION_SECONDS,
            "frames_per_second": FRAMES_PER_SECOND,
            "max_frames": MAX_FRAMES,
            "analysis_width": fixture.plan.width,
            "analysis_height": fixture.plan.height,
            "times": fixture.plan.times.iter().map(|t| round6(*t)).collect::<Vec<_>>(),
            "downscale": format!("area average of {SUPERSAMPLE}x{SUPERSAMPLE} sub-pixel rays per analysis pixel"),
        },
        "path": { "start": PATH_START, "end": PATH_END, "look_at": LOOK_AT, "interpolation": "linear in sample time" },
        "intrinsics": {
            "model": "PINHOLE",
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.focal,
            "fy": intrinsics.focal,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
            "pixel_centers": "half-integer (COLMAP convention)",
        },
        "pose_convention": "x_camera = R * x_world + t; camera x right, y down, z forward; rotation row-major",
        "depth": {
            "format": "PFM float32 little-endian, rows bottom to top",
            "value": "camera-space z in metres of the surface through each pixel center",
        },
        "frames": frames,
    })
}

fn write_ppm(path: &Path, width: u32, height: u32, rgb: impl Iterator<Item = [u8; 3]>) {
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    for pixel in rgb {
        bytes.extend_from_slice(&pixel);
    }
    fs::write(path, bytes).expect("write PPM");
}

fn write_pgm(path: &Path, width: u32, height: u32, values: &[u8]) {
    let mut bytes = format!("P5\n{width} {height}\n255\n").into_bytes();
    bytes.extend_from_slice(values);
    fs::write(path, bytes).expect("write PGM");
}

/// Portable float map: little-endian float32, rows stored bottom to top.
fn write_pfm(path: &Path, width: u32, height: u32, values: &[f32]) {
    let mut bytes = format!("Pf\n{width} {height}\n-1.0\n").into_bytes();
    for row in (0..height).rev() {
        for column in 0..width {
            bytes.extend_from_slice(&values[(row * width + column) as usize].to_le_bytes());
        }
    }
    fs::write(path, bytes).expect("write PFM");
}

fn write_mesh_ply(path: &Path, element: &Element) {
    let (vertices, triangles) = element.shape.mesh();
    let mut bytes = format!(
        "ply\nformat binary_little_endian 1.0\ncomment video-to-3d synthetic scene element {} (local coordinates)\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement face {}\nproperty list uchar uint vertex_indices\nend_header\n",
        element.id,
        vertices.len(),
        triangles.len()
    )
    .into_bytes();
    for vertex in &vertices {
        for value in vertex {
            bytes.extend_from_slice(&(*value as f32).to_le_bytes());
        }
    }
    for triangle in &triangles {
        bytes.push(3);
        for index in triangle {
            bytes.extend_from_slice(&index.to_le_bytes());
        }
    }
    fs::write(path, bytes).expect("write PLY");
}

fn write_fixture(root: &Path, fixture: &Fixture) {
    let images = root.join("images");
    let truth = root.join("truth");
    let meshes = truth.join("meshes");
    for directory in [&images, &truth, &meshes] {
        fs::create_dir_all(directory).expect("create fixture directory");
    }
    for (index, view) in fixture.views.iter().enumerate() {
        let (width, height) = (view.frame.width, view.frame.height);
        write_ppm(
            &images.join(format!("frame-{index:02}.ppm")),
            width,
            height,
            view.frame
                .rgba
                .as_chunks::<4>()
                .0
                .iter()
                .map(|pixel| [pixel[0], pixel[1], pixel[2]]),
        );
        write_pfm(
            &truth.join(format!("depth-{index:02}.pfm")),
            width,
            height,
            &view.depth,
        );
        write_pgm(
            &truth.join(format!("instances-{index:02}.pgm")),
            width,
            height,
            &view.instances,
        );
    }
    for element in &fixture.elements {
        write_mesh_ply(&meshes.join(format!("{}.ply", element.id)), element);
    }
    let intrinsics = &fixture.intrinsics;
    fs::write(
        truth.join("colmap-camera-params.txt"),
        format!(
            "{},{},{},{}\n",
            intrinsics.focal, intrinsics.focal, intrinsics.cx, intrinsics.cy
        ),
    )
    .expect("write COLMAP camera parameters");
    for (name, document) in [
        ("scene.json", scene_document(fixture)),
        ("cameras.json", cameras_document(fixture)),
    ] {
        fs::write(
            truth.join(name),
            serde_json::to_vec_pretty(&document).expect("serialize truth"),
        )
        .expect("write truth");
    }
}

fn browser_options() -> ReconstructionOptions {
    ReconstructionOptions {
        max_features: 320,
        min_feature_distance: 7,
        descriptor_radius: 3,
        match_radius: 42,
        max_descriptor_distance: 36.0,
        ratio_threshold: 0.82,
        focal_length_pixels: None,
        max_dense_working_set_bytes: 256 * 1024 * 1024,
    }
}

/// Camera evidence of the browser pipeline on the fixture frames, against the truth.
fn camera_stage_report(fixture: &Fixture) -> Value {
    let result = match reconstruct_browser(&ReconstructionRequest {
        frames: fixture
            .views
            .iter()
            .map(|view| view.frame.clone())
            .collect(),
        options: browser_options(),
    }) {
        Ok(result) => result,
        Err(error) => {
            return json!({ "available": false, "error": error.to_string() });
        }
    };
    let state = &result.camera_state;
    let registered: Vec<_> = state
        .calibrated_seed_cameras
        .iter()
        .chain(&state.registered_cameras)
        .collect();
    let truth_centers: Vec<Vector3<f64>> =
        fixture.views.iter().map(|view| view.pose.center).collect();
    let matched: Vec<_> = registered
        .iter()
        .map(|camera| {
            (
                Vector3::new(
                    f64::from(camera.x),
                    f64::from(camera.y),
                    f64::from(camera.z),
                ),
                truth_centers[camera.frame_index],
            )
        })
        .collect();
    let rmse = common::normalized_center_rmse(&matched, common::trajectory_span(&truth_centers));
    json!({
        "available": true,
        "frames": fixture.views.len(),
        "registered_frames": registered.iter().map(|camera| camera.frame_index).collect::<Vec<_>>(),
        "normalized_center_rmse": rmse.map(round6),
        "dense_points": result.reconstruction.dense_points.len(),
        "mesh_triangles": result.reconstruction.mesh_triangles.len(),
        "gated": false,
        "note": "evidence only: object decomposition, placement and export checks against scene.json need the scene pipeline (#76/#150)",
    })
}

fn main() {
    let output_dir = env::args().nth(1).filter(|value| value != "-");
    let case = env::args().nth(2).unwrap_or_else(|| CASE.to_owned());
    assert_eq!(case, CASE, "unknown synthetic scene fixture: {case}");
    let fixture = generate(SUPERSAMPLE);
    let cameras = camera_stage_report(&fixture);
    if let Some(output_dir) = output_dir.as_deref() {
        let root = Path::new(output_dir);
        write_fixture(root, &fixture);
        let report = json!({
            "schema": "video-to-3d/synthetic-scene-stage-report/v1",
            "case": CASE,
            "stages": { "cameras": cameras },
        });
        fs::write(
            root.join("stage-report.json"),
            serde_json::to_vec_pretty(&report).expect("serialize report"),
        )
        .expect("write stage report");
    }
    let objects = fixture
        .elements
        .iter()
        .filter(|element| element.kind == Kind::Object)
        .count();
    println!(
        "synthetic-scene case={CASE} frames={} elements={} objects={} registered={} normalized_center_rmse={}",
        fixture.views.len(),
        fixture.elements.len(),
        objects,
        cameras["registered_frames"].as_array().map_or(0, Vec::len),
        cameras["normalized_center_rmse"],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moller_trumbore(
        origin: &Vector3<f64>,
        direction: &Vector3<f64>,
        corners: [Vector3<f64>; 3],
    ) -> Option<f64> {
        let edge1 = corners[1] - corners[0];
        let edge2 = corners[2] - corners[0];
        let p = direction.cross(&edge2);
        let det = edge1.dot(&p);
        if det.abs() < 1e-14 {
            return None;
        }
        let inverse = 1.0 / det;
        let s = origin - corners[0];
        let u = s.dot(&p) * inverse;
        if !(-1e-9..=1.0 + 1e-9).contains(&u) {
            return None;
        }
        let q = s.cross(&edge1);
        let v = direction.dot(&q) * inverse;
        if v < -1e-9 || u + v > 1.0 + 1e-9 {
            return None;
        }
        let t = edge2.dot(&q) * inverse;
        (t > 1e-9).then_some(t)
    }

    /// The fixture as written to disk, read back only through its files.
    struct Exported {
        width: u32,
        height: u32,
        focal: f64,
        cx: f64,
        cy: f64,
        /// Per frame: world-to-camera rotation, center, depth and instance masks.
        frames: Vec<(Matrix3<f64>, Vector3<f64>, Vec<f32>, Vec<u8>)>,
        /// World triangles of every exported mesh with their instance id.
        triangles: Vec<(u8, [Vector3<f64>; 3])>,
        /// Tessellation bound per instance id.
        bounds: std::collections::BTreeMap<u8, f64>,
    }

    fn header_end(bytes: &[u8], lines: usize) -> (Vec<String>, usize) {
        let mut cursor = 0;
        let mut out = Vec::new();
        while out.len() < lines {
            let end = cursor
                + bytes[cursor..]
                    .iter()
                    .position(|b| *b == b'\n')
                    .expect("header line");
            out.push(String::from_utf8(bytes[cursor..end].to_vec()).expect("ASCII header"));
            cursor = end + 1;
        }
        (out, cursor)
    }

    fn read_ply(path: &Path) -> (Vec<Vector3<f64>>, Vec<[u32; 3]>) {
        let bytes = fs::read(path).expect("read PLY");
        let text_end = bytes
            .windows(11)
            .position(|w| w == b"end_header\n")
            .expect("PLY header")
            + 11;
        let header = String::from_utf8(bytes[..text_end].to_vec()).expect("ASCII header");
        // The declared layout must be exactly what the payload decoding below assumes.
        let declared: Vec<&str> = header
            .lines()
            .filter(|line| !line.starts_with("comment ") && !line.starts_with("element "))
            .collect();
        assert_eq!(
            declared,
            [
                "ply",
                "format binary_little_endian 1.0",
                "property float x",
                "property float y",
                "property float z",
                "property list uchar uint vertex_indices",
                "end_header",
            ],
            "unexpected PLY layout in {}",
            path.display()
        );
        let elements: Vec<&str> = header
            .lines()
            .filter(|line| line.starts_with("element "))
            .collect();
        assert!(
            elements.len() == 2
                && elements[0].starts_with("element vertex ")
                && elements[1].starts_with("element face "),
            "PLY must declare vertices, then faces"
        );
        let count = |name: &str| -> usize {
            header
                .lines()
                .find_map(|line| line.strip_prefix(&format!("element {name} ")))
                .expect("element count")
                .parse()
                .expect("count")
        };
        let (vertices, faces) = (count("vertex"), count("face"));
        let f32_at =
            |offset: usize| f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let u32_at =
            |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let positions = (0..vertices)
            .map(|v| {
                let base = text_end + v * 12;
                Vector3::new(
                    f64::from(f32_at(base)),
                    f64::from(f32_at(base + 4)),
                    f64::from(f32_at(base + 8)),
                )
            })
            .collect();
        let start = text_end + vertices * 12;
        let triangles = (0..faces)
            .map(|f| {
                let base = start + f * 13;
                assert_eq!(bytes[base], 3);
                [u32_at(base + 1), u32_at(base + 5), u32_at(base + 9)]
            })
            .collect();
        assert_eq!(bytes.len(), start + faces * 13, "PLY has trailing bytes");
        (positions, triangles)
    }

    fn exported() -> &'static Exported {
        static EXPORTED: std::sync::OnceLock<Exported> = std::sync::OnceLock::new();
        EXPORTED.get_or_init(|| {
            let root =
                env::temp_dir().join(format!("video-to-3d-synthetic-room-{}", std::process::id()));
            let _ = fs::remove_dir_all(&root);
            write_fixture(&root, fixture());
            let json = |name: &str| -> Value {
                serde_json::from_slice(
                    &fs::read(root.join("truth").join(name)).expect("read truth"),
                )
                .expect("JSON")
            };
            let (scene, cameras) = (json("scene.json"), json("cameras.json"));
            let number = |value: &Value| value.as_f64().expect("number");
            let intrinsics = &cameras["intrinsics"];
            let (width, height) = (
                number(&intrinsics["width"]) as u32,
                number(&intrinsics["height"]) as u32,
            );
            let pixels = (width * height) as usize;
            let frames = cameras["frames"]
                .as_array()
                .expect("frames")
                .iter()
                .map(|frame| {
                    let values = |key: &str| {
                        frame[key]
                            .as_array()
                            .expect(key)
                            .iter()
                            .map(number)
                            .collect::<Vec<_>>()
                    };
                    let rotation = Matrix3::from_row_slice(&values("rotation_world_to_camera"));
                    let center = Vector3::from_row_slice(&values("center"));
                    let translation = Vector3::from_row_slice(&values("translation"));
                    assert!(
                        (rotation * center + translation).norm() < 1e-5,
                        "serialized R, t and center disagree with x_camera = R x_world + t"
                    );
                    let depth_bytes =
                        fs::read(root.join(frame["depth"].as_str().unwrap())).expect("depth");
                    let (header, start) = header_end(&depth_bytes, 3);
                    assert_eq!(
                        header,
                        [
                            "Pf".to_owned(),
                            format!("{width} {height}"),
                            "-1.0".to_owned()
                        ]
                    );
                    assert_eq!(
                        depth_bytes.len() - start,
                        pixels * 4,
                        "PFM payload is not width x height floats"
                    );
                    let mut depth = vec![0.0_f32; pixels];
                    for (index, chunk) in depth_bytes[start..].as_chunks::<4>().0.iter().enumerate()
                    {
                        // Rows are stored bottom to top.
                        let (row, column) = (index / width as usize, index % width as usize);
                        depth[(height as usize - 1 - row) * width as usize + column] =
                            f32::from_le_bytes(*chunk);
                    }
                    let mask_bytes =
                        fs::read(root.join(frame["instances"].as_str().unwrap())).expect("mask");
                    let (header, start) = header_end(&mask_bytes, 3);
                    assert_eq!(
                        header,
                        [
                            "P5".to_owned(),
                            format!("{width} {height}"),
                            "255".to_owned()
                        ]
                    );
                    assert_eq!(
                        mask_bytes.len() - start,
                        pixels,
                        "PGM payload is not width x height bytes"
                    );
                    (rotation, center, depth, mask_bytes[start..].to_vec())
                })
                .collect();
            let mut triangles = Vec::new();
            let mut bounds = std::collections::BTreeMap::new();
            for node in scene["nodes"].as_array().expect("nodes") {
                let id = number(&node["instance_id"]) as u8;
                bounds.insert(id, number(&node["tessellation_bound_m"]));
                let m: Vec<f64> = node["world_from_local"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(number)
                    .collect();
                assert_eq!(&m[12..], &[0.0, 0.0, 0.0, 1.0]);
                let rotation = Matrix3::new(m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]);
                let translation = Vector3::new(m[3], m[7], m[11]);
                let (vertices, faces) = read_ply(&root.join(node["mesh"].as_str().unwrap()));
                let world: Vec<Vector3<f64>> = vertices
                    .iter()
                    .map(|v| rotation * v + translation)
                    .collect();
                triangles.extend(
                    faces
                        .iter()
                        .map(|f| (id, f.map(|corner| world[corner as usize]))),
                );
            }
            let _ = fs::remove_dir_all(&root);
            Exported {
                width,
                height,
                focal: number(&intrinsics["fx"]),
                cx: number(&intrinsics["cx"]),
                cy: number(&intrinsics["cy"]),
                frames,
                triangles,
                bounds,
            }
        })
    }

    fn fixture() -> &'static Fixture {
        static FIXTURE: std::sync::OnceLock<Fixture> = std::sync::OnceLock::new();
        // Masks and depth do not depend on the color supersampling; one ray per
        // pixel keeps the debug-build tests fast.
        FIXTURE.get_or_init(|| generate(1))
    }

    #[test]
    fn sampling_and_intrinsics_follow_the_browser_plan() {
        let fixture = fixture();
        assert_eq!(fixture.views.len(), MAX_FRAMES);
        assert_eq!(
            (fixture.intrinsics.width, fixture.intrinsics.height),
            (360, 203)
        );
        assert!((fixture.intrinsics.focal - 0.86 * 360.0).abs() < 1e-12);
    }

    #[test]
    fn generation_is_deterministic() {
        let elements = scene();
        let plan = sampling_plan();
        let intrinsics = Intrinsics::for_plan(&plan);
        let again = render(&elements, &intrinsics, Pose::at(plan.times[7]), 1);
        let first = &fixture().views[7];
        assert_eq!(again.frame.rgba, first.frame.rgba);
        assert_eq!(again.depth, first.depth);
        assert_eq!(again.instances, first.instances);
    }

    #[test]
    fn poses_are_rotations_and_look_at_the_target() {
        let target = Vector3::from(LOOK_AT);
        for view in &fixture().views {
            let r = view.pose.rotation;
            assert!((r * r.transpose() - Matrix3::identity()).norm() < 1e-12);
            assert!((r.determinant() - 1.0).abs() < 1e-12);
            let camera = r * (target - view.pose.center);
            assert!(camera.x.abs() < 1e-9 && camera.y.abs() < 1e-9 && camera.z > 0.0);
        }
    }

    #[test]
    fn inventory_has_at_least_three_independent_objects_visible_across_the_clip() {
        let fixture = fixture();
        let document = scene_document(fixture);
        let inventory = &document["expected_inventory"];
        let objects = inventory["objects"].as_array().expect("objects");
        assert!(objects.len() >= 3);
        assert_eq!(inventory["min_independent_object_nodes"], 3);
        let ids: std::collections::BTreeSet<&str> =
            fixture.elements.iter().map(|element| element.id).collect();
        assert_eq!(ids.len(), fixture.elements.len(), "element ids are unique");
        for node in document["nodes"].as_array().expect("nodes") {
            let visible = node["visible_frames"]
                .as_array()
                .expect("visible frames")
                .len();
            if node["kind"] == "object" {
                // Stable identity needs the object in most sampled keyframes.
                assert!(
                    visible >= MAX_FRAMES * 2 / 3,
                    "object {} is visible in only {visible} frames",
                    node["id"]
                );
            }
        }
        // Every architectural element the inventory claims as observable is seen somewhere,
        // except the wall behind the camera.
        for node in document["nodes"].as_array().expect("nodes") {
            if node["kind"] == "architecture"
                && node["id"] != "wall-front"
                && node["id"] != "ceiling"
            {
                assert!(
                    !node["visible_frames"]
                        .as_array()
                        .expect("frames")
                        .is_empty(),
                    "architectural element {} is never visible",
                    node["id"]
                );
            }
        }
    }

    #[test]
    fn objects_rest_on_the_floor_inside_the_room_without_interpenetration() {
        let elements = scene();
        let objects: Vec<&Element> = elements.iter().filter(|e| e.kind == Kind::Object).collect();
        for object in &objects {
            let pivot = object.world_point(&object.shape.base());
            assert!(
                pivot.y.abs() < 1e-12,
                "{} does not rest on the floor",
                object.id
            );
            let [low, high] = world_bounds(object);
            assert!(low[0] > -ROOM_HALF_WIDTH && high[0] < ROOM_HALF_WIDTH);
            assert!(low[2] > ROOM_FRONT && high[2] < ROOM_BACK);
            assert!(low[1] > -ROOM_HEIGHT && high[1] <= 1e-9);
        }
        for (i, a) in objects.iter().enumerate() {
            for b in &objects[i + 1..] {
                let [a_low, a_high] = world_bounds(a);
                let [b_low, b_high] = world_bounds(b);
                let separated =
                    (0..3).any(|axis| a_high[axis] < b_low[axis] || b_high[axis] < a_low[axis]);
                assert!(separated, "{} and {} overlap", a.id, b.id);
            }
        }
    }

    #[test]
    fn depth_and_masks_back_project_onto_the_labelled_element() {
        // Independent of ray intersection: the back-projected pixel-center point must lie
        // on the analytic surface (signed-distance formula) of the element its mask names.
        let fixture = fixture();
        let intrinsics = &fixture.intrinsics;
        for view in fixture.views.iter().step_by(3) {
            for py in (0..intrinsics.height).step_by(4) {
                for px in (0..intrinsics.width).step_by(4) {
                    let pixel = (py * intrinsics.width + px) as usize;
                    let id = view.instances[pixel];
                    assert!(id > 0);
                    let element = &fixture.elements[usize::from(id) - 1];
                    let camera = intrinsics.ray(f64::from(px) + 0.5, f64::from(py) + 0.5)
                        * f64::from(view.depth[pixel]);
                    let world = view.pose.rotation.transpose() * camera + view.pose.center;
                    let distance = element.shape.surface_distance(&element.local_point(&world));
                    assert!(
                        distance < 1e-4 * f64::from(view.depth[pixel]),
                        "pixel ({px}, {py}) labelled {} is {distance} m off its surface",
                        element.id
                    );
                }
            }
        }
    }

    #[test]
    fn exported_meshes_and_transforms_reproduce_depth_and_masks() {
        // Read back only the written files (cameras.json, scene.json transforms, PLY
        // meshes, PFM depth, PGM masks) and ray-cast the world-transformed meshes: the
        // nearest triangle's instance and depth must match the exported truth up to the
        // tessellation bound.
        let exported = exported();
        let mut checked = 0;
        let mut mask_disagreements = 0;
        for (rotation, center, depth, masks) in exported.frames.iter().step_by(6) {
            for py in (2..exported.height).step_by(9) {
                for px in (2..exported.width).step_by(9) {
                    let pixel = (py * exported.width + px) as usize;
                    let camera_direction = Vector3::new(
                        (f64::from(px) + 0.5 - exported.cx) / exported.focal,
                        (f64::from(py) + 0.5 - exported.cy) / exported.focal,
                        1.0,
                    );
                    let direction = rotation.transpose() * camera_direction;
                    let (id, t, corners) = exported
                        .triangles
                        .iter()
                        .filter_map(|(id, corners)| {
                            moller_trumbore(center, &direction, *corners).map(|t| (*id, t, corners))
                        })
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .expect("closed room: every ray hits an exported mesh");
                    let truth_id = masks[pixel];
                    let truth_depth = f64::from(depth[pixel]);
                    let bound = exported.bounds[&truth_id].max(exported.bounds[&id]);
                    // Depth along the camera z equals the ray parameter (unit z). A facet
                    // within `bound` of the surface along its normal can differ by
                    // bound / cos(incidence) along a grazing ray.
                    let facet = (corners[1] - corners[0])
                        .cross(&(corners[2] - corners[0]))
                        .normalize();
                    let incidence = facet.dot(&direction).abs() / direction.norm();
                    let tolerance = 2e-5 * truth_depth + 2.0 * bound / incidence.max(0.1);
                    if id != truth_id {
                        mask_disagreements += 1;
                        // Only possible where a curved silhouette's tessellation differs.
                        assert!(bound > 0.0, "flat element mismatch at ({px}, {py})");
                    } else {
                        assert!(
                            (t - truth_depth).abs() <= tolerance,
                            "mesh depth {t} vs truth {truth_depth} at ({px}, {py}) on instance {id}"
                        );
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked > 1000);
        assert!(
            mask_disagreements * 200 < checked,
            "{mask_disagreements} of {checked} disagree"
        );
    }

    #[test]
    fn camera_axes_follow_the_documented_convention() {
        // Independent of the cross products in `Pose::at`: for a level look-at in a y-down
        // world, camera x is the horizontal (fz, 0, -fx) direction and camera y points
        // toward the floor.
        for view in &fixture().views {
            let r = view.pose.rotation;
            let forward = (Vector3::from(LOOK_AT) - view.pose.center).normalize();
            let right = Vector3::new(forward.z, 0.0, -forward.x).normalize();
            assert!(
                (r.row(0).transpose() - right).norm() < 1e-12,
                "camera x is not image right"
            );
            assert!((r.row(2).transpose() - forward).norm() < 1e-12);
            assert!(r[(1, 1)] > 0.0, "camera y does not point toward the floor");
            // The floor point below the target appears below the image center.
            let below =
                r * (Vector3::from(LOOK_AT) + Vector3::new(0.0, 0.3, 0.0) - view.pose.center);
            assert!(below.y > 0.0);
        }
    }

    #[test]
    fn hits_report_the_expected_face_cap_and_radial_normals() {
        let toward = |origin: [f64; 3], direction: [f64; 3], shape: Shape| {
            shape
                .intersect(&Vector3::from(origin), &Vector3::from(direction))
                .expect("hit")
        };
        let cube = Shape::Box {
            half: [1.0, 2.0, 3.0],
        };
        for (origin, direction, normal, t) in [
            ([0.0, 0.0, -10.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], 7.0),
            ([10.0, 0.5, 0.5], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 9.0),
            ([0.2, -10.0, 0.1], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], 8.0),
            ([0.0, 0.0, -10.0], [0.05, 0.0, 1.0], [0.0, 0.0, -1.0], 7.0),
        ] {
            let (hit_t, hit_normal) = toward(origin, direction, cube);
            assert!(
                (hit_t - t).abs() < 1e-12 && (hit_normal - Vector3::from(normal)).norm() < 1e-12
            );
        }
        let cylinder = Shape::Cylinder {
            radius: 0.5,
            half_height: 1.0,
        };
        let (t, normal) = toward([0.0, -5.0, 0.1], [0.0, 1.0, 0.0], cylinder);
        assert!(
            (t - 4.0).abs() < 1e-12 && (normal - Vector3::new(0.0, -1.0, 0.0)).norm() < 1e-12,
            "top cap"
        );
        let (t, normal) = toward([0.0, 5.0, 0.1], [0.0, -1.0, 0.0], cylinder);
        assert!(
            (t - 4.0).abs() < 1e-12 && (normal - Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-12,
            "bottom cap"
        );
        let (t, normal) = toward([3.0, 0.3, 0.0], [-1.0, 0.0, 0.0], cylinder);
        assert!(
            (t - 2.5).abs() < 1e-12 && (normal - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-12,
            "side"
        );
        assert!(cylinder
            .intersect(&Vector3::new(3.0, 1.5, 0.0), &Vector3::new(-1.0, 0.0, 0.0))
            .is_none());
        let sphere = Shape::Sphere { radius: 2.0 };
        let (t, normal) = toward([0.0, 0.0, -6.0], [0.0, 0.0, 1.0], sphere);
        assert!((t - 4.0).abs() < 1e-12 && (normal - Vector3::new(0.0, 0.0, -1.0)).norm() < 1e-12);
        // Every rendered hit faces its camera.
        let fixture = fixture();
        let intrinsics = &fixture.intrinsics;
        for view in fixture.views.iter().step_by(5) {
            for py in (0..intrinsics.height).step_by(7) {
                for px in (0..intrinsics.width).step_by(7) {
                    let direction = view
                        .pose
                        .world_direction(&intrinsics.ray(f64::from(px) + 0.5, f64::from(py) + 0.5));
                    let (_, _, normal) =
                        trace(&fixture.elements, &view.pose.center, &direction).unwrap();
                    assert!(
                        normal.dot(&direction) < 0.0,
                        "back-facing normal at ({px}, {py})"
                    );
                    assert!((normal.norm() - 1.0).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn camera_centers_lie_outside_every_element() {
        // `Shape::intersect` reports entry hits only, which is exact for exterior origins.
        let fixture = fixture();
        for view in &fixture.views {
            for element in &fixture.elements {
                assert!(
                    !element
                        .shape
                        .contains(&element.local_point(&view.pose.center)),
                    "{}",
                    element.id
                );
            }
        }
    }

    #[test]
    fn meshes_are_closed_and_outward_wound() {
        for element in scene() {
            let (vertices, triangles) = element.shape.mesh();
            let mut edges = std::collections::BTreeMap::<(u32, u32), i32>::new();
            let mut volume = 0.0;
            for t in &triangles {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    *edges.entry((a.min(b), a.max(b))).or_default() += if a < b { 1 } else { -1 };
                }
                let [a, b, c] = t.map(|i| Vector3::from(vertices[i as usize]));
                volume += a.dot(&b.cross(&c)) / 6.0;
            }
            assert!(
                edges.values().all(|balance| *balance == 0),
                "{} is not closed",
                element.id
            );
            assert!(volume > 0.0, "{} is not outward wound", element.id);
        }
    }

    #[test]
    fn instance_masks_separate_every_object_from_the_environment() {
        let fixture = fixture();
        for (index, element) in fixture.elements.iter().enumerate() {
            if element.kind != Kind::Object {
                continue;
            }
            let total: usize = fixture
                .views
                .iter()
                .map(|view| pixel_counts(view, fixture.elements.len())[index])
                .sum();
            assert!(
                total > VISIBLE_PIXELS * MAX_FRAMES / 2,
                "{} covers {total} pixels",
                element.id
            );
        }
    }
}
