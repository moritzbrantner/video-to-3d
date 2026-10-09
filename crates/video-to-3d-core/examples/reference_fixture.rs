//! Reference fixtures with exact per-stage truth (issue #128).
//!
//! Each fixture renders a textured façade with mild depth relief at the browser's sampling
//! plan (18 frames, 360 px analysis width, the browser's reconstruction options) and exports
//! the truth every pipeline stage can be checked against: sampled frames, intrinsics and
//! poses, per-frame depth, the true surface mesh with its connectivity, and the albedo.
//! Everything is generated from a fixed seed and parameters; nothing is committed.
//!
//! Usage: `reference_fixture <fixture-dir|-> <case>`, where `case` is one of
//! `slow-lateral-pan`, `pure-rotation` or `overlapping-references`. The fixture directory
//! receives `images/` (the frames COLMAP and video-to-3d both consume), `truth/` and
//! `rust-evidence.json` and `stage-report.json` (per-stage checks against the truth, see
//! `reference_stages`). The last stdout line is the `golden-rust` metrics line.

mod common;
mod reference_stages;

use std::{env, fs, path::Path};

use nalgebra::{Matrix3, Rotation3, Vector3};
use serde_json::{json, Value};
use video_to_3d_core::{
    classic_reference_patch_regions, reconstruct_browser, BrowserReconstructionResult, FrameInput,
    ReconstructionOptions, ReconstructionRequest,
};

/// Generator seed for every procedural texture and relief detail.
const SEED: u64 = 0x0128_7e71_f00d_5eed;

// The notional source clip. The analysis size follows `buildVideoSamplingPlan` in
// `apps/web/src/videoSampling.ts`: width min(360, source width), height rounded to aspect.
const SOURCE_WIDTH: u32 = 1920;
const SOURCE_HEIGHT: u32 = 1080;
const DURATION_SECONDS: f64 = 16.0;
const FRAMES_PER_SECOND: f64 = 1.25;
const MAX_FRAMES: usize = 18;
const ANALYSIS_MAX_WIDTH: u32 = 360;
/// Sub-pixel rays per axis: each analysis pixel is the area average of a 3×3 grid, the
/// box-filter downscale of a 1080 × 609 source.
const SUPERSAMPLE: u32 = 3;
/// Equal to the pipeline's default focal assumption (`0.86 × max(width, height)`), so the
/// fixtures isolate parallax and topology from intrinsics error.
const FOCAL_FACTOR: f64 = 0.86;

/// Trevi-like façade distance and per-interval lateral step of the slow pan.
const TREVI_DISTANCE: f64 = 6.0;
const PAN_STEP: f64 = 0.15;

const RAY_MARCH_STEPS: usize = 32;
const RAY_BISECTION_STEPS: usize = 24;
const MESH_SPACING: f64 = 0.05;
const ALBEDO_PIXELS_PER_METER: f64 = 100.0;
const OVERLAP_GRID: (u32, u32) = (36, 20);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Case {
    SlowLateralPan,
    PureRotation,
    OverlappingReferences,
}

impl Case {
    const ALL: [Self; 3] = [
        Self::SlowLateralPan,
        Self::PureRotation,
        Self::OverlappingReferences,
    ];

    fn parse(value: &str) -> Self {
        Self::ALL
            .into_iter()
            .find(|case| case.name() == value)
            .unwrap_or_else(|| panic!("unknown reference fixture: {value}"))
    }

    fn name(self) -> &'static str {
        match self {
            Self::SlowLateralPan => "slow-lateral-pan",
            Self::PureRotation => "pure-rotation",
            Self::OverlappingReferences => "overlapping-references",
        }
    }

    fn intent(self) -> &'static str {
        match self {
            Self::SlowLateralPan => {
                "Small lateral steps along a textured façade with mild depth relief; relief parallax per sampling interval sits near the baseline gate. Cameras should register."
            }
            Self::PureRotation => {
                "The same façade and viewing direction with the camera rotating in place. Negative control: the translation-baseline gate must reject it."
            }
            Self::OverlappingReferences => {
                "A slow pan where neighbouring reference patches see the same surface. The truth is one connected surface."
            }
        }
    }

    fn spec(self) -> FixtureSpec {
        let trevi = Facade {
            distance: TREVI_DISTANCE,
            relief: 0.45,
            relief_edge: 0.08,
            extent: [-6.0, 6.0, -3.0, 3.0],
        };
        match self {
            Self::SlowLateralPan => FixtureSpec {
                facade: trevi,
                motion: Motion::Lateral { step: PAN_STEP },
            },
            Self::PureRotation => FixtureSpec {
                facade: trevi,
                // The yaw step that shifts the façade plane by the pan's image motion.
                motion: Motion::Yaw {
                    step: (PAN_STEP / TREVI_DISTANCE).atan(),
                },
            },
            Self::OverlappingReferences => FixtureSpec {
                facade: Facade {
                    distance: 4.0,
                    relief: 1.0,
                    // Gentle slopes keep the true surface free of near-vertical steps, so
                    // one connected mesh is reachable without bridging discontinuities.
                    relief_edge: 0.4,
                    extent: [-5.0, 5.0, -2.5, 2.5],
                },
                motion: Motion::Lateral { step: 0.2 },
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Motion {
    /// Translation along +x at constant speed; `step` is metres per sampling interval.
    Lateral { step: f64 },
    /// Rotation about the camera's y axis at a fixed center; `step` is radians per interval.
    Yaw { step: f64 },
}

#[derive(Clone, Copy, Debug)]
struct FixtureSpec {
    facade: Facade,
    motion: Motion,
}

/// The browser sampling plan, mirrored from `buildVideoSamplingPlan`.
#[derive(Clone, Debug, PartialEq)]
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

/// Pinhole intrinsics; pixel centers sit at half-integer coordinates (COLMAP convention).
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

    fn project(&self, camera_point: &Vector3<f64>) -> Option<(f64, f64)> {
        (camera_point.z > 1e-9).then(|| {
            (
                self.focal * camera_point.x / camera_point.z + self.cx,
                self.focal * camera_point.y / camera_point.z + self.cy,
            )
        })
    }
}

/// World-to-camera pose: `x_camera = rotation * (x_world - center)`. Camera axes are
/// x right, y down, z forward.
#[derive(Clone, Copy, Debug)]
struct Pose {
    rotation: Matrix3<f64>,
    center: Vector3<f64>,
}

impl Pose {
    fn at(spec: &FixtureSpec, time: f64) -> Self {
        let plan_interval = 0.9 * DURATION_SECONDS / (MAX_FRAMES - 1) as f64;
        let progress = (time - DURATION_SECONDS * 0.5) / plan_interval;
        match spec.motion {
            Motion::Lateral { step } => Self {
                rotation: Matrix3::identity(),
                center: Vector3::new(progress * step, 0.0, 0.0),
            },
            Motion::Yaw { step } => Self {
                // Camera-to-world is a yaw about +y; world-to-camera is its transpose.
                rotation: Rotation3::from_axis_angle(&Vector3::y_axis(), progress * step)
                    .matrix()
                    .transpose(),
                center: Vector3::zeros(),
            },
        }
    }

    fn translation(&self) -> Vector3<f64> {
        -(self.rotation * self.center)
    }

    fn world_to_camera(&self, world: &Vector3<f64>) -> Vector3<f64> {
        self.rotation * (world - self.center)
    }

    fn world_direction(&self, camera_direction: &Vector3<f64>) -> Vector3<f64> {
        self.rotation.transpose() * camera_direction
    }
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

/// Smooth value noise on a lattice of `cell` metres.
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

/// A fronto-parallel façade at `distance` along +z whose relief protrudes toward the
/// camera: the surface is `z = distance - height(x, y)` with `0 ≤ height ≤ relief`.
#[derive(Clone, Copy, Debug)]
struct Facade {
    distance: f64,
    relief: f64,
    /// Width of the smooth ramp at every relief edge.
    relief_edge: f64,
    /// `[x_min, x_max, y_min, y_max]` of the exported mesh and albedo, in metres.
    extent: [f64; 4],
}

const PILASTER_PERIOD: f64 = 1.7;
const PILASTER_HALF_WIDTH: f64 = 0.22;
const BLOCK_WIDTH: f64 = 0.62;
const BLOCK_HEIGHT: f64 = 0.26;
const MORTAR: f64 = 0.018;
const SPOT_CELL: f64 = 0.18;

impl Facade {
    /// Trevi-like relief: a recessed wall with protruding pilasters, a cornice band, a
    /// plinth, and recessed niches between the pilasters. Smooth edges keep it a
    /// single-valued heightfield, so the true surface is one connected sheet.
    fn height(&self, x: f64, y: f64) -> f64 {
        let r = self.relief;
        let local = (x + 0.5 * PILASTER_PERIOD).rem_euclid(PILASTER_PERIOD) - 0.5 * PILASTER_PERIOD;
        let pilaster = 1.0
            - smoothstep(
                PILASTER_HALF_WIDTH,
                PILASTER_HALF_WIDTH + self.relief_edge,
                local.abs(),
            );
        let band = |low: f64, high: f64| {
            smoothstep(low - self.relief_edge, low, y)
                * (1.0 - smoothstep(high, high + self.relief_edge, y))
        };
        let cornice = band(-1.55, -1.25);
        let plinth = band(1.15, 4.0);
        let niche_x = (local.abs() - 0.5 * PILASTER_PERIOD) / 0.36;
        let niche_y = (y + 0.1) / 0.6;
        let niche = 1.0 - smoothstep(0.8, 1.0, niche_x.hypot(niche_y));

        let wall = 0.22 * r * (1.0 - niche);
        wall.max(0.6 * r * pilaster)
            .max(r * cornice)
            .max(0.45 * r * plinth)
            .clamp(0.0, r)
    }

    fn surface_z(&self, x: f64, y: f64) -> f64 {
        self.distance - self.height(x, y)
    }

    /// Outward (camera-facing) unit normal of the heightfield.
    fn normal(&self, x: f64, y: f64) -> Vector3<f64> {
        let eps = 1e-3;
        let hx = (self.height(x + eps, y) - self.height(x - eps, y)) / (2.0 * eps);
        let hy = (self.height(x, y + eps) - self.height(x, y - eps)) / (2.0 * eps);
        -Vector3::new(hx, hy, 1.0).normalize()
    }

    /// Travertine-like stone blocks in running bond, with per-block tone, grain and
    /// weathering spots, so every image region carries distinctive corners.
    fn albedo(&self, x: f64, y: f64) -> [f64; 3] {
        let row = (y / BLOCK_HEIGHT).floor();
        let offset = if (row as i64).rem_euclid(2) == 1 {
            0.5 * BLOCK_WIDTH
        } else {
            0.0
        };
        let column = ((x + offset) / BLOCK_WIDTH).floor();
        let in_x = (x + offset) - column * BLOCK_WIDTH;
        let in_y = y - row * BLOCK_HEIGHT;
        if in_x < MORTAR || in_y < MORTAR {
            return [0.40, 0.38, 0.34];
        }
        let (column, row) = (column as i64, row as i64);
        let tone = 0.70 + 0.42 * hash(column, row, 1);
        let tint = [
            1.0,
            0.94 + 0.08 * hash(column, row, 2),
            0.84 + 0.12 * hash(column, row, 3),
        ];
        let grain = 0.78 + 0.44 * value_noise(x, y, 0.06, 4);
        let mottling = 0.85 + 0.3 * value_noise(x, y, 0.35, 5);

        let mut spot = 1.0;
        let cell_x = (x / SPOT_CELL).floor() as i64;
        let cell_y = (y / SPOT_CELL).floor() as i64;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (sx, sy) = (cell_x + dx, cell_y + dy);
                if hash(sx, sy, 6) >= 0.5 {
                    continue;
                }
                let center_x = (sx as f64 + hash(sx, sy, 7)) * SPOT_CELL;
                let center_y = (sy as f64 + hash(sx, sy, 8)) * SPOT_CELL;
                let radius = 0.025 + 0.035 * hash(sx, sy, 9);
                if (x - center_x).hypot(y - center_y) < radius {
                    spot = if hash(sx, sy, 10) < 0.6 { 0.42 } else { 1.3 };
                }
            }
        }

        let base = [0.86, 0.79, 0.66];
        std::array::from_fn(|channel| {
            (base[channel] * tint[channel] * tone * grain * mottling * spot).clamp(0.0, 1.0)
        })
    }

    /// Lambertian shading under one fixed directional light plus ambient.
    fn radiance(&self, x: f64, y: f64) -> [f64; 3] {
        let light = Vector3::new(-0.4, -0.6, -1.0).normalize();
        let shade = 0.3 + 0.7 * self.normal(x, y).dot(&light).max(0.0);
        self.albedo(x, y).map(|value| (value * shade).min(1.0))
    }

    /// Ray parameter of the first surface hit, by marching between the relief bounds and
    /// bisecting the first sign change.
    fn intersect(&self, origin: &Vector3<f64>, direction: &Vector3<f64>) -> Option<f64> {
        if direction.z <= 1e-9 {
            return None;
        }
        let signed = |t: f64| {
            let point = origin + direction * t;
            point.z - self.surface_z(point.x, point.y)
        };
        let near = (self.distance - self.relief - 1e-6 - origin.z) / direction.z;
        let far = (self.distance + 1e-6 - origin.z) / direction.z;
        let mut previous = near;
        for step in 1..=RAY_MARCH_STEPS {
            let t = near + (far - near) * step as f64 / RAY_MARCH_STEPS as f64;
            if signed(t) >= 0.0 {
                let (mut low, mut high) = (previous, t);
                for _ in 0..RAY_BISECTION_STEPS {
                    let mid = 0.5 * (low + high);
                    if signed(mid) >= 0.0 {
                        high = mid;
                    } else {
                        low = mid;
                    }
                }
                return Some(0.5 * (low + high));
            }
            previous = t;
        }
        None
    }

    fn contains(&self, x: f64, y: f64) -> bool {
        let [x0, x1, y0, y1] = self.extent;
        (x0..=x1).contains(&x) && (y0..=y1).contains(&y)
    }
}

/// One rendered view: the browser-sampled frame and its exact depth.
struct View {
    pose: Pose,
    frame: FrameInput,
    /// Camera-space z of the surface hit through each pixel center, row-major.
    depth: Vec<f32>,
}

fn surface_hit(
    facade: &Facade,
    intrinsics: &Intrinsics,
    pose: &Pose,
    u: f64,
    v: f64,
) -> Vector3<f64> {
    let direction = pose.world_direction(&intrinsics.ray(u, v));
    let t = facade
        .intersect(&pose.center, &direction)
        .expect("every fixture ray hits the façade");
    let hit = pose.center + direction * t;
    assert!(
        facade.contains(hit.x, hit.y),
        "fixture view leaves the exported façade extent at ({:.3}, {:.3})",
        hit.x,
        hit.y
    );
    hit
}

fn render(facade: &Facade, intrinsics: &Intrinsics, pose: Pose) -> View {
    let (width, height) = (intrinsics.width, intrinsics.height);
    let mut rgba = vec![255_u8; (width * height * 4) as usize];
    let mut depth = vec![0.0_f32; (width * height) as usize];
    let samples = (SUPERSAMPLE * SUPERSAMPLE) as f64;
    for py in 0..height {
        for px in 0..width {
            let mut sum = [0.0_f64; 3];
            for sy in 0..SUPERSAMPLE {
                for sx in 0..SUPERSAMPLE {
                    let u = px as f64 + (sx as f64 + 0.5) / SUPERSAMPLE as f64;
                    let v = py as f64 + (sy as f64 + 0.5) / SUPERSAMPLE as f64;
                    let hit = surface_hit(facade, intrinsics, &pose, u, v);
                    let radiance = facade.radiance(hit.x, hit.y);
                    for channel in 0..3 {
                        sum[channel] += radiance[channel];
                    }
                }
            }
            let offset = ((py * width + px) * 4) as usize;
            for channel in 0..3 {
                rgba[offset + channel] = (sum[channel] / samples * 255.0).round() as u8;
            }
            let center = surface_hit(facade, intrinsics, &pose, px as f64 + 0.5, py as f64 + 0.5);
            depth[(py * width + px) as usize] = pose.world_to_camera(&center).z as f32;
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
    }
}

/// The true surface sampled on a regular grid: vertex positions, texture coordinates
/// and two triangles per cell.
struct TrueMesh {
    columns: usize,
    rows: usize,
    vertices: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    triangles: Vec<[u32; 3]>,
}

fn true_mesh(facade: &Facade) -> TrueMesh {
    let [x0, x1, y0, y1] = facade.extent;
    let columns = ((x1 - x0) / MESH_SPACING).round() as usize + 1;
    let rows = ((y1 - y0) / MESH_SPACING).round() as usize + 1;
    let mut vertices = Vec::with_capacity(columns * rows);
    let mut uvs = Vec::with_capacity(columns * rows);
    for row in 0..rows {
        for column in 0..columns {
            let x = x0 + (x1 - x0) * column as f64 / (columns - 1) as f64;
            let y = y0 + (y1 - y0) * row as f64 / (rows - 1) as f64;
            vertices.push([x as f32, y as f32, facade.surface_z(x, y) as f32]);
            uvs.push([((x - x0) / (x1 - x0)) as f32, ((y - y0) / (y1 - y0)) as f32]);
        }
    }
    let mut triangles = Vec::with_capacity((columns - 1) * (rows - 1) * 2);
    for row in 0..rows - 1 {
        for column in 0..columns - 1 {
            let a = (row * columns + column) as u32;
            let b = a + 1;
            let c = a + columns as u32;
            let d = c + 1;
            triangles.push([a, c, b]);
            triangles.push([b, c, d]);
        }
    }
    TrueMesh {
        columns,
        rows,
        vertices,
        uvs,
        triangles,
    }
}

/// Union-find root of every vertex after joining the vertices of each triangle.
fn component_labels(
    vertex_count: usize,
    triangles: impl Iterator<Item = [usize; 3]>,
) -> Vec<usize> {
    fn root(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }
    let mut parent: Vec<usize> = (0..vertex_count).collect();
    for [a, b, c] in triangles {
        for (left, right) in [(a, b), (b, c)] {
            let (left, right) = (root(&mut parent, left), root(&mut parent, right));
            if left != right {
                parent[left] = right;
            }
        }
    }
    (0..vertex_count)
        .map(|vertex| root(&mut parent, vertex))
        .collect()
}

/// Connected components among the vertices that triangles reference.
fn connected_components(
    vertex_count: usize,
    triangles: impl Iterator<Item = [usize; 3]> + Clone,
) -> usize {
    let mut used = vec![false; vertex_count];
    for vertex in triangles.clone().flatten() {
        used[vertex] = true;
    }
    let labels = component_labels(vertex_count, triangles);
    (0..vertex_count)
        .filter(|&vertex| used[vertex] && labels[vertex] == vertex)
        .count()
}

/// Fraction of a pixel grid of view `from` whose surface point is visible in view `to`.
fn view_overlap(facade: &Facade, intrinsics: &Intrinsics, from: &Pose, to: &Pose) -> f64 {
    let (columns, rows) = OVERLAP_GRID;
    let mut visible = 0;
    for row in 0..rows {
        for column in 0..columns {
            let u = (column as f64 + 0.5) * intrinsics.width as f64 / columns as f64;
            let v = (row as f64 + 0.5) * intrinsics.height as f64 / rows as f64;
            let world = surface_hit(facade, intrinsics, from, u, v);
            let camera = to.world_to_camera(&world);
            let Some((pu, pv)) = intrinsics.project(&camera) else {
                continue;
            };
            if !(0.0..intrinsics.width as f64).contains(&pu)
                || !(0.0..intrinsics.height as f64).contains(&pv)
            {
                continue;
            }
            let direction = to.world_direction(&intrinsics.ray(pu, pv));
            let unoccluded = facade
                .intersect(&to.center, &direction)
                .is_some_and(|t| (to.center + direction * t - world).norm() < 1e-3 * camera.z);
            visible += usize::from(unoccluded);
        }
    }
    visible as f64 / (columns * rows) as f64
}

/// Largest translation parallax between the nearest and farthest surface seen by `from`,
/// for the baseline to `to`: `focal × baseline × (1 / z_near − 1 / z_far)` in pixels.
fn relief_parallax_pixels(intrinsics: &Intrinsics, from: &View, to: &Pose) -> f64 {
    let (near, far) = from
        .depth
        .iter()
        .fold((f64::INFINITY, 0.0_f64), |(near, far), &depth| {
            (near.min(depth as f64), far.max(depth as f64))
        });
    let baseline = (to.center - from.pose.center).norm();
    intrinsics.focal * baseline * (1.0 / near - 1.0 / far)
}

fn rotation_degrees(left: &Pose, right: &Pose) -> f64 {
    let relative = right.rotation * left.rotation.transpose();
    let cosine = ((relative.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
    cosine.acos().to_degrees()
}

/// The browser's reconstruction options (`reconstructFrames` in
/// `apps/web/src/reconstruction.ts`); the focal length is left to the pipeline.
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

fn write_ppm(path: &Path, width: u32, height: u32, rgb: impl Iterator<Item = [u8; 3]>) {
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    for pixel in rgb {
        bytes.extend_from_slice(&pixel);
    }
    fs::write(path, bytes).expect("write PPM");
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

fn write_mesh_ply(path: &Path, mesh: &TrueMesh) {
    let mut bytes = format!(
        "ply\nformat binary_little_endian 1.0\ncomment video-to-3d reference fixture true surface\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty float s\nproperty float t\nelement face {}\nproperty list uchar uint vertex_indices\nend_header\n",
        mesh.vertices.len(),
        mesh.triangles.len()
    )
    .into_bytes();
    for (position, uv) in mesh.vertices.iter().zip(&mesh.uvs) {
        for value in position.iter().chain(uv) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    for triangle in &mesh.triangles {
        bytes.push(3);
        for index in triangle {
            bytes.extend_from_slice(&index.to_le_bytes());
        }
    }
    fs::write(path, bytes).expect("write PLY");
}

fn albedo_size(facade: &Facade) -> (u32, u32) {
    let [x0, x1, y0, y1] = facade.extent;
    (
        ((x1 - x0) * ALBEDO_PIXELS_PER_METER).round() as u32,
        ((y1 - y0) * ALBEDO_PIXELS_PER_METER).round() as u32,
    )
}

fn albedo_texels(facade: &Facade) -> impl Iterator<Item = [u8; 3]> + '_ {
    let [x0, _, y0, _] = facade.extent;
    let (width, height) = albedo_size(facade);
    (0..height).flat_map(move |row| {
        (0..width).map(move |column| {
            let x = x0 + (column as f64 + 0.5) / ALBEDO_PIXELS_PER_METER;
            let y = y0 + (row as f64 + 0.5) / ALBEDO_PIXELS_PER_METER;
            facade
                .albedo(x, y)
                .map(|value| (value * 255.0).round() as u8)
        })
    })
}

struct Fixture {
    case: Case,
    spec: FixtureSpec,
    plan: SamplingPlan,
    intrinsics: Intrinsics,
    views: Vec<View>,
}

fn generate(case: Case) -> Fixture {
    let spec = case.spec();
    let plan = sampling_plan();
    let intrinsics = Intrinsics::for_plan(&plan);
    let views = plan
        .times
        .iter()
        .map(|&time| render(&spec.facade, &intrinsics, Pose::at(&spec, time)))
        .collect();
    Fixture {
        case,
        spec,
        plan,
        intrinsics,
        views,
    }
}

fn round6(value: f64) -> f64 {
    (value * 1e6).round() / 1e6
}

fn truth_document(fixture: &Fixture, mesh: &TrueMesh) -> Value {
    let Fixture {
        case,
        spec,
        plan,
        intrinsics,
        views,
    } = fixture;
    let facade = &spec.facade;
    let (albedo_width, albedo_height) = albedo_size(facade);
    let mesh_components = connected_components(
        mesh.vertices.len(),
        mesh.triangles.iter().map(|t| t.map(|index| index as usize)),
    );
    let frames: Vec<Value> = views
        .iter()
        .zip(&plan.times)
        .enumerate()
        .map(|(index, (view, time))| {
            let (near, far) = view
                .depth
                .iter()
                .fold((f32::INFINITY, 0.0_f32), |(near, far), &depth| {
                    (near.min(depth), far.max(depth))
                });
            json!({
                "index": index,
                "time_seconds": round6(*time),
                "image": format!("images/frame-{index:02}.ppm"),
                "depth": format!("truth/depth-{index:02}.pfm"),
                "rotation_world_to_camera": view.pose.rotation.transpose().as_slice().iter().map(|v| round6(*v)).collect::<Vec<_>>(),
                "translation": view.pose.translation().iter().map(|v| round6(*v)).collect::<Vec<_>>(),
                "center": view.pose.center.iter().map(|v| round6(*v)).collect::<Vec<_>>(),
                "depth_range": [near, far],
            })
        })
        .collect();
    let adjacent: Vec<Value> = views
        .windows(2)
        .enumerate()
        .map(|(index, pair)| {
            json!({
                "from_frame": index,
                "to_frame": index + 1,
                "baseline_m": round6((pair[1].pose.center - pair[0].pose.center).norm()),
                "rotation_degrees": round6(rotation_degrees(&pair[0].pose, &pair[1].pose)),
                "relief_parallax_pixels": round6(relief_parallax_pixels(intrinsics, &pair[0], &pair[1].pose)),
                "overlap": round6(view_overlap(facade, intrinsics, &pair[0].pose, &pair[1].pose)),
            })
        })
        .collect();
    let overlap: Vec<Vec<f64>> = views
        .iter()
        .map(|from| {
            views
                .iter()
                .map(|to| round6(view_overlap(facade, intrinsics, &from.pose, &to.pose)))
                .collect()
        })
        .collect();
    let clip_parallax = relief_parallax_pixels(intrinsics, &views[0], &views[views.len() - 1].pose);
    let motion = match spec.motion {
        Motion::Lateral { step } => {
            json!({ "kind": "lateral", "step_m_per_sample": step, "axis": "+x", "rotation": "identity" })
        }
        Motion::Yaw { step } => {
            json!({ "kind": "yaw", "step_radians_per_sample": step, "axis": "camera +y", "center": [0.0, 0.0, 0.0] })
        }
    };

    json!({
        "schema": "video-to-3d/reference-fixture-truth/v1",
        "case": case.name(),
        "intent": case.intent(),
        "generator": {
            "example": "crates/video-to-3d-core/examples/reference_fixture.rs",
            "seed": format!("{SEED:#018x}"),
            "motion": motion,
            "facade": {
                "kind": "heightfield",
                "surface": "z = distance - height(x, y), 0 <= height <= relief",
                "distance_m": facade.distance,
                "relief_m": facade.relief,
                "extent_m": facade.extent,
                "pilaster_period_m": PILASTER_PERIOD,
                "pilaster_half_width_m": PILASTER_HALF_WIDTH,
                "relief_edge_m": facade.relief_edge,
                "block_size_m": [BLOCK_WIDTH, BLOCK_HEIGHT],
                "mortar_m": MORTAR,
            },
            "ray_march_steps": RAY_MARCH_STEPS,
            "ray_bisection_steps": RAY_BISECTION_STEPS,
        },
        "sampling": {
            "contract": "apps/web/src/videoSampling.ts buildVideoSamplingPlan",
            "source_width": SOURCE_WIDTH,
            "source_height": SOURCE_HEIGHT,
            "duration_seconds": DURATION_SECONDS,
            "frames_per_second": FRAMES_PER_SECOND,
            "max_frames": MAX_FRAMES,
            "sample_count": plan.times.len(),
            "analysis_width": plan.width,
            "analysis_height": plan.height,
            "times": plan.times.iter().map(|t| round6(*t)).collect::<Vec<_>>(),
            "downscale": format!("area average of {SUPERSAMPLE}x{SUPERSAMPLE} sub-pixel rays per analysis pixel"),
            "reconstruction_options": "apps/web/src/reconstruction.ts reconstructFrames",
        },
        "intrinsics": {
            "model": "PINHOLE",
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.focal,
            "fy": intrinsics.focal,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
            "pixel_centers": "half-integer (COLMAP convention)",
            "focal_note": "equals the pipeline default 0.86 x max(width, height)",
        },
        "pose_convention": "x_camera = R * x_world + t; camera x right, y down, z forward; rotation row-major",
        "frames": frames,
        "depth": {
            "format": "PFM float32 little-endian, rows bottom to top",
            "value": "camera-space z in metres of the surface through each pixel center",
        },
        "surface": {
            "mesh": "truth/surface.ply",
            "format": "binary little-endian PLY; vertex x y z s t; triangle faces",
            "grid_spacing_m": MESH_SPACING,
            "grid": [mesh.columns, mesh.rows],
            "vertices": mesh.vertices.len(),
            "triangles": mesh.triangles.len(),
            "connected_components": mesh_components,
            "holes": 0,
        },
        "texture": {
            "albedo": "truth/albedo.ppm",
            "width": albedo_width,
            "height": albedo_height,
            "pixels_per_meter": ALBEDO_PIXELS_PER_METER,
            "uv": "s = (x - x_min) / (x_max - x_min), t = (y - y_min) / (y_max - y_min); texel row 0 at y_min",
            "shading": "frame radiance = albedo * (0.3 + 0.7 * max(0, n . l)), l = normalize(-0.4, -0.6, -1)",
        },
        "adjacent_pairs": adjacent,
        "clip_relief_parallax_pixels": round6(clip_parallax),
        "view_overlap": {
            "grid": [OVERLAP_GRID.0, OVERLAP_GRID.1],
            "definition": "fraction of row view's pixel grid whose surface point is unoccluded inside column view",
            "matrix": overlap,
        },
    })
}

fn write_fixture(root: &Path, fixture: &Fixture, mesh: &TrueMesh) {
    let images = root.join("images");
    let truth = root.join("truth");
    fs::create_dir_all(&images).expect("create images directory");
    fs::create_dir_all(&truth).expect("create truth directory");
    let intrinsics = &fixture.intrinsics;
    for (index, view) in fixture.views.iter().enumerate() {
        write_ppm(
            &images.join(format!("frame-{index:02}.ppm")),
            view.frame.width,
            view.frame.height,
            view.frame
                .rgba
                .as_chunks::<4>()
                .0
                .iter()
                .map(|pixel| [pixel[0], pixel[1], pixel[2]]),
        );
        write_pfm(
            &truth.join(format!("depth-{index:02}.pfm")),
            view.frame.width,
            view.frame.height,
            &view.depth,
        );
    }
    write_mesh_ply(&truth.join("surface.ply"), mesh);
    let (albedo_width, albedo_height) = albedo_size(&fixture.spec.facade);
    write_ppm(
        &truth.join("albedo.ppm"),
        albedo_width,
        albedo_height,
        albedo_texels(&fixture.spec.facade),
    );
    fs::write(
        truth.join("colmap-camera-params.txt"),
        format!(
            "{},{},{},{}\n",
            intrinsics.focal, intrinsics.focal, intrinsics.cx, intrinsics.cy
        ),
    )
    .expect("write COLMAP camera parameters");
    let document = truth_document(fixture, mesh);
    fs::write(
        truth.join("truth.json"),
        serde_json::to_vec_pretty(&document).expect("serialize truth"),
    )
    .expect("write truth");
}

/// Topology of the accepted mesh: connected components, how many of them each reference
/// patch has on its own, and triangles whose vertices come from more than one patch.
struct MeshTopology {
    components: usize,
    /// Sum over reference patches of the components of that patch's own triangles.
    per_patch_components: usize,
    /// Share of all triangles in the largest component.
    largest_component_share: f64,
    cross_reference_triangles: usize,
    /// Vertices of one patch that a cross-reference triangle joins to another patch.
    shared_vertices: usize,
    /// Components of the graph of meshed reference patches, joined by cross-reference
    /// triangles: 1 when every patch is connected to every other.
    reference_patch_components: usize,
    meshed_reference_patches: usize,
}

fn mesh_topology(result: &BrowserReconstructionResult) -> MeshTopology {
    let reconstruction = &result.reconstruction;
    let mut patch_of_point = vec![usize::MAX; reconstruction.dense_points.len()];
    for region in classic_reference_patch_regions(&reconstruction.dense.reference_patches) {
        let reference = region.reference_frame.unwrap_or(usize::MAX);
        let end = (region.points.start + region.points.count).min(patch_of_point.len());
        for slot in &mut patch_of_point[region.points.start.min(end)..end] {
            *slot = reference;
        }
    }
    let triangles = || {
        reconstruction
            .mesh_triangles
            .iter()
            .map(|triangle| [triangle.a, triangle.b, triangle.c])
    };
    let crossing = || {
        triangles().filter(|vertices| {
            let first = patch_of_point.get(vertices[0]);
            vertices
                .iter()
                .any(|&vertex| patch_of_point.get(vertex) != first)
        })
    };
    let cross_reference_triangles = crossing().count();
    let mut shared = std::collections::BTreeSet::new();
    shared.extend(crossing().flatten());
    let shared_vertices = shared.len();
    let reference_patch_components = patch_graph_components(&patch_of_point, triangles());
    let vertex_count = reconstruction.dense_points.len();
    let mut patches: Vec<usize> = patch_of_point.clone();
    patches.sort_unstable();
    patches.dedup();
    let per_patch_components = patches
        .iter()
        .map(|&patch| {
            connected_components(
                vertex_count,
                triangles().filter(|vertices| {
                    vertices
                        .iter()
                        .all(|&vertex| patch_of_point.get(vertex) == Some(&patch))
                }),
            )
        })
        .sum();
    let labels = component_labels(vertex_count, triangles());
    let mut triangles_per_component = std::collections::HashMap::new();
    for [a, _, _] in triangles() {
        *triangles_per_component.entry(labels[a]).or_insert(0_usize) += 1;
    }
    let largest = triangles_per_component.values().copied().max().unwrap_or(0);
    let total = reconstruction.mesh_triangles.len();
    MeshTopology {
        components: connected_components(vertex_count, triangles()),
        per_patch_components,
        largest_component_share: if total == 0 {
            0.0
        } else {
            largest as f64 / total as f64
        },
        cross_reference_triangles,
        shared_vertices,
        reference_patch_components,
        meshed_reference_patches: reconstruction
            .mesh
            .reference_patches
            .iter()
            .filter(|patch| patch.accepted_triangles > 0)
            .count(),
    }
}

/// Components of the graph whose nodes are the reference patches that own triangle
/// vertices and whose edges are the triangles spanning two patches. Vertices without a
/// patch (`usize::MAX` or out of range) are ignored.
fn patch_graph_components(
    patch_of_point: &[usize],
    triangles: impl Iterator<Item = [usize; 3]>,
) -> usize {
    let mut patches = std::collections::BTreeMap::new();
    let mut edges = Vec::new();
    for vertices in triangles {
        let owners: Vec<usize> = vertices
            .iter()
            .filter_map(|&vertex| patch_of_point.get(vertex).copied())
            .filter(|&patch| patch != usize::MAX)
            .collect();
        for &patch in &owners {
            let next = patches.len();
            patches.entry(patch).or_insert(next);
        }
        for pair in owners.windows(2) {
            edges.push([patches[&pair[0]], patches[&pair[1]], patches[&pair[1]]]);
        }
    }
    let labels = component_labels(patches.len(), edges.into_iter());
    labels
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
}

fn format_metric(value: Option<f64>) -> String {
    value.map_or_else(|| "nan".to_owned(), |value| format!("{value:.6}"))
}

fn main() {
    let output_dir = env::args().nth(1);
    let case = Case::parse(env::args().nth(2).as_deref().unwrap_or("slow-lateral-pan"));
    let fixture = generate(case);
    let mesh = true_mesh(&fixture.spec.facade);

    let output_dir = output_dir.filter(|value| value != "-");
    if let Some(output_dir) = output_dir.as_deref() {
        write_fixture(Path::new(output_dir), &fixture, &mesh);
    }

    let result = reconstruct_browser(&ReconstructionRequest {
        frames: fixture
            .views
            .iter()
            .map(|view| view.frame.clone())
            .collect(),
        options: browser_options(),
    })
    .expect("reference fixture reconstruction");
    let reconstruction = &result.reconstruction;
    let state = &result.camera_state;
    let registered_images = state.calibrated_seed_cameras.len() + state.registered_cameras.len();
    let reprojection = reconstruction
        .multi_view
        .bundle_adjustment
        .final_median_reprojection_error_pixels
        .or_else(|| {
            reconstruction
                .calibrated_pair
                .as_ref()
                .map(|pair| pair.median_reprojection_error_pixels)
        })
        .map(f64::from);
    let truth_centers: Vec<Vector3<f64>> =
        fixture.views.iter().map(|view| view.pose.center).collect();
    let pose_rmse = (registered_images >= 3)
        .then(|| {
            let matched: Vec<_> = state
                .calibrated_seed_cameras
                .iter()
                .chain(&state.registered_cameras)
                .map(|camera| {
                    (
                        Vector3::new(camera.x as f64, camera.y as f64, camera.z as f64),
                        truth_centers[camera.frame_index],
                    )
                })
                .collect();
            common::normalized_center_rmse(&matched, common::trajectory_span(&truth_centers))
        })
        .flatten();
    let decisive_gate = reconstruction
        .bootstrap
        .decisive_gate
        .map(|gate| {
            serde_json::to_value(gate)
                .ok()
                .and_then(|value| value.as_str().map(str::to_owned))
                .unwrap_or_else(|| "unknown".to_owned())
        })
        .unwrap_or_else(|| "none".to_owned());
    let topology = mesh_topology(&result);

    if let Some(output_dir) = output_dir.as_deref() {
        let evidence = json!({
            "case": case.name(),
            "registered_frames": state.calibrated_seed_cameras.iter().chain(&state.registered_cameras).map(|camera| camera.frame_index).collect::<Vec<_>>(),
            "bootstrap": reconstruction.bootstrap,
            "seed_candidates": reconstruction.seed_candidates,
            "calibrated_pair": reconstruction.calibrated_pair,
            "dense_reference_patches": reconstruction.dense.reference_patches,
            "mesh": reconstruction.mesh,
            "mesh_topology": {
                "components": topology.components,
                "per_patch_components": topology.per_patch_components,
                "largest_component_share": topology.largest_component_share,
                "cross_reference_triangles": topology.cross_reference_triangles,
                "shared_vertices": topology.shared_vertices,
                "reference_patch_components": topology.reference_patch_components,
                "meshed_reference_patches": topology.meshed_reference_patches,
            },
            "warnings": reconstruction.warnings,
        });
        fs::write(
            Path::new(output_dir).join("rust-evidence.json"),
            serde_json::to_vec_pretty(&evidence).expect("serialize evidence"),
        )
        .expect("write evidence");
        let stages = reference_stages::stage_report(&reference_stages::StageInputs {
            fixture: &fixture,
            result: &result,
            registered_images,
            normalized_pose_rmse: pose_rmse,
            decisive_gate: &decisive_gate,
            topology: &topology,
        });
        fs::write(
            Path::new(output_dir).join("stage-report.json"),
            serde_json::to_vec_pretty(&stages).expect("serialize stage report"),
        )
        .expect("write stage report");
    }

    println!(
        "golden-rust case={} registered_images={} points={} median_reprojection_error_pixels={} normalized_pose_rmse={} decisive_gate={} dense_points={} mesh_triangles={} mesh_components={} per_patch_components={} largest_component_share={:.4} meshed_reference_patches={} cross_reference_triangles={}",
        case.name(),
        registered_images,
        reconstruction.points.len(),
        format_metric(reprojection),
        format_metric(pose_rmse),
        decisive_gate,
        reconstruction.dense_points.len(),
        reconstruction.mesh_triangles.len(),
        topology.components,
        topology.per_patch_components,
        topology.largest_component_share,
        topology.meshed_reference_patches,
        topology.cross_reference_triangles,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(case: Case) -> (FixtureSpec, Intrinsics, Vec<Pose>) {
        let spec = case.spec();
        let plan = sampling_plan();
        let intrinsics = Intrinsics::for_plan(&plan);
        let poses = plan
            .times
            .iter()
            .map(|&time| Pose::at(&spec, time))
            .collect();
        (spec, intrinsics, poses)
    }

    #[test]
    fn sampling_matches_the_browser_plan() {
        let plan = sampling_plan();
        assert_eq!((plan.width, plan.height), (360, 203));
        assert_eq!(plan.times.len(), 18);
        assert!((plan.times[0] - 0.8).abs() < 1e-9);
        assert!((plan.times[17] - 15.2).abs() < 1e-9);
        assert!((Intrinsics::for_plan(&plan).focal - 309.6).abs() < 1e-9);
    }

    #[test]
    fn browser_options_match_the_core_defaults_the_browser_restates() {
        let browser = browser_options();
        let defaults = ReconstructionOptions::default();
        assert_eq!(browser.max_features, defaults.max_features);
        assert_eq!(browser.min_feature_distance, defaults.min_feature_distance);
        assert_eq!(browser.match_radius, defaults.match_radius);
        assert_eq!(browser.focal_length_pixels, None);
    }

    #[test]
    fn truth_depth_reprojects_consistently_between_views() {
        for case in Case::ALL {
            let (spec, intrinsics, poses) = spec(case);
            let (from, to) = (&poses[4], &poses[5]);
            for (u, v) in [(40.5, 30.5), (180.5, 101.5), (300.5, 170.5)] {
                let world = surface_hit(&spec.facade, &intrinsics, from, u, v);
                let depth = from.world_to_camera(&world).z;
                let back = from.center + from.world_direction(&(intrinsics.ray(u, v) * depth));
                assert!(
                    (back - world).norm() < 1e-6,
                    "{case:?} depth back-projection"
                );
                let camera = to.world_to_camera(&world);
                let (pu, pv) = intrinsics.project(&camera).expect("in front");
                let again = surface_hit(&spec.facade, &intrinsics, to, pu, pv);
                assert!(
                    (again - world).norm() < 1e-4,
                    "{case:?}: surface point is not consistent across views"
                );
            }
        }
    }

    #[test]
    fn motions_match_their_intent() {
        let (_, intrinsics, pan) = spec(Case::SlowLateralPan);
        let (_, _, rotation) = spec(Case::PureRotation);
        for pair in pan.windows(2) {
            assert!(rotation_degrees(&pair[0], &pair[1]) < 1e-9);
            assert!(((pair[1].center - pair[0].center).norm() - PAN_STEP).abs() < 1e-9);
        }
        for pair in rotation.windows(2) {
            assert_eq!(pair[0].center, pair[1].center);
            let degrees = rotation_degrees(&pair[0], &pair[1]);
            assert!((degrees - (PAN_STEP / TREVI_DISTANCE).atan().to_degrees()).abs() < 1e-6);
        }
        // Same image motion of the façade plane: pan step / distance equals the yaw step.
        let pan_shift = intrinsics.focal * PAN_STEP / TREVI_DISTANCE;
        let yaw_shift = intrinsics.focal * (PAN_STEP / TREVI_DISTANCE).atan().tan();
        assert!((pan_shift - yaw_shift).abs() < 1e-9);
    }

    #[test]
    fn slow_pan_parallax_sits_near_the_baseline_gate() {
        let (spec, intrinsics, poses) = spec(Case::SlowLateralPan);
        let near = spec.facade.distance - spec.facade.relief;
        let far = spec.facade.distance;
        let per_pair = intrinsics.focal * PAN_STEP * (1.0 / near - 1.0 / far);
        // Below the 1.25 px rotation-only gate per interval, but two intervals exceed it.
        assert!(per_pair > 0.5 && per_pair < 1.25, "{per_pair}");
        assert!(2.0 * per_pair > 1.25, "{per_pair}");
        let clip = per_pair * (poses.len() - 1) as f64;
        assert!(clip > 8.0, "{clip}");
    }

    #[test]
    fn neighbouring_views_see_the_same_surface() {
        let (spec, intrinsics, poses) = spec(Case::OverlappingReferences);
        for pair in poses.windows(2) {
            let overlap = view_overlap(&spec.facade, &intrinsics, &pair[0], &pair[1]);
            assert!(overlap > 0.85, "{overlap}");
        }
        let ends = view_overlap(
            &spec.facade,
            &intrinsics,
            &poses[0],
            &poses[poses.len() - 1],
        );
        // The pan covers new surface, yet first and last views still share some of it.
        assert!(ends > 0.05 && ends < 0.5, "{ends}");
    }

    #[test]
    fn every_view_stays_inside_the_exported_surface() {
        for case in Case::ALL {
            let (spec, intrinsics, poses) = spec(case);
            for pose in &poses {
                for (u, v) in [
                    (0.0, 0.0),
                    (intrinsics.width as f64, 0.0),
                    (0.0, intrinsics.height as f64),
                    (intrinsics.width as f64, intrinsics.height as f64),
                ] {
                    // `surface_hit` asserts the hit lies inside the façade extent.
                    surface_hit(&spec.facade, &intrinsics, pose, u, v);
                }
            }
        }
    }

    #[test]
    fn true_surface_is_one_connected_component() {
        for case in Case::ALL {
            let mesh = true_mesh(&case.spec().facade);
            assert_eq!(
                connected_components(
                    mesh.vertices.len(),
                    mesh.triangles.iter().map(|t| t.map(|index| index as usize)),
                ),
                1
            );
            assert_eq!(
                mesh.triangles.len(),
                2 * (mesh.columns - 1) * (mesh.rows - 1)
            );
        }
    }

    #[test]
    fn patch_graph_needs_every_reference_patch_connected() {
        // Three patches of two vertices each (0-1, 2-3, 4-5) plus an unowned vertex 6.
        let patch_of_point = [10, 10, 20, 20, 30, 30, usize::MAX];
        let within = [[0, 1, 0], [2, 3, 2], [4, 5, 4]];
        assert_eq!(
            patch_graph_components(&patch_of_point, within.into_iter()),
            3
        );
        // One crossing triangle joins two patches; the third stays apart.
        let partial = [[0, 1, 0], [2, 3, 2], [4, 5, 4], [1, 2, 6]];
        assert_eq!(
            patch_graph_components(&patch_of_point, partial.into_iter()),
            2
        );
        let full = [[0, 1, 0], [2, 3, 2], [4, 5, 4], [1, 2, 3], [3, 4, 5]];
        assert_eq!(patch_graph_components(&patch_of_point, full.into_iter()), 1);
        assert_eq!(
            patch_graph_components(&patch_of_point, std::iter::empty()),
            0
        );
    }

    #[test]
    fn connected_components_counts_separate_sheets() {
        let triangles = [[0, 1, 2], [1, 2, 3], [4, 5, 6]];
        assert_eq!(connected_components(8, triangles.into_iter()), 2);
    }

    #[test]
    fn rendering_is_deterministic_and_textured() {
        let (spec, intrinsics, poses) = spec(Case::SlowLateralPan);
        let small = Intrinsics {
            width: 48,
            height: 27,
            focal: intrinsics.focal * 48.0 / 360.0,
            cx: 24.0,
            cy: 13.5,
        };
        let first = render(&spec.facade, &small, poses[0]);
        let second = render(&spec.facade, &small, poses[0]);
        assert_eq!(first.frame.rgba, second.frame.rgba);
        assert_eq!(first.depth, second.depth);
        let luma: Vec<u8> = first
            .frame
            .rgba
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| p[1])
            .collect();
        let (low, high) = luma.iter().fold((u8::MAX, 0), |(low, high), &value| {
            (low.min(value), high.max(value))
        });
        assert!(high - low > 80, "texture contrast {low}..{high}");
        assert!(first
            .depth
            .iter()
            .all(|&depth| depth >= 5.55 - 1e-3 && depth <= 6.0 + 1e-3));
    }
}
