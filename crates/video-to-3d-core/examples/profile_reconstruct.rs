use std::hint::black_box;

use video_to_3d_core::{reconstruct, FrameInput, ReconstructionOptions, ReconstructionRequest};

const WIDTH: u32 = 192;
const HEIGHT: u32 = 144;
const FRAME_COUNT: u32 = 10;

fn texture(x: u32, y: u32) -> [u8; 3] {
    let mixed = x
        .wrapping_mul(73_856_093)
        .wrapping_add(y.wrapping_mul(19_349_663))
        .rotate_left((x.wrapping_add(y) % 31) + 1);
    let checker = (((x / 6) + (y / 6)) & 1) as u8 * 48;
    [
        mixed as u8 ^ checker,
        mixed.rotate_left(9) as u8 ^ checker.saturating_add(17),
        mixed.rotate_left(17) as u8 ^ checker.saturating_add(31),
    ]
}

fn synthetic_frame(frame_index: u32) -> FrameInput {
    let mut rgba = vec![0_u8; (WIDTH * HEIGHT * 4) as usize];
    let camera_shift = frame_index * 2;
    let foreground_shift = frame_index * 4;

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let foreground = (52..140).contains(&x) && (36..108).contains(&y);
            let shift = if foreground {
                foreground_shift
            } else {
                camera_shift
            };
            let source_x = x.wrapping_add(shift) % WIDTH;
            let source_y = y.wrapping_add(frame_index % 3) % HEIGHT;
            let [r, g, b] = texture(source_x, source_y);
            let offset = ((y * WIDTH + x) * 4) as usize;
            rgba[offset] = r;
            rgba[offset + 1] = g;
            rgba[offset + 2] = b;
            rgba[offset + 3] = 255;
        }
    }

    FrameInput {
        width: WIDTH,
        height: HEIGHT,
        rgba,
    }
}

fn main() {
    let request = ReconstructionRequest {
        frames: (0..FRAME_COUNT).map(synthetic_frame).collect(),
        options: ReconstructionOptions {
            max_features: 280,
            min_feature_distance: 6,
            descriptor_radius: 3,
            match_radius: 48,
            ..ReconstructionOptions::default()
        },
    };

    let mut checksum = 0_usize;
    for _ in 0..3 {
        let result =
            reconstruct(black_box(&request)).expect("deterministic reconstruction workload");
        checksum = checksum
            .wrapping_add(result.cameras.len())
            .wrapping_add(result.points.len())
            .wrapping_add(result.pairs.iter().map(|pair| pair.matches).sum::<usize>())
            .wrapping_add(result.multi_view.track_count);
    }

    println!("reconstruction-checksum={checksum}");
}
