//! Independent ownership/cancellation contracts; API names may be translated.
use super::*;

#[allow(dead_code)]
mod golden {
    use crate as video_to_3d_core;
    pub fn request() -> crate::ReconstructionRequest {
        let points = scene();
        crate::ReconstructionRequest {
            frames: (0..FRAME_COUNT)
                .map(|i| render_frame(i, &points, Scenario::Lateral))
                .collect(),
            options: Default::default(),
        }
    }
    include!("../examples/golden_fixture.rs");
}

#[test]
fn owned_consolidation_preserves_dense_allocation_and_borrowed_semantics() {
    let request = golden::request();
    let cameras = reconstruct_cameras(&request).unwrap();
    let dense = reconstruct_dense(&request, &cameras).unwrap();
    let before = dense.to_json();
    let borrowed = reconstruct_mesh(&cameras, &dense).unwrap();
    assert_eq!(dense.to_json(), before, "borrowed mesh remains immutable");
    let pointer = dense.dense.points.as_ptr();
    let capacity = dense.dense.points.capacity();
    let (owned, mesh) = reconstruct_mesh_owned(&cameras, dense).unwrap();
    assert_eq!(owned.dense.points.as_ptr(), pointer);
    assert_eq!(owned.dense.points.capacity(), capacity);
    assert_eq!(mesh.to_json(), borrowed.to_json());
    let original = DenseStageArtifact::from_json(&before).unwrap();
    for moved in &borrowed.mesh.consolidated_points {
        assert_eq!(owned.dense.points[moved.index], moved.point);
    }
    assert_eq!(owned.dense.points.len(), original.dense.points.len());
}

#[test]
fn real_stages_poll_cancellation_inside_work_and_preserve_uncanceled_bytes() {
    let request = golden::request();
    let cameras = reconstruct_cameras(&request).unwrap();
    let dense = reconstruct_dense(&request, &cameras).unwrap();
    let mesh = reconstruct_mesh(&cameras, &dense).unwrap();
    assert_eq!(
        reconstruct_cameras_with_cancel(&request, || false)
            .unwrap()
            .to_json(),
        cameras.to_json()
    );
    assert_eq!(
        reconstruct_dense_with_cancel(&request, &cameras, || false)
            .unwrap()
            .to_json(),
        dense.to_json()
    );
    assert_eq!(
        reconstruct_mesh_with_cancel(&cameras, &dense, || false)
            .unwrap()
            .to_json(),
        mesh.to_json()
    );
    // Different intermediate cancellation requests; no clock or exact poll-count assertion.
    for cancel_at in [2, 4] {
        let mut polls = 0;
        assert!(reconstruct_cameras_with_cancel(&request, || {
            polls += 1;
            polls >= cancel_at
        })
        .is_err());
        assert!(polls >= cancel_at);
        let mut polls = 0;
        assert!(reconstruct_dense_with_cancel(&request, &cameras, || {
            polls += 1;
            polls >= cancel_at
        })
        .is_err());
        assert!(polls >= cancel_at);
        let mut polls = 0;
        assert!(reconstruct_mesh_with_cancel(&cameras, &dense, || {
            polls += 1;
            polls >= cancel_at
        })
        .is_err());
        assert!(polls >= cancel_at);
    }
}
