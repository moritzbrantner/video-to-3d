//! Independent #137 repair acceptance: previously stored native mesh semantics
//! cannot remain current after introducing clipped derived geometry.
use video_to_3d_core::scene_project::OperationKind;

#[test]
fn native_mesh_identity_versions_new_clipping_semantics() {
    assert!(
        OperationKind::SurfaceMesh.implementation_revision() > 2,
        "revision two stored meshes predate source-triangle/barycentric clipping; rebuild them instead of reporting their old seams current"
    );
}
