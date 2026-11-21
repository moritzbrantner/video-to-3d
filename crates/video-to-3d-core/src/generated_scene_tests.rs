use super::*;
use crate::object_generation::GenerationRequest;
use crate::scene_project::ProjectPath;
use crate::semantic_inventory::{MaskEncoding, SemanticMask};
use crate::EvidenceCameraAuthority;
use gltf::binary::Header;
use std::fs;

fn triangle(meta: Value) -> Vec<u8> {
    let mut bin: Vec<_> = [0f32, 0., 0., 1., 0., 0., 0., 1., 0.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    bin.extend([0, 1, 2, 0]);
    let doc = json!({"asset":{"version":"2.0","extras":{"video_to_3d":meta}},"scene":0,"scenes":[{"nodes":[0,1]}],
        "nodes":[{"mesh":0},{"extras":{"video_to_3d":{"role":"observed_object_surface","object_id":"chair"}}}],
        "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],"materials":[],
        "buffers":[{"byteLength":40}],"bufferViews":[{"buffer":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":3}],
        "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},
        {"bufferView":1,"componentType":5121,"count":3,"type":"SCALAR"}]});
    Glb {
        header: Header {
            magic: *b"glTF",
            version: 2,
            length: 0,
        },
        json: Cow::Owned(serde_json::to_vec(&doc).unwrap()),
        bin: Some(Cow::Owned(bin)),
    }
    .to_vec()
    .unwrap()
}

#[test]
fn completed_export_selects_validated_mesh_in_matched_gauge_and_preserves_observed_scene() {
    let root = std::env::temp_dir().join(format!(
        "video-to-3d-completed-export-{}",
        std::process::id()
    ));
    fs::create_dir_all(&root).unwrap();
    let candidate = triangle(Value::Null);
    fs::write(root.join("chair.glb"), &candidate).unwrap();
    let identity = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
    let cameras: Vec<_> = (0..2)
        .map(|frame_index| FrameCamera {
            frame_index,
            rotation: identity,
            translation: [frame_index as f64, 0., 0.],
            focal_pixels: 200.,
            principal_point: [100.; 2],
        })
        .collect();
    let evidence: Vec<_> = cameras
        .iter()
        .map(|c| EvidenceCamera {
            frame_index: c.frame_index,
            authority: EvidenceCameraAuthority::RegisteredGeometry,
            rotation: c.rotation.map(|v| v as f32),
            translation: c.translation.map(|v| v as f32),
            confidence: None,
            median_reprojection_error_pixels: None,
        })
        .collect();
    let hash = ContentHash::of_bytes(b"inventory");
    let base = triangle(
        json!({"schema":crate::textured_glb::TEXTURED_SURFACE_SCHEMA,"accepted_camera_evidence":evidence,"semantic_inventory":hash}),
    );
    let mesh = GeneratedMesh {
        path: ProjectPath::new("chair.glb").unwrap(),
        content_hash: ContentHash::of_bytes(&candidate),
        vertices: 3,
        triangles: 1,
    };
    let request = GenerationRequest {
        request_id: "g".into(),
        object: "chair".into(),
        label: "chair".into(),
        observation: "s1".into(),
        frame_index: 0,
        source_frame_hash: hash.clone(),
        width: 2,
        height: 2,
        crop: [0, 0, 2, 2],
        mask_hash: hash.clone(),
        mask: SemanticMask {
            encoding: MaskEncoding::RleRowMajor,
            width: 2,
            height: 2,
            counts: vec![0, 4],
        },
        observed_points: 3,
    };
    let generation=ObjectGenerationArtifact::from_json(&json!({"schema_version":1,"status":"available",
       "object_references":hash,"provider":{"id":"triposr","revision":"test","models":[],"runtime":null},
       "objects":[{"request":request,"classification":"generative_candidate","output":mesh}]}).to_string()).unwrap();
    assert_eq!(
        candidate_geometry(&candidate, &mesh, &generation)
            .unwrap()
            .positions
            .len(),
        3
    );
    let accepted = json!({"verdict":"accepted","needs_review":false,"views":[],"reason":"two consistent views"});
    let mut placement=ObjectPlacementArtifact::from_json(&json!({"schema_version":2,"status":"available",
        "object_generation":ContentHash::of_bytes(generation.to_json().as_bytes()),"up":[0,1,0],"up_source":"camera_up",
        "supports":[],"metric_hint_labels":[],"unplaced":[],"placements":[{"object":"chair","label":"chair","generation_request":"g",
        "rotation":identity,"translation":[1,2,3],"scale":2,"scale_source":"observed","observed_points":3,"reference_frame":0,
        "contact":"no_contact_evidence","reason":"stationary","validation":accepted}]}).to_string()).unwrap();
    let mut silhouettes: MultiViewObjectReport = serde_json::from_value(
        json!({"schema_version":1,"semantic_inventory":hash,"object_identity":hash,"objects":[]}),
    )
    .unwrap();
    let export = |p: &ObjectPlacementArtifact, s: &MultiViewObjectReport| {
        append_validated_objects(&base, &root, &generation, p, &hash, &cameras, s)
    };
    let result = export(&placement, &silhouettes).unwrap();
    let glb = Glb::from_slice(&result).unwrap();
    let doc: Value = serde_json::from_slice(&glb.json).unwrap();
    let node = &doc["nodes"][2];
    assert_eq!(
        node["matrix"],
        json!([2., 0., 0., 0., 0., -2., 0., 0., 0., 0., -2., 0., 1., -2., -3., 1.])
    );
    assert_eq!(doc["scenes"][0]["nodes"], json!([0, 2]));
    assert_eq!(doc["scenes"][1]["nodes"], json!([0, 1]));
    assert_eq!(
        node["extras"]["video_to_3d"]["geometry_origin"],
        "generative_completion"
    );
    // Parse indexed data through the GLB importer after appending it to real preexisting views.
    let model = gltf::Gltf::from_slice(&result).unwrap();
    let prim = model.meshes().nth(1).unwrap().primitives().next().unwrap();
    let reader = prim.reader(|_| model.blob.as_deref());
    assert_eq!(
        reader
            .read_indices()
            .unwrap()
            .into_u32()
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(
        reader.read_positions().unwrap().last().unwrap(),
        [0., 1., 0.]
    );
    let mut different = cameras.clone();
    different[0].translation[1] += 0.1;
    assert!(append_validated_objects(
        &base,
        &root,
        &generation,
        &placement,
        &hash,
        &different,
        &silhouettes
    )
    .is_err());
    let mut rounded_cameras = cameras.clone();
    rounded_cameras[0].rotation[1] = 1e-8;
    let native = triangle(json!({
        "schema": crate::textured_glb::TEXTURED_SURFACE_SCHEMA,
        "accepted_camera_evidence": evidence,
        "provider": {"id":"colmap-patch-match"},
    }));
    let completed = append_validated_objects(
        &native,
        &root,
        &generation,
        &placement,
        &hash,
        &rounded_cameras,
        &silhouettes,
    )
    .unwrap();
    let document: Value =
        serde_json::from_slice(&Glb::from_slice(&completed).unwrap().json).unwrap();
    assert_eq!(
        document["asset"]["extras"]["video_to_3d"]["semantic_inventory"],
        hash.as_str()
    );
    assert_eq!(document["scenes"][0]["nodes"], json!([0, 2]));
    silhouettes.objects.push(serde_json::from_value(json!({"object":"chair","label":"chair","views":[0,1],"reason":"held-out silhouette matches","output":mesh,"validation":accepted})).unwrap());
    let priority = export(&placement, &silhouettes).unwrap();
    let d: Value = serde_json::from_slice(&Glb::from_slice(&priority).unwrap().json).unwrap();
    assert_eq!(d["nodes"].as_array().unwrap().len(), 3);
    assert_eq!(
        d["nodes"][2]["extras"]["video_to_3d"]["method"],
        "multi_view_silhouette"
    );
    // The common fitting views favor the generated geometry despite a stronger
    // hull score in its reserved validation frame.
    let residual = |frame: usize, overlap: f64| {
        serde_json::from_value(
            json!({"observation":format!("s{frame}"),"frame_index":frame,
            "verdict":"consistent","projected_vertices":3,"inside_share":overlap,"box_iou":overlap,
            "landmarks":0,"depth_violation_share":0.0,"reason":"consistent"}),
        )
        .unwrap()
    };
    let hull = &mut silhouettes.objects[0];
    hull.holdout_frame = Some(2);
    hull.validation.as_mut().unwrap().views =
        vec![residual(0, 0.65), residual(1, 0.65), residual(2, 1.)];
    placement.placements[0].validation.views =
        vec![residual(0, 0.8), residual(1, 0.8), residual(2, 0.1)];
    let comparison = compare_candidates(hull, &placement.placements[0]).unwrap();
    assert!(comparison.prefers_generated());
    assert_eq!(comparison.fitting_frames, vec![0, 1]);
    let bytes = export(&placement, &silhouettes).unwrap();
    let d: Value = serde_json::from_slice(&Glb::from_slice(&bytes).unwrap().json).unwrap();
    assert_eq!(
        d["nodes"][2]["extras"]["video_to_3d"]["geometry_origin"],
        "generative_completion"
    );
    placement.placements[0].validation.views.clear();
    silhouettes.objects[0]
        .validation
        .as_mut()
        .unwrap()
        .views
        .clear();
    // The object shelf also exposes isolated, unplaced hypotheses, without
    // duplicating an identity or silently promoting a contradicted mesh.
    silhouettes.objects.push(serde_json::from_value(json!({
        "object":"lamp","label":"lamp","views":[0,1],"reason":"weak fit","output":mesh,
        "validation":{"verdict":"contradicted","needs_review":true,"reason":"mask disagreement","views":[]}
    })).unwrap());
    let library = export_object_library(&root, &generation, &placement, &silhouettes).unwrap();
    let lib: Value = serde_json::from_slice(&Glb::from_slice(&library).unwrap().json).unwrap();
    assert_eq!(
        lib["asset"]["extras"]["video_to_3d"]["world_placement"],
        false
    );
    assert_eq!(lib["nodes"].as_array().unwrap().len(), 2);
    assert_eq!(
        lib["nodes"][0]["extras"]["video_to_3d"]["method"],
        "multi_view_silhouette"
    );
    assert_eq!(
        lib["nodes"][1]["extras"]["video_to_3d"]["placement_verdict"],
        "contradicted"
    );
    assert_ne!(lib["nodes"][0]["matrix"], lib["nodes"][1]["matrix"]);
    gltf::Gltf::from_slice(&library).unwrap();
    placement.placements[0].validation.verdict = PlacementVerdict::Contradicted;
    placement.placements[0].validation.needs_review = true;
    silhouettes.objects.clear();
    assert!(export(&placement, &silhouettes)
        .unwrap_err()
        .contains("no object"));
    fs::remove_dir_all(root).unwrap();
}
