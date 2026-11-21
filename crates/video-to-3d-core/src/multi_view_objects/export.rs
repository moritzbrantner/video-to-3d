use super::{HullMesh, HullOutcome, View};
use crate::object_identity::{FrameCamera, ObjectIdentityArtifact};
use crate::placement_validation::PlacementVerdict;
use crate::semantic_inventory::SemanticInventoryArtifact;
use crate::{
    EvidenceCamera, EvidenceCameraAuthority, EvidenceOrigin, EvidenceRange, EvidenceScale,
    MeshTriangle, Point3, ReconstructionEvidenceView, ReconstructionProviderClass,
    ReconstructionProviderDescriptor, SurfaceEvidenceRegion,
};
use serde_json::json;
use std::collections::BTreeMap;

/// Frame index, dimensions and local RGBA8 pixels in sampled camera coordinates.
pub type FramePixels<'a> = (usize, usize, usize, &'a [u8]);

pub fn hull_glb(
    mesh: &HullMesh,
    outcome: &HullOutcome,
    frames: &[FramePixels<'_>],
    inventory: &SemanticInventoryArtifact,
    identities: &ObjectIdentityArtifact,
    cameras: &[FrameCamera],
) -> Result<Vec<u8>, String> {
    let object = identities
        .objects
        .iter()
        .find(|obj| obj.id == outcome.object)
        .ok_or("No matching object identity")?;
    let observations: BTreeMap<_, _> = inventory
        .observations
        .iter()
        .map(|obs| (obs.id.as_str(), obs))
        .collect();
    let mut colors = vec![([0_u32; 3], 0_u32); mesh.vertices.len()];
    for id in &object.observations {
        let obs = observations
            .get(id.as_str())
            .ok_or("Object observation is missing")?;
        let Some(cam) = cameras.iter().find(|c| c.frame_index == obs.frame_index) else {
            continue;
        };
        let Some(view) = View::new(cam, obs) else {
            continue;
        };
        let Some((_, w, h, rgba)) = frames.iter().find(|f| f.0 == obs.frame_index) else {
            continue;
        };
        if view.width != *w || view.height != *h || rgba.len() != w * h * 4 {
            return Err("Object mask and color frame dimensions differ".into());
        }
        // The nearest surface at a pixel supplies its color. The far side
        // remains neutral unless another camera actually sees that surface.
        let mut zbuf = vec![f64::INFINITY; w * h];
        let projected: Vec<_> = mesh
            .vertices
            .iter()
            .map(|&p| {
                let i = view.pixel(p)?;
                if !view.mask[i] {
                    return None;
                }
                let (_, z) = crate::object_identity::project(cam, p)?;
                zbuf[i] = zbuf[i].min(z);
                Some((i, z))
            })
            .collect();
        for ((sum, n), sample) in colors.iter_mut().zip(projected) {
            if let Some((i, z)) = sample {
                if z <= zbuf[i] * 1.005 {
                    for c in 0..3 {
                        sum[c] += u32::from(rgba[4 * i + c]);
                    }
                    *n += 1;
                }
            }
        }
    }
    let accepted = mesh.validation.verdict == PlacementVerdict::Accepted;
    let confidence = outcome.holdout_mask_iou.unwrap_or(0.) as f32;
    // Materialize the interchange only at export, where glTF needs owned buffers.
    let points: Vec<_> = mesh
        .vertices
        .iter()
        .zip(colors)
        .map(|(p, (sum, n))| {
            let [r, g, b] = sum.map(|v| v.checked_div(n).unwrap_or(160) as u8);
            Point3 {
                x: p[0] as f32,
                y: p[1] as f32,
                z: p[2] as f32,
                r,
                g,
                b,
                confidence,
            }
        })
        .collect();
    let triangles: Vec<_> = mesh
        .triangles
        .iter()
        .map(|f| MeshTriangle {
            a: f[0] as usize,
            b: f[1] as usize,
            c: f[2] as usize,
            confidence,
        })
        .collect();
    let evidence_cameras = cameras
        .iter()
        .filter(|c| accepted && outcome.views.contains(&c.frame_index))
        .map(|c| EvidenceCamera {
            frame_index: c.frame_index,
            authority: EvidenceCameraAuthority::RegisteredGeometry,
            rotation: c.rotation.map(|v| v as f32),
            translation: c.translation.map(|v| v as f32),
            confidence: None,
            median_reprojection_error_pixels: None,
        })
        .collect();
    let (class, origin) = if accepted {
        (
            ReconstructionProviderClass::GeometricMultiView,
            EvidenceOrigin::RevalidatedCompletion,
        )
    } else {
        (
            ReconstructionProviderClass::GenerativeCompletion,
            EvidenceOrigin::GenerativeCompletion,
        )
    };
    let reference = outcome.views.first().copied().filter(|_| accepted);
    let evidence = ReconstructionEvidenceView::new(
        ReconstructionProviderDescriptor::new("multi-view-silhouette", class, Some("1".into()))?,
        EvidenceScale::ArbitraryMonocular,
        evidence_cameras,
        vec![SurfaceEvidenceRegion {
            origin,
            reference_frame: reference,
            source_frames: outcome
                .views
                .iter()
                .copied()
                .filter(|f| accepted && Some(*f) != reference)
                .collect(),
            points: EvidenceRange::new(0, points.len()),
        }],
        &points,
        &triangles,
    )?;
    let mut bin = Vec::new();
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in evidence.points {
        for (k, v) in [p.x, p.y, p.z].iter().enumerate() {
            lo[k] = lo[k].min(*v);
            hi[k] = hi[k].max(*v);
            bin.extend(v.to_le_bytes());
        }
    }
    let colors_start = bin.len();
    // Normalized COLOR_0 is linear RGB in glTF.
    let linear = |v: u8| {
        let c = f32::from(v) / 255.;
        let value = if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        };
        (255. * value).round() as u8
    };
    for p in evidence.points {
        bin.extend([linear(p.r), linear(p.g), linear(p.b), 255]);
    }
    let indices_start = bin.len();
    for f in evidence.triangles {
        for i in [f.a, f.b, f.c] {
            bin.extend((i as u32).to_le_bytes());
        }
    }
    let meta = json!({"role":"object_candidate","method":"multi_view_silhouette","object_id":object.id,"label":object.label,"geometry_origin":origin,"source_frames":outcome.views,"holdout_frame":outcome.holdout_frame,"holdout_mask_iou":outcome.holdout_mask_iou,"placement_verdict":mesh.validation.verdict});
    let doc = json!({"asset":{"version":"2.0","extras":{"video_to_3d":meta}},"scene":0,"scenes":[{"nodes":[0]}],
        "nodes":[{"name":format!("{} {}",object.label,object.id),"mesh":0,"scale":[1,-1,-1],"extras":{"video_to_3d":meta}}],
        "meshes":[{"primitives":[{"attributes":{"POSITION":0,"COLOR_0":1},"indices":2,"material":0}]}],
        "materials":[{"doubleSided":true,"pbrMetallicRoughness":{"metallicFactor":0,"roughnessFactor":1}}],
        "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":colors_start,"target":34962},{"buffer":0,"byteOffset":colors_start,"byteLength":indices_start-colors_start,"target":34962},{"buffer":0,"byteOffset":indices_start,"byteLength":bin.len()-indices_start,"target":34963}],
        "accessors":[{"bufferView":0,"componentType":5126,"count":points.len(),"type":"VEC3","min":lo,"max":hi},{"bufferView":1,"componentType":5121,"normalized":true,"count":points.len(),"type":"VEC4"},{"bufferView":2,"componentType":5125,"count":triangles.len()*3,"type":"SCALAR"}]});
    crate::textured_glb::pack_scene_glb(
        doc,
        bin,
        &mut crate::ReconstructionControl::new(&mut || false),
    )
}
