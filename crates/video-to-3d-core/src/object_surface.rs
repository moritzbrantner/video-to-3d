//! Attribute existing observed surface triangles to stationary, mask-supported identities.
use crate::object_identity::{
    identify_objects, inside, project, FrameCamera, MotionClass, FRONT_DEPTH_BAND,
};
use crate::object_masks::{check_object_masks, MaskConsistency, MaskQuality};
use crate::scene_project::ContentHash;
use crate::semantic_inventory::{SemanticInventoryArtifact, SemanticObservation};
use crate::{EvidenceOrigin, ReconstructionEvidenceView};
use std::collections::{BTreeMap, BTreeSet};

pub struct ObjectSurfacePartition {
    pub(crate) groups: Vec<Option<usize>>,
    pub(crate) objects: Vec<(String, String)>,
    pub(crate) inventory_hash: ContentHash,
}

impl ObjectSurfacePartition {
    pub fn diagnostic(&self) -> String {
        let assigned = self.groups.iter().flatten().count();
        let count = self.groups.iter().flatten().collect::<BTreeSet<_>>().len();
        format!("{count} independently selectable object surface(s), {assigned} accepted triangle(s) supported by multiple masks.")
    }
}

struct MaskView<'a> {
    camera: &'a FrameCamera,
    observation: &'a SemanticObservation,
    depth: f64,
}

impl MaskView<'_> {
    fn can_judge(&self, center: [f64; 3]) -> bool {
        let Some(([x, y], depth)) = project(self.camera, center) else {
            return false;
        };
        self.observation.mask.as_ref().is_some_and(|mask| {
            x >= 0.
                && x < f64::from(mask.width)
                && y >= 0.
                && y < f64::from(mask.height)
                && (depth - self.depth).abs() <= FRONT_DEPTH_BAND * self.depth
        })
    }

    fn supports(&self, corners: &[[f64; 3]; 3]) -> bool {
        let mut pixels = [[0.; 2]; 3];
        for (point, pixel) in corners.iter().zip(&mut pixels) {
            let Some((uv, z)) = project(self.camera, *point) else {
                return false;
            };
            if (z - self.depth).abs() > FRONT_DEPTH_BAND * self.depth
                || !inside(self.observation, uv)
            {
                return false;
            }
            *pixel = uv;
        }
        // Test the full projected footprint so a face spanning a gap between
        // chair legs does not acquire the chair's identity.
        let lo = [0, 1].map(|i| {
            pixels
                .iter()
                .map(|p| p[i].floor() as u32)
                .min()
                .unwrap_or(0)
        });
        let hi = [0, 1].map(|i| {
            pixels
                .iter()
                .map(|p| p[i].floor() as u32)
                .max()
                .unwrap_or(0)
        });
        for y in lo[1]..=hi[1] {
            for x in lo[0]..=hi[0] {
                let p = [x as f64 + 0.5, y as f64 + 0.5];
                let side = [0, 1, 2].map(|i| {
                    let a = pixels[i];
                    let b = pixels[(i + 1) % 3];
                    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
                });
                if (side.iter().all(|s| *s >= 0.) || side.iter().all(|s| *s <= 0.))
                    && !inside(self.observation, p)
                {
                    return false;
                }
            }
        }
        true
    }
}

/// Partition an accepted mesh without changing its points, faces, or materials.
/// The caller binds the inventory to the decoded frame bytes before invoking export.
pub fn partition_object_surfaces(
    evidence: &ReconstructionEvidenceView<'_>,
    inventory: &SemanticInventoryArtifact,
    inventory_hash: ContentHash,
    dimensions: [u32; 2],
    focal: Option<f32>,
) -> Result<ObjectSurfacePartition, String> {
    let mut partition = ObjectSurfacePartition {
        groups: vec![None; evidence.triangles.len()],
        objects: Vec::new(),
        inventory_hash,
    };
    if inventory
        .frames
        .iter()
        .any(|f| [f.width, f.height] != dimensions)
    {
        return Err("semantic mask dimensions differ from reconstructed keyframes".into());
    }
    let Some(focal) = focal.filter(|f| f.is_finite() && *f > 0.) else {
        return Ok(partition);
    };
    if inventory.observations.is_empty() {
        return Ok(partition);
    }
    let cameras: Vec<_> = evidence
        .cameras
        .iter()
        .map(|c| FrameCamera {
            frame_index: c.frame_index,
            rotation: c.rotation.map(f64::from),
            translation: c.translation.map(f64::from),
            focal_pixels: f64::from(focal),
            principal_point: dimensions.map(|d| d as f64 / 2.),
        })
        .collect();
    let observed = |index| {
        evidence.regions.iter().any(|r| {
            r.origin == EvidenceOrigin::GeometricMultiView
                && (r.points.start..r.points.start + r.points.count).contains(&index)
        })
    };
    let xyz = |i: usize| {
        let p = &evidence.points[i];
        [p.x, p.y, p.z].map(f64::from)
    };
    let points: Vec<_> = (0..evidence.points.len())
        .filter(|i| observed(*i))
        .map(xyz)
        .collect();
    let identities = identify_objects(
        inventory,
        partition.inventory_hash.clone(),
        &cameras,
        &points,
    );
    let identity_hash = ContentHash::of_bytes(identities.to_json().as_bytes());
    let checked = check_object_masks(
        inventory,
        partition.inventory_hash.clone(),
        &identities,
        identity_hash,
        &cameras,
        &points,
    );
    let ambiguous: BTreeSet<_> = identities
        .ambiguities
        .iter()
        .flat_map(|a| &a.objects)
        .collect();
    let observations: BTreeMap<_, _> = inventory.observations.iter().map(|o| (&o.id, o)).collect();
    let uncertain: BTreeSet<_> = inventory
        .uncertainties
        .iter()
        .filter_map(|u| u.observation.as_ref())
        .collect();
    let mut claimed = vec![false; evidence.triangles.len()];
    for obj in &identities.objects {
        if ambiguous.contains(&obj.id)
            || !matches!(
                obj.motion,
                MotionClass::StaticStructure | MotionClass::MovableStationary
            )
        {
            continue;
        }
        let views: Vec<_> = checked
            .masks
            .iter()
            .filter(|m| {
                m.object == obj.id
                    && m.consistency == MaskConsistency::Consistent
                    && m.quality == Some(MaskQuality::Confident)
                    && !uncertain.contains(&m.observation)
            })
            .filter_map(|m| {
                Some(MaskView {
                    camera: cameras.iter().find(|c| c.frame_index == m.frame_index)?,
                    observation: observations.get(&m.observation)?,
                    depth: obj
                        .locations
                        .iter()
                        .find(|l| l.observation == m.observation)?
                        .depth,
                })
            })
            .collect();
        if views.len() < 2 {
            continue;
        }
        let group = partition.objects.len();
        partition.objects.push((obj.id.clone(), obj.label.clone()));
        for (index, face) in evidence.triangles.iter().enumerate() {
            let indices = [face.a, face.b, face.c];
            // A derived seam-clip corner lies on an observed triangle and inherits (never
            // adds) its provenance, so it may carry an object label; localization above
            // still uses observed points only.
            if !indices
                .iter()
                .all(|i| observed(evidence.provenance_point(*i)))
            {
                continue;
            }
            let corners = indices.map(xyz);
            let center =
                std::array::from_fn(|axis| corners.iter().map(|p| p[axis]).sum::<f64>() / 3.);
            let mut eligible = 0;
            let mut supporting = 0;
            for view in &views {
                if view.can_judge(center) {
                    eligible += 1;
                    supporting += usize::from(view.supports(&corners));
                }
            }
            if enough_mask_views(supporting, eligible) {
                // Competing claims stay with the environment until evidence distinguishes them.
                partition.groups[index] = (!claimed[index]).then_some(group);
                claimed[index] = true;
            }
        }
    }
    Ok(partition)
}

fn enough_mask_views(supporting: usize, eligible: usize) -> bool {
    // Two coincident silhouettes can both include unrelated nearby geometry.
    supporting >= 2 && supporting * 2 > eligible
}

#[cfg(test)]
#[test]
fn a_face_needs_support_in_most_informative_views() {
    assert!(!enough_mask_views(1, 1));
    assert!(enough_mask_views(2, 2));
    assert!(enough_mask_views(2, 3));
    assert!(!enough_mask_views(2, 18));
}
