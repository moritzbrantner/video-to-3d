import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


source = Path(__file__).resolve().parents[1] / "colmap-sparse" / "tracks.py"
spec = importlib.util.spec_from_file_location("sparse_tracks", source)
tracks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracks)


class SparseGaugeTests(unittest.TestCase):
    def test_ambiguous_keypoints_do_not_qualify_the_widest_pair(self):
        images = [
            SimpleNamespace(image_id=index, projection_center=lambda x=x: (x, 0, 0))
            for index, x in enumerate([0, 1, 5])
        ]
        observations = [[0, 1]] * 12 + [[0, 2]] * 10 + [[0, 2, 2]] * 3
        points = {
            index: SimpleNamespace(track=SimpleNamespace(elements=[SimpleNamespace(image_id=i) for i in ids]))
            for index, ids in enumerate(observations)
        }
        left, right, exported = tracks.choose_exportable_seed(SimpleNamespace(points3D=points), images)
        self.assertEqual((left.image_id, right.image_id), (0, 1))
        self.assertEqual(exported, list(range(22)))

    def test_global_recovery_tries_a_supported_gauge_when_the_longest_pair_only_sees_far_points(self):
        images = [SimpleNamespace(image_id=i, projection_center=lambda x=x: (x, 0, 0))
                  for i, x in enumerate((0, 2, 10))]
        points = {}
        for i in range(24):
            left, right, depth = (0, 2, 2000) if i < 12 else (0, 1, 10)
            points[i] = SimpleNamespace(
                xyz=(i * .01, 0, depth),
                track=SimpleNamespace(elements=[images[left], images[right]]),
            )
        model = SimpleNamespace(points3D=points)
        first, second, _ = tracks.choose_exportable_seed(model, images, check_global_parallax=True)
        self.assertEqual((first.image_id, second.image_id), (0, 1))
        with self.assertRaisesRegex(ValueError, "no camera pair"):
            tracks.choose_exportable_seed(SimpleNamespace(points3D=dict(list(points.items())[:12])), images,
                                          check_global_parallax=True)
