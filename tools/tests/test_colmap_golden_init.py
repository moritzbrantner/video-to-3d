import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "colmap_golden_init.py"
spec = importlib.util.spec_from_file_location("colmap_golden_init", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ColmapGoldenInitTests(unittest.TestCase):
    def images(self, count=8):
        return [(index + 11, f"frame-{index:02}.ppm") for index in range(count)]

    def test_prioritizes_first_to_midpoint_for_revisit_baseline(self):
        pairs = module.candidate_pairs(self.images())
        self.assertEqual(pairs[0], (11, 15, "frame-00.ppm", "frame-04.ppm"))

    def test_candidates_are_bounded_distinct_and_deterministic(self):
        pairs = module.candidate_pairs(self.images())
        self.assertEqual(
            pairs,
            [
                (11, 15, "frame-00.ppm", "frame-04.ppm"),
                (11, 16, "frame-00.ppm", "frame-05.ppm"),
                (12, 16, "frame-01.ppm", "frame-05.ppm"),
                (13, 17, "frame-02.ppm", "frame-06.ppm"),
                (11, 18, "frame-00.ppm", "frame-07.ppm"),
            ],
        )

    def test_short_sequences_deduplicate_candidate_pairs(self):
        pairs = module.candidate_pairs(self.images(3))
        identities = [(left, right) for left, right, _, _ in pairs]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertTrue(all(left != right for left, right in identities))

    def test_requires_two_images(self):
        with self.assertRaisesRegex(ValueError, "at least two images"):
            module.candidate_pairs(self.images(1))

    def test_database_images_are_ordered_by_name_not_insertion(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "database.db"
            connection = sqlite3.connect(database)
            try:
                connection.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
                connection.executemany(
                    "INSERT INTO images VALUES (?, ?)",
                    [(9, "frame-02.ppm"), (4, "frame-00.ppm"), (7, "frame-01.ppm")],
                )
                connection.commit()
            finally:
                connection.close()

            self.assertEqual(
                module.load_images(database),
                [(4, "frame-00.ppm"), (7, "frame-01.ppm"), (9, "frame-02.ppm")],
            )


if __name__ == "__main__":
    unittest.main()
