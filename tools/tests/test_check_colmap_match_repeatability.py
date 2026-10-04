from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_colmap_match_repeatability import compare


class MatchRepeatabilityTests(unittest.TestCase):
    def test_rejects_changed_matches_and_verified_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.db"
            right = Path(directory) / "right.db"
            for path in (left, right):
                with sqlite3.connect(path) as connection:
                    for table in ("matches", "two_view_geometries"):
                        connection.execute(f"CREATE TABLE {table} (pair_id INTEGER, data BLOB)")
                        connection.execute(f"INSERT INTO {table} VALUES (1, ?)", (b"same",))
            self.assertEqual(compare(left, right), [])
            with sqlite3.connect(right) as connection:
                connection.execute("UPDATE matches SET data = ?", (b"different",))
            self.assertEqual(len(compare(left, right)), 1)
            self.assertIn("matches differ", compare(left, right)[0])
            with sqlite3.connect(right) as connection:
                connection.execute("UPDATE two_view_geometries SET data = ?", (b"different",))
            self.assertEqual(len(compare(left, right)), 2)

    def test_missing_database_fails_without_creating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.db"
            with self.assertRaises(sqlite3.OperationalError):
                compare(missing, missing)
            self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
