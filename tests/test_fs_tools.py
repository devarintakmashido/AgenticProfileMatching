import tempfile
import unittest
from pathlib import Path

from fs_tools import list_files, read_file, search_in_file, write_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESUMES_DIR = PROJECT_ROOT / "sample_data" / "resumes"


class FsToolsTests(unittest.TestCase):
    def test_list_files_filters_by_extension(self) -> None:
        files = list_files(str(RESUMES_DIR), ".txt")
        self.assertTrue(files)
        self.assertTrue(all(item["extension"] == ".txt" for item in files))

    def test_read_file_reads_text_resume(self) -> None:
        result = read_file(str(RESUMES_DIR / "resume_john_doe.txt"))
        self.assertTrue(result["success"])
        self.assertIn("Senior Python Developer", result["content"])

    def test_search_in_file_is_case_insensitive(self) -> None:
        result = search_in_file(str(RESUMES_DIR / "resume_john_doe.txt"), "python")
        self.assertTrue(result["success"])
        self.assertGreater(result["match_count"], 0)

    def test_write_file_creates_nested_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_file = Path(tmp_dir) / "nested" / "output.txt"
            result = write_file(str(nested_file), "hello world")
            self.assertTrue(result["success"])
            self.assertEqual(nested_file.read_text(encoding="utf-8"), "hello world")


if __name__ == "__main__":
    unittest.main()
