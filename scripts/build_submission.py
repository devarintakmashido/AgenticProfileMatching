"""Build a clean submission package for the assignment."""

from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_ROOT / "submission_package" / "agentic_profile_matching"
ZIP_PATH = PROJECT_ROOT / "submission_package" / "agentic_profile_matching"

INCLUDE_PATHS = [
    "README.md",
    "SUBMISSION_CHECKLIST.md",
    "requirements.txt",
    "fs_tools.py",
    "llm_file_assistant.py",
    "matching_agent.py",
    "streamlit_app.py",
    "scripts/generate_sample_resumes.py",
    "scripts/check_submission.py",
    "scripts/build_submission.py",
    "docs",
    "tests",
    "sample_data/resumes",
    "sample_data/generated_resumes",
    "generated_reports",
    "generated_summaries",
]


def copy_path(relative_path: str) -> None:
    source = PROJECT_ROOT / relative_path
    destination = PACKAGE_DIR / relative_path
    if source.is_dir():
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    package_root = PROJECT_ROOT / "submission_package"
    if package_root.exists():
        shutil.rmtree(package_root)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    for relative_path in INCLUDE_PATHS:
        copy_path(relative_path)

    archive = shutil.make_archive(str(ZIP_PATH), "zip", package_root, "agentic_profile_matching")
    print(f"Created clean submission folder: {PACKAGE_DIR}")
    print(f"Created zip archive: {archive}")


if __name__ == "__main__":
    main()
