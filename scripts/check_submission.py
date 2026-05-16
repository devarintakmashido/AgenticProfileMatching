"""Check that the assignment deliverables are present before submission."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "requirements.txt",
    "fs_tools.py",
    "llm_file_assistant.py",
    "matching_agent.py",
    "streamlit_app.py",
    "scripts/generate_sample_resumes.py",
    "docs/state_machine.md",
    "docs/test_scenarios.md",
    "tests/test_fs_tools.py",
    "tests/test_matching_agent.py",
    "sample_data/resumes",
    "sample_data/generated_resumes",
    "generated_reports/candidate_match_report.md",
]


def main() -> None:
    missing = [path for path in REQUIRED_PATHS if not (PROJECT_ROOT / path).exists()]
    generated_resumes = list((PROJECT_ROOT / "sample_data/generated_resumes").glob("*.txt"))
    mixed_resume_files = list((PROJECT_ROOT / "sample_data/resumes").glob("*"))
    langgraph_available = importlib.util.find_spec("langgraph") is not None
    streamlit_available = importlib.util.find_spec("streamlit") is not None

    print("Submission readiness check")
    print(f"- Required paths present: {'yes' if not missing else 'no'}")
    if missing:
        for path in missing:
            print(f"  missing: {path}")
    print(f"- Generated resume count: {len(generated_resumes)}")
    print(f"- Mixed-format sample resume count: {len(mixed_resume_files)}")
    print(f"- LangGraph installed in active Python: {'yes' if langgraph_available else 'no'}")
    print(f"- Streamlit installed in active Python: {'yes' if streamlit_available else 'no'}")

    if missing or len(generated_resumes) < 100 or len(mixed_resume_files) < 5:
        raise SystemExit(1)

    print("Ready for submission.")


if __name__ == "__main__":
    main()
