"""Generate deterministic dummy resumes for the multi-round screening demo."""

from __future__ import annotations

from pathlib import Path


OUTPUT_DIR = Path("sample_data/generated_resumes")

ROLES = [
    ("Backend Engineer", ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"]),
    ("Frontend Engineer", ["React", "TypeScript", "JavaScript", "Accessibility", "Jest"]),
    ("DevOps Engineer", ["Kubernetes", "Terraform", "AWS", "Docker", "Linux"]),
    ("Data Analyst", ["SQL", "Python", "Tableau", "Power BI", "ETL"]),
    ("ML Engineer", ["Python", "PyTorch", "scikit-learn", "Airflow", "Docker"]),
]

FIRST_NAMES = [
    "Aarav",
    "Isha",
    "Kabir",
    "Meera",
    "Rohan",
    "Tara",
    "Nikhil",
    "Anaya",
    "Vikram",
    "Sara",
]

LAST_NAMES = [
    "Sharma",
    "Rao",
    "Mehta",
    "Kapoor",
    "Patel",
    "Nair",
    "Iyer",
    "Sen",
    "Das",
    "Malhotra",
]


def build_resume(index: int) -> str:
    role, skills = ROLES[index % len(ROLES)]
    first_name = FIRST_NAMES[index % len(FIRST_NAMES)]
    last_name = LAST_NAMES[(index * 3) % len(LAST_NAMES)]
    display_name = f"{first_name} {last_name} C{index:03d}"
    years = 2 + (index % 9)
    primary_skills = ", ".join(skills)
    secondary_skills = ", ".join(ROLES[(index + 1) % len(ROLES)][1][:2])
    return f"""{display_name}
{role}

Email: candidate{index:03d}@example.com
Phone: +91-90000-{index:05d}
Location: Bengaluru, India

Summary
{role} with {years}+ years experience building production systems and collaborating with product teams.

Skills
{primary_skills}, {secondary_skills}

Experience
Senior {role}, DemoCorp, 2021-Present
- Delivered production projects using {skills[0]}, {skills[1]}, and {skills[2]}.
- Partnered with cross-functional teams on reliability, maintainability, and delivery quality.

Engineer, BuildWorks, 2018-2021
- Improved platform workflows and automated recurring engineering tasks.

Education
B.Tech Computer Science
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for index in range(1, 101):
        filepath = OUTPUT_DIR / f"generated_resume_{index:03d}.txt"
        filepath.write_text(build_resume(index), encoding="utf-8")
    print(f"Generated 100 resumes in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
