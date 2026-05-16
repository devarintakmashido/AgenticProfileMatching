# Agentic Profile Matching

This project implements the Agentic Profile Matching assignment as a runnable Python application with:

- `fs_tools.py` for structured file system tools
- `llm_file_assistant.py` for natural-language queries with optional OpenAI tool calling
- `matching_agent.py` for LangGraph-based candidate matching workflow, with a local fallback runner
- `sample_data/resumes/` with 7 dummy resume files across `.txt`, `.docx`, and `.pdf`
- `sample_data/generated_resumes/` with 100 deterministic resumes for multi-round screening
- `generated_summaries/` as the output folder for assistant-created summaries
- `generated_reports/` as the output folder for candidate match reports
- `streamlit_app.py` for an optional web chat interface
- `docs/state_machine.md` with the required state machine diagram
- `docs/test_scenarios.md` with 5+ conversation flows
- `tests/` with a unittest suite for file tools and agent behavior
- `SUBMISSION_CHECKLIST.md` plus scripts for final submission checks and packaging

## Features

### Part A: Core file tools

- `read_file(filepath)` reads `.txt`, `.pdf`, and `.docx` files and returns extracted text with metadata
- `list_files(directory, extension=None)` lists files with size and modified date
- `write_file(filepath, content)` writes content and creates parent directories automatically
- `search_in_file(filepath, keyword)` performs case-insensitive search and returns contextual matches

### Part B: LLM integration

- OpenAI function calling support through `llm_file_assistant.py`
- Local fallback mode when `OPENAI_API_KEY` is not configured
- Natural language examples from the assignment brief are supported

### Agentic profile matching

- Tracks conversation history, job requirements, candidate shortlist, and reasoning
- Workflow: `START -> Parse JD -> Extract Requirements -> Search Resumes -> Initial Screen -> Deep Analysis -> Final Rank -> Generate Report -> Human Feedback Loop -> END`
- Uses local TF-IDF retrieval as a lightweight RAG search over resume text
- Includes `extract_requirements(jd)`, `compare_candidates(candidate_ids)`, and `generate_interview_questions(candidate_id)`
- Demonstrates multi-round screening: top 10 from 100 resumes, second-round analysis, final hire/maybe/no-hire recommendations
- Supports iterative refinement and re-ranking during a CLI chat session
- Produces strengths, gaps, improvement suggestions, and hire/maybe/no-hire recommendations

## Project structure

```text
.
├── fs_tools.py
├── llm_file_assistant.py
├── matching_agent.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── scripts/
├── docs/
├── generated_summaries/
├── generated_reports/
└── sample_data/
    ├── generated_resumes/
    └── resumes/
```

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optional: configure OpenAI for tool calling:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Usage

Run the Milestone 1 file assistant from the project root:

```bash
python3 llm_file_assistant.py "Read all resumes in the resumes folder"
python3 llm_file_assistant.py "Find resumes mentioning Python experience"
python3 llm_file_assistant.py "Create a summary file for resume_john_doe.txt"
```

If an OpenAI API key is available, the assistant will use tool calling automatically. Without it, the application falls back to a deterministic local interpreter for the assignment’s sample prompts.

Run the profile matching agent:

```bash
python3 scripts/generate_sample_resumes.py
python3 matching_agent.py "Find me candidates with React and 3+ years experience" --resumes sample_data/generated_resumes
python3 matching_agent.py "Find candidates with Python, FastAPI, Docker and 5+ years experience" --resumes sample_data/generated_resumes --save-report
python3 matching_agent.py --chat --resumes sample_data/generated_resumes
```

Inside chat mode, try:

```text
Find candidates with Python and Docker
Compare the top 3 matches side by side
Why did the top candidate rank higher than the second?
Now require AWS
Generate interview questions for the top candidate
exit
```

Run the optional Streamlit chat UI:

```bash
streamlit run streamlit_app.py
```

## Run tests

```bash
python3 scripts/check_submission.py
python3 -m unittest discover -s tests -v
```

Build a clean zip for upload:

```bash
python3 scripts/build_submission.py
```

## Assignment deliverables

- LangGraph-based agent implementation: `matching_agent.py`
- State machine diagram: `docs/state_machine.md`
- Chat interface: `python3 matching_agent.py --chat` or `streamlit run streamlit_app.py`
- Test scenarios: `docs/test_scenarios.md`
- Demo report output: `generated_reports/candidate_match_report.md`
- Clean submission helper: `scripts/build_submission.py`

## Direct tool examples

```python
from fs_tools import list_files, read_file, search_in_file, write_file

print(list_files("sample_data/resumes", ".txt"))
print(read_file("sample_data/resumes/resume_john_doe.txt"))
print(search_in_file("sample_data/resumes/resume_john_doe.txt", "Python"))
print(write_file("notes/output.txt", "Hello from fs_tools"))
```

