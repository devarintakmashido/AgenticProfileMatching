# Test Scenarios

These flows cover the 5+ conversation scenarios requested in the PDF brief.

## 1. Find Matching Candidates

```bash
python3 matching_agent.py "Find me candidates with React and 3+ years experience" --resumes sample_data/generated_resumes
```

Expected: Emily Clark appears as the top React match with strengths, gaps, and recommendation.

## 2. Python Backend Search

```bash
python3 matching_agent.py "Find candidates with Python, FastAPI, Docker and 5+ years experience" --resumes sample_data/generated_resumes
```

Expected: the report shows a top 10 initial screen from the generated 100-resume dataset.

## 3. Compare Top 3 Side By Side

```bash
printf 'Find candidates with Python and Docker\nCompare the top 3 matches side by side\nexit\n' | python3 matching_agent.py --chat --resumes sample_data/generated_resumes
```

Expected: The second response is a JSON side-by-side comparison of the top 3 current matches.

## 4. Ranking Explanation

```bash
printf 'Find candidates with Python and SQL\nWhy did the top candidate rank higher than the second?\nexit\n' | python3 matching_agent.py --chat --resumes sample_data/generated_resumes
```

Expected: The assistant explains the score, strengths, and gaps behind the ranking.

## 5. Iterative Refinement

```bash
printf 'Find candidates with Python experience\nNow require Docker and AWS\nexit\n' | python3 matching_agent.py --chat --resumes sample_data/generated_resumes
```

Expected: The agent updates requirements, re-ranks candidates, and explains the new ranking.

## 6. Interview Questions

```bash
printf 'Find candidates with Kubernetes and Python\nGenerate interview questions for the top candidate\nexit\n' | python3 matching_agent.py --chat --resumes sample_data/generated_resumes
```

Expected: The assistant generates screening questions based on matched skills and gaps.

## 7. Save Report

```bash
python3 matching_agent.py "Find candidates with React and TypeScript" --resumes sample_data/generated_resumes --save-report
```

Expected: A report is written to `generated_reports/candidate_match_report.md`.
