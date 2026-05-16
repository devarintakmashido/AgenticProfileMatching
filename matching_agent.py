"""Agentic profile matching workflow using LangGraph when available.

The module also has a deterministic local runner so the demo works without an
LLM API key or external services.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from fs_tools import list_files, read_file, write_file

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover
    END = "__end__"
    StateGraph = None


DEFAULT_RESUME_DIR = Path("sample_data/resumes")
REPORT_DIR = Path("generated_reports")

KNOWN_SKILLS = {
    "python",
    "fastapi",
    "django",
    "postgresql",
    "redis",
    "docker",
    "aws",
    "react",
    "next.js",
    "typescript",
    "javascript",
    "sql",
    "tableau",
    "power bi",
    "kubernetes",
    "terraform",
    "linux",
    "github actions",
    "bash",
    "pytorch",
    "scikit-learn",
    "airflow",
    "accessibility",
    "jest",
}

STOPWORDS = {
    "and",
    "are",
    "candidate",
    "candidates",
    "compare",
    "experience",
    "find",
    "for",
    "give",
    "have",
    "hire",
    "match",
    "matches",
    "me",
    "need",
    "of",
    "or",
    "rank",
    "resume",
    "resumes",
    "the",
    "to",
    "top",
    "with",
    "years",
}


class Requirements(TypedDict):
    original_jd: str
    must_have: List[str]
    nice_to_have: List[str]
    min_years: int
    keywords: List[str]


class Candidate(TypedDict, total=False):
    candidate_id: str
    name: str
    file: str
    content: str
    years: int
    matched_must_have: List[str]
    matched_nice_to_have: List[str]
    missing_must_have: List[str]
    score: float
    retrieval_score: float
    strengths: List[str]
    gaps: List[str]
    improvement_suggestions: List[str]
    recommendation: str
    reasoning: str
    interview_questions: List[str]


class AgentState(TypedDict, total=False):
    conversation_history: List[Dict[str, str]]
    job_description: str
    requirements: Requirements
    resume_directory: str
    all_candidates: List[Candidate]
    initial_screen: List[Candidate]
    second_round: List[Candidate]
    final_recommendations: List[Candidate]
    shortlist: List[Candidate]
    report: str
    feedback: str
    final_answer: str


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokens(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9.+#-]*", text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def _candidate_name(content: str, fallback: str) -> str:
    for line in content.splitlines():
        clean = line.strip()
        if clean:
            return clean
    return fallback.replace("_", " ").replace("-", " ").title()


def _extract_years(text: str) -> int:
    year_matches = [int(match) for match in re.findall(r"(\d+)\+?\s+years?", text.lower())]
    if year_matches:
        return max(year_matches)

    ranges = re.findall(r"\b(20\d{2})\s*[-–]\s*(20\d{2}|present)\b", text.lower())
    total = 0
    for start, end in ranges:
        end_year = 2026 if end == "present" else int(end)
        total += max(0, end_year - int(start))
    return total


def _skills_in_text(text: str) -> List[str]:
    normalized = _normalize(text)
    return sorted(skill for skill in KNOWN_SKILLS if skill in normalized)


def _cosine_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
    common = set(left).intersection(right)
    numerator = sum(left[key] * right[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _tfidf_vectors(query: str, documents: List[str]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    tokenized_documents = [_tokens(document) + _skills_in_text(document) for document in documents]
    query_tokens = _tokens(query) + _skills_in_text(query)
    document_count = max(1, len(tokenized_documents))
    document_frequency: Counter[str] = Counter()

    for tokens in tokenized_documents:
        document_frequency.update(set(tokens))

    def vectorize(tokens: List[str]) -> Dict[str, float]:
        counts = Counter(tokens)
        vector: Dict[str, float] = {}
        for token, count in counts.items():
            idf = math.log((document_count + 1) / (document_frequency.get(token, 0) + 1)) + 1
            vector[token] = count * idf
        return vector

    return vectorize(query_tokens), [vectorize(tokens) for tokens in tokenized_documents]


def extract_requirements(jd: str) -> Requirements:
    """Parse a job description into must-have, nice-to-have, years, and keywords."""
    normalized = _normalize(jd)
    skills = _skills_in_text(jd)
    min_years = 0
    years_match = re.search(r"(\d+)\+?\s+years?", normalized)
    if years_match:
        min_years = int(years_match.group(1))

    nice_to_have: List[str] = []
    for phrase in ("nice to have", "preferred", "bonus", "plus"):
        if phrase in normalized:
            trailing_text = normalized.split(phrase, maxsplit=1)[-1]
            nice_to_have.extend(_skills_in_text(trailing_text))

    nice_to_have = sorted(set(nice_to_have))
    must_have = sorted(skill for skill in skills if skill not in nice_to_have)
    keywords = [word for word, _ in Counter(_tokens(jd)).most_common(12)]

    return {
        "original_jd": jd,
        "must_have": must_have,
        "nice_to_have": nice_to_have,
        "min_years": min_years,
        "keywords": keywords,
    }


def rag_search_resumes(jd: str, directory: str = str(DEFAULT_RESUME_DIR), limit: int = 100) -> List[Candidate]:
    """Local TF-IDF RAG-style retrieval over resume text."""
    requirements = extract_requirements(jd)
    query_terms = set(requirements["must_have"] + requirements["nice_to_have"] + requirements["keywords"])
    loaded_candidates: List[Candidate] = []
    seen_candidate_ids = set()

    for file_info in list_files(directory):
        if file_info["extension"] not in {".txt", ".pdf", ".docx"}:
            continue
        candidate_id = Path(file_info["name"]).stem
        if candidate_id in seen_candidate_ids:
            continue

        result = read_file(file_info["path"])
        if not result.get("success"):
            continue

        content = result["content"]
        seen_candidate_ids.add(candidate_id)
        loaded_candidates.append(
            {
                "candidate_id": candidate_id,
                "name": _candidate_name(content, candidate_id),
                "file": file_info["path"],
                "content": content,
                "years": _extract_years(content),
            }
        )

    documents = [candidate["content"] for candidate in loaded_candidates]
    query_vector, document_vectors = _tfidf_vectors(jd, documents)
    candidates: List[Candidate] = []
    for candidate, document_vector in zip(loaded_candidates, document_vectors):
        content_terms = set(_tokens(candidate["content"]) + _skills_in_text(candidate["content"]))
        overlap = len(query_terms.intersection(content_terms))
        retrieval_score = round(_cosine_similarity(query_vector, document_vector) * 100, 2)
        if retrieval_score == 0 and overlap == 0 and query_terms:
            continue
        candidates.append({**candidate, "retrieval_score": retrieval_score, "score": retrieval_score})

    return sorted(
        candidates,
        key=lambda item: (item.get("retrieval_score", 0), item.get("years", 0)),
        reverse=True,
    )[:limit]


def rank_candidates(candidates: List[Candidate], requirements: Requirements) -> List[Candidate]:
    ranked: List[Candidate] = []
    for candidate in candidates:
        normalized = _normalize(candidate["content"])
        matched_must = [skill for skill in requirements["must_have"] if skill in normalized]
        matched_nice = [skill for skill in requirements["nice_to_have"] if skill in normalized]
        missing_must = [skill for skill in requirements["must_have"] if skill not in normalized]

        years = candidate.get("years", 0)
        must_total = max(1, len(requirements["must_have"]))
        nice_total = max(1, len(requirements["nice_to_have"]))
        must_score = (len(matched_must) / must_total) * 60
        nice_score = (len(matched_nice) / nice_total) * 15 if requirements["nice_to_have"] else 0
        if requirements["min_years"]:
            years_score = min(25, (years / requirements["min_years"]) * 25)
        else:
            years_score = min(25, years * 4)
        score = must_score + nice_score + years_score - len(missing_must) * 10
        score = max(0, min(100, score))

        strengths = []
        if matched_must:
            strengths.append("Matches required skills: " + ", ".join(matched_must))
        if matched_nice:
            strengths.append("Also has preferred skills: " + ", ".join(matched_nice))
        if years >= requirements["min_years"] and requirements["min_years"]:
            strengths.append(f"Meets the {requirements['min_years']}+ years requirement")

        gaps = []
        if missing_must:
            gaps.append("Missing required skills: " + ", ".join(missing_must))
        if requirements["min_years"] and years < requirements["min_years"]:
            gaps.append(f"Shows about {years} years against {requirements['min_years']} requested")

        improvement_suggestions = []
        for skill in missing_must:
            improvement_suggestions.append(f"Validate or upskill on {skill} before the next round.")
        if requirements["min_years"] and years < requirements["min_years"]:
            improvement_suggestions.append("Ask for project depth to confirm whether experience is underrepresented.")
        if not improvement_suggestions and score < 85:
            improvement_suggestions.append("Use the screening call to verify depth beyond keyword matches.")

        recommendation = "hire" if score >= 70 else "maybe" if score >= 45 else "no-hire"
        ranked.append(
            {
                **candidate,
                "matched_must_have": matched_must,
                "matched_nice_to_have": matched_nice,
                "missing_must_have": missing_must,
                "score": score,
                "strengths": strengths or ["Relevant background found in resume"],
                "gaps": gaps or ["No major gaps found from available text"],
                "improvement_suggestions": improvement_suggestions or ["Proceed to role-specific technical screening."],
                "recommendation": recommendation,
                "reasoning": (
                    f"Score {score}/100 from required skills, preferred skills, years of experience, "
                    f"and retrieval score {candidate.get('retrieval_score', 0)}/100."
                ),
            }
        )

    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def compare_candidates(candidate_ids: List[str], candidates: Optional[List[Candidate]] = None) -> Dict[str, Any]:
    """Create a side-by-side comparison for selected candidates."""
    candidates = candidates or []
    selected = [candidate for candidate in candidates if candidate["candidate_id"] in candidate_ids]
    return {
        "candidate_ids": candidate_ids,
        "comparison": [
            {
                "name": candidate["name"],
                "score": candidate.get("score", 0),
                "strengths": candidate.get("strengths", []),
                "gaps": candidate.get("gaps", []),
                "improvement_suggestions": candidate.get("improvement_suggestions", []),
                "recommendation": candidate.get("recommendation", "unknown"),
            }
            for candidate in selected
        ],
    }


def generate_interview_questions(candidate_id: str, candidates: Optional[List[Candidate]] = None) -> Dict[str, Any]:
    """Generate screening questions for one candidate."""
    candidates = candidates or []
    candidate = next((item for item in candidates if item["candidate_id"] == candidate_id), None)
    if candidate is None:
        return {"candidate_id": candidate_id, "error": "Candidate not found in current shortlist."}

    questions = [
        f"Walk me through a project where you used {skill} in production."
        for skill in candidate.get("matched_must_have", [])[:3]
    ]
    actionable_gaps = [
        gap
        for gap in candidate.get("gaps", [])
        if not gap.lower().startswith("no major gaps")
    ]
    for gap in actionable_gaps[:2]:
        questions.append(f"How would you address this potential gap: {gap}?")
    if not questions:
        questions.append("Which recent project best represents your current engineering judgment?")

    return {"candidate_id": candidate_id, "candidate_name": candidate["name"], "questions": questions[:5]}


def parse_jd(state: AgentState) -> AgentState:
    history = state.get("conversation_history", [])
    if not state.get("job_description") and history:
        state["job_description"] = history[-1]["content"]
    return state


def extract_requirements_node(state: AgentState) -> AgentState:
    state["requirements"] = extract_requirements(state.get("job_description", ""))
    return state


def search_resumes_node(state: AgentState) -> AgentState:
    state["all_candidates"] = rag_search_resumes(
        state.get("job_description", ""),
        state.get("resume_directory", str(DEFAULT_RESUME_DIR)),
        limit=100,
    )
    return state


def rank_candidates_node(state: AgentState) -> AgentState:
    requirements = state["requirements"]
    ranked = rank_candidates(state.get("all_candidates", []), requirements)
    state["initial_screen"] = ranked[:10]
    state["second_round"] = deep_analyze_candidates(state["initial_screen"], requirements)
    state["final_recommendations"] = state["second_round"][:5]
    state["shortlist"] = state["second_round"]
    return state


def generate_report_node(state: AgentState) -> AgentState:
    state["report"] = generate_match_report(state.get("shortlist", []), state["requirements"])
    return state


def human_feedback_node(state: AgentState) -> AgentState:
    feedback = state.get("feedback", "").strip()
    if feedback:
        updated_jd = state.get("job_description", "") + "\nAdditional feedback: " + feedback
        state["job_description"] = updated_jd
        state["requirements"] = extract_requirements(updated_jd)
        ranked = rank_candidates(state.get("all_candidates", []), state["requirements"])
        state["initial_screen"] = ranked[:10]
        state["second_round"] = deep_analyze_candidates(state["initial_screen"], state["requirements"])
        state["final_recommendations"] = state["second_round"][:5]
        state["shortlist"] = state["second_round"]
        state["report"] = generate_match_report(state["shortlist"], state["requirements"])
    state["final_answer"] = state.get("report", "")
    return state


def build_graph() -> Any:
    """Build the LangGraph state machine required by the assignment."""
    if StateGraph is None:
        return None

    graph = StateGraph(AgentState)
    graph.add_node("parse_jd", parse_jd)
    graph.add_node("extract_requirements", extract_requirements_node)
    graph.add_node("search_resumes", search_resumes_node)
    graph.add_node("rank_candidates", rank_candidates_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("human_feedback_loop", human_feedback_node)
    graph.set_entry_point("parse_jd")
    graph.add_edge("parse_jd", "extract_requirements")
    graph.add_edge("extract_requirements", "search_resumes")
    graph.add_edge("search_resumes", "rank_candidates")
    graph.add_edge("rank_candidates", "generate_report")
    graph.add_edge("generate_report", "human_feedback_loop")
    graph.add_edge("human_feedback_loop", END)
    return graph.compile()


def run_workflow(job_description: str, resume_directory: str = str(DEFAULT_RESUME_DIR), feedback: str = "") -> AgentState:
    initial_state: AgentState = {
        "conversation_history": [{"role": "user", "content": job_description}],
        "job_description": job_description,
        "resume_directory": resume_directory,
        "feedback": feedback,
    }

    graph = build_graph()
    if graph is not None:
        return graph.invoke(initial_state)

    state = parse_jd(initial_state)
    for step in (
        extract_requirements_node,
        search_resumes_node,
        rank_candidates_node,
        generate_report_node,
        human_feedback_node,
    ):
        state = step(state)
    return state


def generate_match_report(candidates: List[Candidate], requirements: Requirements) -> str:
    lines = [
        "# Candidate Match Report",
        "",
        "## Requirements",
        f"- Must-have: {', '.join(requirements['must_have']) or 'Not explicitly detected'}",
        f"- Nice-to-have: {', '.join(requirements['nice_to_have']) or 'None detected'}",
        f"- Minimum years: {requirements['min_years'] or 'Not specified'}",
        "",
        "## Multi-Round Screening",
        f"- Initial screen: top {min(10, len(candidates))} candidates from retrieved resumes",
        f"- Second round: deep analysis of top {min(10, len(candidates))}",
        "- Final round: hire/maybe/no-hire recommendations with strengths, gaps, and suggestions",
        "",
        "## Ranked Shortlist",
    ]

    if not candidates:
        lines.append("No matching candidates found from the available resumes.")
        return "\n".join(lines)

    for index, candidate in enumerate(candidates, start=1):
        lines.extend(
            [
                f"{index}. {candidate['name']} ({candidate['candidate_id']}) - {candidate['score']}/100 - {candidate['recommendation']}",
                f"   Strengths: {'; '.join(candidate['strengths'])}",
                f"   Gaps: {'; '.join(candidate['gaps'])}",
                f"   Suggestions: {'; '.join(candidate['improvement_suggestions'])}",
                f"   Reasoning: {candidate['reasoning']}",
            ]
        )
    return "\n".join(lines)


def deep_analyze_candidates(candidates: List[Candidate], requirements: Requirements) -> List[Candidate]:
    """Second-round analysis that rewards breadth and clear role fit."""
    analyzed: List[Candidate] = []
    for candidate in candidates:
        breadth = len(_skills_in_text(candidate["content"]))
        nice_fit = len(candidate.get("matched_nice_to_have", []))
        adjusted_score = min(100, round(candidate.get("score", 0) + min(8, breadth * 0.5) + nice_fit * 2, 2))
        analyzed.append(
            {
                **candidate,
                "score": adjusted_score,
                "reasoning": (
                    f"Second-round score {adjusted_score}/100 after deeper skill breadth review. "
                    f"Initial reasoning: {candidate['reasoning']}"
                ),
            }
        )
    return sorted(analyzed, key=lambda item: item["score"], reverse=True)


class MatchingAgentSession:
    def __init__(self, resume_directory: str = str(DEFAULT_RESUME_DIR)) -> None:
        self.resume_directory = resume_directory
        self.state: AgentState = {"conversation_history": [], "resume_directory": resume_directory}

    def ask(self, query: str) -> str:
        self.state.setdefault("conversation_history", []).append({"role": "user", "content": query})
        lower = query.lower()

        if "compare" in lower:
            return self._compare(query)
        if lower.startswith("why") or "rank higher" in lower:
            return self._explain_ranking(query)
        if "interview question" in lower or "screening question" in lower:
            return self._questions(query)
        if any(word in lower for word in ("adjust", "refine", "now require", "also require", "prefer")):
            return self._refine(query)

        self.state = run_workflow(query, self.resume_directory)
        return self.state["final_answer"]

    def _compare(self, query: str) -> str:
        shortlist = self.state.get("shortlist", [])
        if not shortlist:
            self.state = run_workflow(query, self.resume_directory)
            shortlist = self.state.get("shortlist", [])

        count_match = re.search(r"top\s+(\d+)", query.lower())
        count = int(count_match.group(1)) if count_match else 3
        ids = [candidate["candidate_id"] for candidate in shortlist[:count]]
        return json.dumps(compare_candidates(ids, shortlist), indent=2)

    def _explain_ranking(self, query: str) -> str:
        shortlist = self.state.get("shortlist", [])
        if len(shortlist) < 2:
            return "Run a candidate search first, then ask why one candidate ranked higher."

        names = _tokens(query)
        mentioned = [
            candidate
            for candidate in shortlist
            if any(part in _normalize(candidate["name"]) for part in names)
        ]
        first = mentioned[0] if mentioned else shortlist[0]
        second = mentioned[1] if len(mentioned) > 1 else shortlist[1]
        return (
            f"{first['name']} ranked higher than {second['name']} because {first['reasoning']} "
            f"Top strengths: {'; '.join(first['strengths'])}. "
            f"{second['name']} gaps: {'; '.join(second['gaps'])}. "
            f"Suggested follow-up: {'; '.join(first.get('improvement_suggestions', []))}"
        )

    def _questions(self, query: str) -> str:
        shortlist = self.state.get("shortlist", [])
        if not shortlist:
            return "Run a candidate search first, then ask for interview questions."

        tokens = _tokens(query)
        candidate = next(
            (
                item
                for item in shortlist
                if item["candidate_id"] in query or any(token in _normalize(item["name"]) for token in tokens)
            ),
            shortlist[0],
        )
        return json.dumps(generate_interview_questions(candidate["candidate_id"], shortlist), indent=2)

    def _refine(self, query: str) -> str:
        if not self.state.get("job_description"):
            self.state = run_workflow(query, self.resume_directory)
        else:
            self.state["feedback"] = query
            self.state = human_feedback_node(self.state)
        return self.state["final_answer"]


def save_report(state: AgentState) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "candidate_match_report.md"
    write_file(str(output), state.get("final_answer") or state.get("report", ""))
    return str(output.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic profile matching assistant.")
    parser.add_argument("query", nargs="?", help="Natural-language recruiting query or job description.")
    parser.add_argument("--resumes", default=str(DEFAULT_RESUME_DIR), help="Directory containing resume files.")
    parser.add_argument("--feedback", default="", help="Optional refinement feedback to apply after first ranking.")
    parser.add_argument("--chat", action="store_true", help="Start an interactive CLI chat.")
    parser.add_argument("--save-report", action="store_true", help="Write the latest match report to generated_reports/.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    session = MatchingAgentSession(args.resumes)

    if args.chat:
        print("Agentic Profile Matching CLI. Type 'exit' to quit.")
        while True:
            query = input("you> ").strip()
            if query.lower() in {"exit", "quit"}:
                break
            print(session.ask(query))
        return

    if not args.query:
        build_parser().print_help()
        return

    answer = session.ask(args.query)
    if args.feedback:
        answer = session.ask(args.feedback)
    print(answer)

    if args.save_report:
        print(f"\nReport saved to: {save_report(session.state)}")


if __name__ == "__main__":
    main()
