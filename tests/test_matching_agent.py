import unittest

from matching_agent import (
    MatchingAgentSession,
    compare_candidates,
    extract_requirements,
    generate_interview_questions,
    rank_candidates,
    run_workflow,
)


class MatchingAgentTests(unittest.TestCase):
    def test_extract_requirements_splits_must_and_nice_to_have(self) -> None:
        requirements = extract_requirements("Need Python and React. Nice to have AWS. 3+ years experience.")
        self.assertIn("python", requirements["must_have"])
        self.assertIn("react", requirements["must_have"])
        self.assertIn("aws", requirements["nice_to_have"])
        self.assertEqual(requirements["min_years"], 3)

    def test_workflow_ranks_react_candidate(self) -> None:
        state = run_workflow("Find candidates with React and 3+ years experience")
        self.assertTrue(state["shortlist"])
        self.assertEqual(state["shortlist"][0]["name"], "Emily Clark")

    def test_compare_candidates_uses_ranked_shortlist(self) -> None:
        state = run_workflow("Find candidates with Python and Docker")
        ids = [candidate["candidate_id"] for candidate in state["shortlist"][:2]]
        comparison = compare_candidates(ids, state["shortlist"])
        self.assertEqual(len(comparison["comparison"]), len(ids))

    def test_generate_interview_questions_for_top_candidate(self) -> None:
        state = run_workflow("Find candidates with Python and SQL")
        candidate_id = state["shortlist"][0]["candidate_id"]
        questions = generate_interview_questions(candidate_id, state["shortlist"])
        self.assertTrue(questions["questions"])

    def test_session_refinement_reranks(self) -> None:
        session = MatchingAgentSession()
        first_answer = session.ask("Find candidates with Python experience")
        refined_answer = session.ask("Now require Docker")
        self.assertIn("Candidate Match Report", first_answer)
        self.assertIn("docker", refined_answer.lower())

    def test_generated_dataset_supports_top_10_screening(self) -> None:
        state = run_workflow(
            "Find candidates with Python, FastAPI, Docker and 5+ years experience",
            "sample_data/generated_resumes",
        )
        self.assertEqual(len(state["initial_screen"]), 10)
        self.assertEqual(len(state["second_round"]), 10)
        self.assertEqual(len(state["final_recommendations"]), 5)
        self.assertIn("Multi-Round Screening", state["final_answer"])


if __name__ == "__main__":
    unittest.main()
