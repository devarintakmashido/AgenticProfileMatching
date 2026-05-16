"""Streamlit chat interface for the profile matching agent."""

from __future__ import annotations

import streamlit as st

from matching_agent import MatchingAgentSession, save_report


st.set_page_config(page_title="Agentic Profile Matching", layout="wide")

st.title("Agentic Profile Matching")

resume_directory = st.sidebar.text_input("Resume directory", "sample_data/generated_resumes")
if "agent_session" not in st.session_state or st.session_state.get("resume_directory") != resume_directory:
    st.session_state.agent_session = MatchingAgentSession(resume_directory)
    st.session_state.resume_directory = resume_directory
    st.session_state.messages = []

with st.sidebar:
    st.caption("Demo prompts")
    st.code(
        "Find candidates with Python, FastAPI, Docker and 5+ years experience\n"
        "Compare the top 3 matches side by side\n"
        "Now require AWS\n"
        "Generate interview questions for the top candidate",
        language="text",
    )
    if st.button("Save latest report"):
        st.success(save_report(st.session_state.agent_session.state))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask for candidates, comparisons, refinements, or interview questions")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = st.session_state.agent_session.ask(prompt)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
