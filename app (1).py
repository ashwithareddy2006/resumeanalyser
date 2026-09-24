import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent

st.set_page_config(page_title="Student AI Agents", page_icon="🤖", layout="wide")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY is not configured.")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.2,
)

@tool
def resume_analysis_tool(resume: str, job_description: str) -> str:
    """Provide resume and job description."""
    return f"RESUME:\n{resume}\n\nJOB DESCRIPTION:\n{job_description}"

@tool
def study_planner_tool(subjects: str, days: int, hours_per_day: float) -> str:
    """Provide study-planning inputs."""
    return f"SUBJECTS:\n{subjects}\n\nDAYS: {days}\nHOURS: {hours_per_day}"

resume_agent = create_agent(
    model=llm,
    tools=[resume_analysis_tool],
    system_prompt=(
        "You are a Resume Analyzer Agent. "
        "Use the tool first. Return these sections: "
        "1. Overall Match Summary; 2. Matching Skills; "
        "3. Missing or Weak Skills; 4. Resume Improvement Suggestions; "
        "5. Five Likely Interview Questions. "
        "Do not invent experience."
    ),
)

study_agent = create_agent(
    model=llm,
    tools=[study_planner_tool],
    system_prompt=(
        "You are a Study Planner Agent. "
        "Use the tool first. Create a realistic daily plan, "
        "prioritization, revision schedule and practice strategy "
        "without exceeding the user's available hours."
    ),
)

st.title("🤖 Student AI Agents")
st.caption("Two LangChain + Gemini agents")

choice = st.selectbox(
    "Choose an agent",
    ["Resume Analyzer Agent", "Study Planner Agent"]
)

if choice == "Resume Analyzer Agent":
    resume = st.text_area("Paste your resume", height=260)
    job = st.text_area("Paste the job description", height=220)

    if st.button("Analyze Resume", type="primary"):
        if not resume.strip() or not job.strip():
            st.warning("Please provide both the resume and job description.")
        else:
            with st.spinner("Analyzing..."):
                result = resume_agent.invoke({
                    "messages": [{
                        "role": "user",
                        "content": (
                            "Analyze this resume for this job.\n\n"
                            f"Resume:\n{resume}\n\n"
                            f"Job Description:\n{job}"
                        )
                    }]
                })
            st.markdown(result["messages"][-1].content)

else:
    subjects = st.text_area(
        "Subjects / topics",
        placeholder="Example: DSA arrays, strings, sorting; DBMS SQL; OS processes"
    )
    days = st.number_input("Number of days", min_value=1, max_value=60, value=7)
    hours = st.number_input(
        "Hours available per day",
        min_value=0.5,
        max_value=16.0,
        value=4.0,
        step=0.5
    )

    if st.button("Create Study Plan", type="primary"):
        if not subjects.strip():
            st.warning("Please enter your subjects/topics.")
        else:
            with st.spinner("Creating your plan..."):
                result = study_agent.invoke({
                    "messages": [{
                        "role": "user",
                        "content": (
                            f"Create my study plan.\nTopics:\n{subjects}\n"
                            f"Days: {days}\nHours per day: {hours}"
                        )
                    }]
                })
            st.markdown(result["messages"][-1].content)
