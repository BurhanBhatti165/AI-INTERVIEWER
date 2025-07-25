from dotenv import load_dotenv
load_dotenv()

import os
import re
import time
import streamlit as st
from datetime import datetime
from typing import List
from pydantic import Field
import logging
from gtts import gTTS
import speech_recognition as sr
import tempfile
import io
import base64
import requests

# Suppress INFO logs for cleaner output
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("pikepdf").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)

# Set USER_AGENT before any imports that might use it
os.environ["USER_AGENT"] = "AI-Interview-App/1.0 (contact: support@xai-interview.com)"

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader

# --- CONFIG ---
MAX_QUESTIONS = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Cache the embedding model to prevent repeated loading
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # Explicitly set device to CPU
    )

embedding = get_embeddings()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

st.set_page_config(page_title="🤖 AI Interviewer", layout="centered")
st.title("🧠 AI-Powered Interview System")
st.subheader("Please read the instructions carefully before starting the interview")
st.markdown("""
Welcome to the AI-Powered Interview System, {name}! This app simulates an interview process using advanced AI technologies.
You can upload your resume, choose an interview type, and select a mode (Text or Voice).
In **Text Mode**, questions and answers are text-based. In **Voice Mode**, the AI speaks questions, and you can respond via speech or text.
The AI will evaluate your answers and provide feedback to help you improve.
**Important**: In Voice Mode, ensure your microphone is enabled for speech input and speakers for audio output.
""".format(name=st.session_state.get('name', 'Candidate')))

st.sidebar.title("Settings")

# --- AGENTS ---
AGENTS = {
    "HR": {
        "url": "https://www.geeksforgeeks.org/hr-interview-questions/",
        "prompt": "You are a professional HR interviewer. Ask relevant soft skill or personality questions based on the resume. We have provided you the resume for context. As an HR interviewer, focus on questions related to soft skills, personality traits, and general behavioral aspects of the candidate. Use the resume to tailor your questions. Do not ask repetitive questions or questions already answered in previous questions.Also dont include the name of the user in the question."
    },
    "Web Development": {
        "url": "https://www.simplilearn.com/web-development-interview-questions-article",
        "prompt": "You are a Web Development interviewer. Ask questions related to HTML, CSS, JavaScript, React, or other web-related technologies based on the resume. We have provided you the resume for context. Focus on web development technologies, frameworks, and best practices. Do not ask questions unrelated to web development, such as AI/ML or Data Science. Do not ask repetitive questions or questions already answered in previous questions.Also dont include the name of the user in the question."
    },
    "AI/ML": {
        "url": "https://www.turing.com/interview-questions/artificial-intelligence",
        "prompt": "You are an AI/ML interviewer. Ask questions related to machine learning algorithms, data processing, model evaluation, and AI concepts based on the resume. We have provided you the resume for context. Focus on AI/ML topics and avoid questions unrelated to AI/ML, such as Web Development or Data Science. Do not ask repetitive questions or questions already answered in previous questions.also dont include the name of the user in the question."
    },
    "Data Science": {
        "url": "https://www.geeksforgeeks.org/data-science/data-science-interview-questions-and-answers/",
        "prompt": "You are a Data Science interviewer. Ask questions related to data analysis, statistical methods, and data visualization based on the resume. We have provided you the resume for context. Focus on Data Science topics and avoid questions unrelated to Data Science, such as Web Development or AI/ML. Do not ask repetitive questions or questions already answered in previous questions.also dont include the name of the user in the question."
    },
    "Project Management": {
        "url": "https://www.simplilearn.com/project-management-interview-questions-and-answers-article",
        "prompt": "You are a Project Management interviewer. Ask questions related to project planning, execution, risk management, and team leadership based on the resume. We have provided you the resume for context. Focus on Project Management topics and avoid questions unrelated to Project Management, such as Web Development or AI/ML. Do not ask repetitive questions or questions already answered in previous questions.also dont include the name of the user in the question."
    }
}


# --- LOADERS ---
@st.cache_resource
def load_agent_retriever(url):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        vectordb = FAISS.from_documents(chunks, embedding)
        return vectordb.as_retriever()
    except requests.exceptions.ConnectionError as e:
        st.error(f"Failed to load web content from {url}: {e}. Using resume data only.")
        return None

def load_resume(file):
    with open("temp_resume.pdf", "wb") as f:
        f.write(file.read())
    loader = PyPDFLoader("temp_resume.pdf")
    return loader.load()

# --- HYBRID RETRIEVER ---
class HybridRetriever(BaseRetriever):
    resume_retriever: BaseRetriever = Field(...)
    agent_retriever: BaseRetriever = Field(default=None)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        resume_docs = self.resume_retriever.invoke(query)
        agent_docs = self.agent_retriever.invoke(query) if self.agent_retriever else []
        combined_docs = resume_docs + agent_docs if resume_docs or agent_docs else []
        return combined_docs

# --- CHAINS ---
def get_agent_chain(retriever, prompt):
    full_prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=prompt + "\n\nResume:\n{context}\nQuestion: {query}\nGenerate one interview question."
    )
    chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | full_prompt
        | llm
        | StrOutputParser()
    )
    return chain

def get_evaluator_chain(domain):
    eval_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template=f"""You are a senior {domain} interviewer evaluating a candidate's answer.

Question: {{question}}
Answer: {{answer}}

Provide:
Score: <1-10>
Feedback: <short feedback>
"""
    )
    return eval_prompt | llm | StrOutputParser()

def parse_eval_output(result):
    score_match = re.search(r"[Ss]core[:\-–]?\s*\*?(\d{1,2})", result)
    feedback_match = re.search(r"[Ff]eedback[:\-–]?\s*(.*)", result, re.DOTALL)
    score = int(score_match.group(1)) if score_match else 0
    feedback = feedback_match.group(1).strip() if feedback_match else "No feedback available."
    return score, feedback

def save_to_file(agent, name, results):
    filename = f"results_{agent.lower().replace(' ', '_')}.txt"
    with open(filename, "a") as f:
        f.write(f"=== Interview Summary ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ===\n")
        f.write(f"Candidate: {name} | Domain: {agent}\n\n")
        for i, entry in enumerate(results, 1):
            f.write(f"Q{i}: {entry['question']}\n")
            f.write(f"A{i}: {entry['answer']}\n")
            f.write(f"Score: {entry['score']}/10\nFeedback: {entry['feedback']}\n\n")
        f.write("=============================================\n\n")

# --- VOICE FUNCTIONS ---
def text_to_speech(text, autoplay=True, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{time.time()}.mp3')
        tts.save(temp_file.name)
        temp_file.close()
        with open(temp_file.name, 'rb') as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        try:
            os.unlink(temp_file.name)
        except PermissionError as e:
            logging.warning(f"Could not delete temp file {temp_file.name}: {e}")
        return f'<audio {"autoplay" if autoplay else ""} src="data:audio/mp3;base64,{audio_b64}"></audio>'
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return ""

def speech_to_text(max_retries=2):
    recognizer = sr.Recognizer()
    for attempt in range(max_retries):
        try:
            with sr.Microphone() as source:
                st.info(f"🎙️ Listening (Attempt {attempt + 1}/{max_retries})... Speak your answer now.")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)
                text = recognizer.recognize_google(audio)
                st.success("✅ Speech recognized!")
                return text
        except sr.WaitTimeoutError:
            st.warning("No speech detected within the time limit.")
        except sr.UnknownValueError:
            st.warning("Could not understand the audio. Please try again.")
        except sr.RequestError as e:
            st.error(f"Speech recognition error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    st.error("Failed to recognize speech after multiple attempts.")
    return None

# --- UI ---
name = st.sidebar.text_input("👤 Your Name", key="candidate_name")
interview_mode = st.sidebar.radio("🎮 Interview Mode", ["Text Mode", "Voice Mode"])
tts_lang = st.sidebar.selectbox("🌐 TTS Language", ["en", "en-uk", "en-au"], key="tts_lang")
agent_choice = st.sidebar.selectbox("🎯 Choose Interview Type", list(AGENTS.keys()))
resume_file = st.sidebar.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

# Reset session state button
if st.sidebar.button("Reset Interview"):
    st.session_state.clear()
    st.rerun()

# Initialize session state
if "interview_data" not in st.session_state:
    st.session_state.interview_data = {
        "questions": [],
        "answers": [],
        "results": [],
        "current_q": 1
    }
if "name" not in st.session_state:
    st.session_state.name = name if name else "Candidate"
if "mode" not in st.session_state:
    st.session_state.mode = interview_mode
if "transcribed_answer" not in st.session_state:
    st.session_state.transcribed_answer = None

# Update name and mode in session state when input changes
if name and name != st.session_state.name:
    st.session_state.name = name
    st.session_state.interview_data = {
        "questions": [],
        "answers": [],
        "results": [],
        "current_q": 1
    }
    st.session_state.current_question = None
    st.session_state.transcribed_answer = None
if interview_mode != st.session_state.mode:
    st.session_state.mode = interview_mode
    st.session_state.interview_data = {
        "questions": [],
        "answers": [],
        "results": [],
        "current_q": 1
    }
    st.session_state.current_question = None
    st.session_state.transcribed_answer = None

# Main logic
if resume_file and name and agent_choice:
    st.markdown(f"### Good morning, {name}! Your {agent_choice} interview in {interview_mode} is ready.")
    with st.spinner("🔄 Loading data and agents..."):
        resume_docs = load_resume(resume_file)
        resume_chunks = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(resume_docs)
        resume_vectordb = FAISS.from_documents(resume_chunks, embedding)
        resume_retriever = resume_vectordb.as_retriever()

        agent_config = AGENTS[agent_choice]
        agent_retriever = load_agent_retriever(agent_config["url"])

        hybrid_retriever = HybridRetriever(
            resume_retriever=resume_retriever,
            agent_retriever=agent_retriever
        )

        qa_chain = get_agent_chain(hybrid_retriever, agent_config["prompt"])

    st.success(f"{agent_choice} Interview Ready ✅")

    if st.session_state.interview_data["current_q"] <= MAX_QUESTIONS:
        if "current_question" not in st.session_state or not st.session_state.current_question:
            st.session_state.current_question = qa_chain.invoke("Start interview")
            if interview_mode == "Voice Mode":
                st.markdown(text_to_speech(st.session_state.current_question, autoplay=True, lang=tts_lang), unsafe_allow_html=True)

        st.markdown(f"### ❓ Question {st.session_state.interview_data['current_q']}")
        st.info(st.session_state.current_question)

        if interview_mode == "Voice Mode":
            if st.button("🔊 Play/Replay Question"):
                st.markdown(text_to_speech(st.session_state.current_question, autoplay=True, lang=tts_lang), unsafe_allow_html=True)

            response_mode = st.radio("📢 Response Mode", ["Text", "Speech"], key=f"response_mode_{st.session_state.interview_data['current_q']}")

            if response_mode == "Text":
                user_answer = st.text_area("✍️ Your Answer", key=f"answer_{st.session_state.interview_data['current_q']}")
                st.session_state.transcribed_answer = None  # Clear transcribed answer
            else:
                if st.button("🎙️ Record Answer"):
                    transcribed = speech_to_text()
                    if transcribed:
                        st.session_state.transcribed_answer = transcribed
                        st.text_area("✍️ Transcribed Answer", value=transcribed, disabled=True, key=f"transcribed_answer_{st.session_state.interview_data['current_q']}")
                    else:
                        st.stop()
                user_answer = st.session_state.transcribed_answer
        else:
            user_answer = st.text_area("✍️ Your Answer", key=f"answer_{st.session_state.interview_data['current_q']}")
            st.session_state.transcribed_answer = None  # Clear transcribed answer

        if st.button("✅ Submit Answer"):
            if not user_answer or not user_answer.strip():
                st.warning("Please enter or record your answer before submitting.")
                st.stop()

            evaluator = get_evaluator_chain(agent_choice)
            with st.spinner("🔍 Evaluating..."):
                eval_result = evaluator.invoke({
                    "question": st.session_state.current_question,
                    "answer": user_answer
                })
                score, feedback = parse_eval_output(eval_result)

            st.success(f"Score: {score}/10")
            st.info(f"💬 Feedback: {feedback}")

            st.session_state.interview_data["questions"].append(st.session_state.current_question)
            st.session_state.interview_data["answers"].append(user_answer)
            st.session_state.interview_data["results"].append({
                "question": st.session_state.current_question,
                "answer": user_answer,
                "score": score,
                "feedback": feedback
            })

            st.session_state.interview_data["current_q"] += 1
            st.session_state.transcribed_answer = None  # Clear for next question

            if st.session_state.interview_data["current_q"] <= MAX_QUESTIONS:
                st.session_state.current_question = qa_chain.invoke("Next question")
                if interview_mode == "Voice Mode":
                    st.markdown(text_to_speech(st.session_state.current_question, autoplay=True, lang=tts_lang), unsafe_allow_html=True)
                st.rerun()
            else:
                st.success("🎉 Interview Completed!")
                total_score = sum(item["score"] for item in st.session_state.interview_data["results"])
                st.markdown(f"### 🧾 Summary (Total Score: {total_score}/{MAX_QUESTIONS*10})")
                for i, r in enumerate(st.session_state.interview_data["results"], 1):
                    st.markdown(f"""
**Q{i}**: {r['question']}  
**A{i}**: {r['answer']}  
**Score**: {r['score']}/10  
**Feedback**: {r['feedback']}  
---
""")
                save_to_file(agent_choice, name, st.session_state.interview_data["results"])
                st.balloons()
else:
    if not name:
        st.warning("Please enter your name.")
    if not resume_file:
        st.warning("Please upload your resume.")
    if not agent_choice:
        st.warning("Please select an interview type.")
