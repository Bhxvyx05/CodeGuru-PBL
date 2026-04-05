import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from utils import extract_text_from_pdf, chunk_text

# ================== CONFIG ==================
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="CodeGuru", layout="wide")

# ================== CACHING (CRITICAL FOR SPEED) ==================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_data(show_spinner=False)
def process_resume(resume_text):
    chunks = chunk_text(resume_text)
    embeddings = load_embeddings()
    return FAISS.from_texts(chunks, embedding=embeddings)

@st.cache_resource
def load_llm():
    models = genai.list_models()

    for m in models:
        if "generateContent" in m.supported_generation_methods:
            print(f"Using model: {m.name}")
            return genai.GenerativeModel(m.name)

    raise RuntimeError("No Gemini text-generation model available for this API key")


# ================== UI ==================
st.title("🧠 CodeGuru – AI Interview Preparation")
st.write("Upload your resume and prepare for HR & Technical interviews")

language = st.selectbox(
    "Choose Language",
    ["English", "Hindi", "Hinglish"]
)

mode = st.radio(
    "Interview Mode",
    ["HR Interview", "Technical Interview", "Mock Interview"]
)

uploaded_resume = st.file_uploader(
    "Upload your Resume (PDF)",
    type=["pdf"]
)

# ================== RESUME ANALYSIS ==================
if uploaded_resume:
    if st.button("🔍 Analyze Resume"):
        with st.spinner("Analyzing resume (one-time)..."):
            resume_text = extract_text_from_pdf(uploaded_resume)
            vector_db = process_resume(resume_text)
            st.session_state["vector_db"] = vector_db

        st.success("✅ Resume analyzed successfully!")

# ================== Q&A ==================
if "vector_db" in st.session_state:
    user_query = st.text_input("Ask your interview question")

    if user_query:
        docs = st.session_state["vector_db"].similarity_search(user_query, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
You are CodeGuru, an expert interview coach.

Interview Mode: {mode}
Language: {language}

Resume Context:
{context}

Instructions:
- Behave like a real interviewer
- Ask follow-up questions when relevant
- HR answers must follow STAR method
- Technical answers must be step-by-step
- Keep answers concise and structured

User Question:
{user_query}
"""

        llm = load_llm()
        response = llm.generate_content(prompt)

        st.markdown("### 🤖 CodeGuru Response")
        st.write(response.text)

        # ================== FEEDBACK ==================
        feedback_prompt = f"""
Evaluate the following answer and give clear improvement suggestions:

{response.text}
"""

        feedback = llm.generate_content(feedback_prompt)

        st.markdown("### 📊 Feedback & Improvement Tips")
        st.write(feedback.text)
