import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import json

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils import extract_text_from_pdf, chunk_text

from skill_extractor import extract_skills, get_top_skills, categorize_skills
from gap_analyzer import analyze_gap, get_gap_recommendations
from star_evaluator import evaluate_answer_star, get_evaluation_color
from difficulty_adapter import DifficultyAdapter
from database import SessionDatabase

# ================== CONFIG ==================
st.set_page_config(
    page_title="CodeGuru - Interview Coach",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "CodeGuru - AI Interview Preparation System"}
)

# ================== CSS STYLING ==================
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    body {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #ecf0f1;
    }
    
    .main {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        color: #ecf0f1;
        border-radius: 8px;
        margin: 5px;
        font-weight: 600;
        font-size: 0.95em;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        border-radius: 8px;
    }
    
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        font-weight: 600;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 212, 255, 0.3);
    }
    
    h1 {
        color: #00d4ff;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    h2 {
        color: #38ef7d;
        border-bottom: 2px solid #00d4ff;
        padding-bottom: 10px;
    }
    
    h3 {
        color: #00d4ff;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #00d4ff, #38ef7d);
    }
</style>
""", unsafe_allow_html=True)

# ================== GEMINI CONFIG ==================
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ================== INITIALIZE COMPONENTS ==================
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
            return genai.GenerativeModel(m.name)
    raise Exception("No suitable Gemini model found")

db = SessionDatabase()
difficulty_adapter = DifficultyAdapter()

# ================== SESSION STATE ==================
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"

if "current_score" not in st.session_state:
    st.session_state.current_score = 50

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "answers_history" not in st.session_state:
    st.session_state.answers_history = []

# ================== HEADER ==================
col_header1, col_header2 = st.columns([3, 1])

with col_header1:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%); padding: 40px; border-radius: 15px; margin-bottom: 20px;'>
        <h1 style='margin: 0; color: white; font-size: 3em; text-shadow: 0 0 20px rgba(0, 0, 0, 0.3);'>🧠 CodeGuru</h1>
        <p style='margin: 10px 0 5px 0; color: #ecf0f1; font-size: 1.3em; font-weight: 600;'>AI Interview Preparation System</p>
        <p style='margin: 0; color: #ecf0f1; font-size: 0.95em; opacity: 0.9;'>Resume-Based Coaching with Adaptive Learning</p>
    </div>
    """, unsafe_allow_html=True)

with col_header2:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; margin-top: 20px;'>
        <h3 style='margin: 0; color: white; text-align: center;'>⭐ Smart Interview Prep</h3>
        <p style='margin: 10px 0 0 0; color: #ecf0f1; font-size: 0.85em; text-align: center;'>Powered by Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

# ================== FEATURES SHOWCASE ==================
st.markdown("""
<div style='background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(56, 239, 125, 0.1) 100%); padding: 25px; border-radius: 12px; border: 2px solid #00d4ff; margin: 20px 0;'>
    <h3 style='margin-top: 0; color: #00d4ff; text-align: center;'>🚀 Why CodeGuru Stands Out</h3>
    <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 15px;'>
        <div style='background: rgba(0, 212, 255, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #00d4ff;'>
            <p style='margin: 0; color: #38ef7d; font-weight: 700;'>✓ Resume Analysis</p>
            <p style='margin: 5px 0 0 0; color: #ecf0f1; font-size: 0.9em;'>Deep skill extraction & insights</p>
        </div>
        <div style='background: rgba(0, 212, 255, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #00d4ff;'>
            <p style='margin: 0; color: #38ef7d; font-weight: 700;'>✓ STAR Evaluation</p>
            <p style='margin: 5px 0 0 0; color: #ecf0f1; font-size: 0.9em;'>4-dimensional answer scoring</p>
        </div>
        <div style='background: rgba(0, 212, 255, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #00d4ff;'>
            <p style='margin: 0; color: #38ef7d; font-weight: 700;'>✓ Question Banks</p>
            <p style='margin: 5px 0 0 0; color: #ecf0f1; font-size: 0.9em;'>Downloadable PDF sets</p>
        </div>
        <div style='background: rgba(0, 212, 255, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #00d4ff;'>
            <p style='margin: 0; color: #38ef7d; font-weight: 700;'>✓ Gap Analysis</p>
            <p style='margin: 5px 0 0 0; color: #ecf0f1; font-size: 0.9em;'>vs Job Descriptions</p>
        </div>
        <div style='background: rgba(0, 212, 255, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #00d4ff;'>
            <p style='margin: 0; color: #38ef7d; font-weight: 700;'>✓ Adaptive Difficulty</p>
            <p style='margin: 5px 0 0 0; color: #ecf0f1; font-size: 0.9em;'>Dynamic question difficulty</p>
        </div>
        <div style='background: rgba(0, 212, 255, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #00d4ff;'>
            <p style='margin: 0; color: #38ef7d; font-weight: 700;'>✓ Progress Tracking</p>
            <p style='margin: 5px 0 0 0; color: #ecf0f1; font-size: 0.9em;'>Analytics & improvement</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;'>
        <h3 style='color: white; margin: 0 0 15px 0;'>⚙️ Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    language = st.selectbox(
        "🌍 Language",
        ["English", "Hindi", "Hinglish"],
        key="language"
    )
    
    st.divider()
    
    with st.expander("📌 How to Use"):
        st.markdown("""
        1. Upload Resume
        2. Analyze Skills
        3. Compare with JD
        4. Practice Questions
        5. Track Progress
        """)

# ================== MAIN TABS ==================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Resume Analysis",
    "🎯 Gap Analysis",
    "🎤 Interview Practice",
    "📈 Progress",
    "ℹ️ About"
])

# ================== TAB 1: RESUME ANALYSIS ==================
with tab1:
    st.markdown("<h2 style='color: #00d4ff;'>📝 Resume Analysis & Skill Insights</h2>", unsafe_allow_html=True)
    st.markdown("Upload your resume for comprehensive AI-powered analysis.")
    st.divider()
    
    uploaded_resume = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"], key="resume_uploader")
    
    if uploaded_resume:
        st.markdown("<div class='success-card'>✅ Resume uploaded successfully!</div>", unsafe_allow_html=True)
        
        if st.button("🔬 Analyze Resume", use_container_width=True, key="analyze_btn"):
            with st.spinner("⏳ Analyzing resume comprehensively..."):
                try:
                    resume_text = extract_text_from_pdf(uploaded_resume)
                    st.session_state["resume_text"] = resume_text
                    st.session_state["vector_db"] = process_resume(resume_text)
                    
                    # Generate analysis
                    from resume_analyzer import generate_detailed_resume_analysis
                    llm = load_llm()
                    detailed_analysis = generate_detailed_resume_analysis(resume_text, llm)
                    st.session_state["resume_analysis"] = detailed_analysis
                    
                    st.markdown("<div class='success-card'>✅ Analysis Complete!</div>", unsafe_allow_html=True)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error analyzing resume: {str(e)}")
        
        # Display analysis
        if "resume_analysis" in st.session_state:
            analysis = st.session_state["resume_analysis"]
            
            # Professional Summary
            st.markdown("<h3 style='color: #38ef7d;'>📋 Professional Summary</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='info-card'>
                {analysis.get('professional_summary', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Key Strengths
            st.markdown("<h3 style='color: #38ef7d;'>✨ Your Key Strengths</h3>", unsafe_allow_html=True)
            strengths = analysis.get('key_strengths', [])
            
            if strengths:
                cols_str = st.columns(2)
                for idx, strength in enumerate(strengths):
                    with cols_str[idx % 2]:
                        st.markdown(f"""
                        <div class='success-card' style='margin: 10px 0;'>
                            ✓ {strength}
                        </div>
                        """, unsafe_allow_html=True)
            
            st.divider()
            
            # Experience
            st.markdown("<h3 style='color: #38ef7d;'>💼 Professional Experience</h3>", unsafe_allow_html=True)
            experiences = analysis.get('experience', [])
            
            if experiences:
                for exp in experiences:
                    with st.expander(f"🏢 {exp.get('position', 'N/A')} @ {exp.get('company', 'N/A')} ({exp.get('duration', 'N/A')})"):
                        st.write("**Key Achievements:**")
                        for achievement in exp.get('achievements', []):
                            st.write(f"• {achievement}")
            else:
                st.info("No experience details found")
            
            st.divider()
            
            # Projects
            st.markdown("<h3 style='color: #38ef7d;'>🚀 Project Portfolio</h3>", unsafe_allow_html=True)
            projects = analysis.get('projects', [])
            
            if projects:
                for idx, project in enumerate(projects, 1):
                    st.write(f"**Project {idx}: {project.get('name', 'N/A')}**")
                    st.write(f"📝 {project.get('description', 'N/A')}")
                    st.write(f"💻 Tech: {', '.join(project.get('technologies', []))}")
                    st.write(f"⭐ Impact: {project.get('impact', 'N/A')}")
                    st.divider()
            else:
                st.info("No projects found")
            
            # Skills
            st.markdown("<h3 style='color: #38ef7d;'>💻 Technical Skills</h3>", unsafe_allow_html=True)
            tech_skills = analysis.get('technical_skills', {})
            
            if tech_skills:
                for category, skills in tech_skills.items():
                    if isinstance(skills, list) and skills:
                        st.write(f"**{category.upper()}:**")
                        for skill in skills:
                            if isinstance(skill, dict):
                                st.write(f"• {skill.get('skill', 'N/A')} - {skill.get('proficiency', 'N/A')}")
                        st.divider()
            
            # Soft Skills
            st.markdown("<h3 style='color: #38ef7d;'>🤝 Soft Skills</h3>", unsafe_allow_html=True)
            soft_skills = analysis.get('soft_skills', [])
            
            if soft_skills:
                cols = st.columns(3)
                for idx, skill in enumerate(soft_skills):
                    with cols[idx % 3]:
                        st.metric(skill, "✓")
            
            st.divider()
            
            # Improvement Suggestions
            st.markdown("<h3 style='color: #00d4ff;'>💡 Improvement Suggestions</h3>", unsafe_allow_html=True)
            suggestions = analysis.get('improvement_suggestions', [])
            
            if suggestions:
                for idx, sug in enumerate(suggestions, 1):
                    with st.expander(f"📌 Suggestion {idx}: {sug.get('area', 'N/A')}"):
                        st.write(f"**Current:** {sug.get('current_state', 'N/A')}")
                        st.write(f"**Suggestion:** {sug.get('suggestion', 'N/A')}")
                        st.write(f"**Impact:** {sug.get('impact', 'N/A')}")
            
            st.divider()
            
            # Overall Score
            scores = analysis.get('scores', {})
            overall = scores.get('overall', 0)
            
            col_sc1, col_sc2, col_sc3, col_sc4 = st.columns(4)
            with col_sc1:
                st.metric("💻 Technical", f"{scores.get('technical_strength', 0)}/100")
            with col_sc2:
                st.metric("🗣️ Communication", f"{scores.get('communication_clarity', 0)}/100")
            with col_sc3:
                st.metric("🚀 Projects", f"{scores.get('project_quality', 0)}/100")
            with col_sc4:
                st.metric("📈 Growth", f"{scores.get('career_growth', 0)}/100")
            
            st.divider()
            
            # BIG SCORE
            score_color = "#38ef7d" if overall >= 70 else "#ff6b6b"
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {score_color} 0%, {"#38ef7d" if overall >= 70 else "#ee5a6f"} 100%); padding: 40px; border-radius: 15px; text-align: center; margin: 20px 0;'>
                <h1 style='color: white; margin: 0; font-size: 3em;'>⭐ {overall}/100</h1>
                <p style='color: white; margin: 10px 0 0 0; font-size: 1.2em;'>OVERALL PROFILE STRENGTH</p>
            </div>
            """, unsafe_allow_html=True)

# ================== TAB 2: GAP ANALYSIS ==================
with tab2:
    st.markdown("<h2 style='color: #00d4ff;'>🎯 Job Description Gap Analysis</h2>", unsafe_allow_html=True)
    
    if "resume_text" not in st.session_state:
        st.markdown("<div class='warning-card'>📌 Upload resume first (Tab 1)</div>", unsafe_allow_html=True)
    else:
        jd_text = st.text_area("📋 Paste Job Description", height=200)
        
        if jd_text and st.button("��� Analyze Gap", use_container_width=True):
            resume_text = st.session_state["resume_text"]
            skills = extract_skills(resume_text)
            gap_analysis = analyze_gap(skills, jd_text)
            st.session_state["gap_analysis"] = gap_analysis
        
        if "gap_analysis" in st.session_state:
            gap_analysis = st.session_state["gap_analysis"]
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Match %", f"{gap_analysis['match_percentage']}%")
            with col_m2:
                st.metric("Matched", gap_analysis['total_matched'])
            with col_m3:
                st.metric("Missing", gap_analysis['gap_count'])
            with col_m4:
                st.metric("Extra", len(gap_analysis['extra_skills']))
            
            st.divider()
            
            col_matched, col_missing, col_extra = st.columns(3)
            
            with col_matched:
                st.markdown("<h4 style='color: #38ef7d;'>✅ Matched</h4>", unsafe_allow_html=True)
                for skill in gap_analysis['matched_skills']:
                    st.markdown(f"<div class='success-card'>✓ {skill}</div>", unsafe_allow_html=True)
            
            with col_missing:
                st.markdown("<h4 style='color: #ff6b6b;'>❌ Missing</h4>", unsafe_allow_html=True)
                for skill in gap_analysis['missing_skills']:
                    st.markdown(f"<div class='warning-card'>• {skill}</div>", unsafe_allow_html=True)
            
            with col_extra:
                st.markdown("<h4 style='color: #00d4ff;'>ℹ️ Extra</h4>", unsafe_allow_html=True)
                for skill in gap_analysis['extra_skills']:
                    st.markdown(f"<div class='info-card'>• {skill}</div>", unsafe_allow_html=True)

# ================== TAB 3: INTERVIEW PRACTICE ==================
with tab3:
    st.markdown("<h2 style='color: #00d4ff;'>🎤 Interview Practice with STAR Evaluation</h2>", unsafe_allow_html=True)
    st.markdown("Select your interview mode and practice with AI-generated questions.")
    st.divider()
    
    if "resume_text" not in st.session_state:
        st.markdown("<div class='warning-card'>📌 Please upload resume first (Tab 1)</div>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: #00d4ff;'>Step 1: Choose Interview Mode</h3>", unsafe_allow_html=True)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            if st.button("🗣️ HR Interview", use_container_width=True, key="hr_btn_select"):
                st.session_state.interview_mode = "HR"
        
        with col_m2:
            if st.button("💻 Technical Interview", use_container_width=True, key="tech_btn_select"):
                st.session_state.interview_mode = "Technical"
        
        with col_m3:
            if st.button("🎭 Mock Interview", use_container_width=True, key="mock_btn_select"):
                st.session_state.interview_mode = "Mock"
        
        if "interview_mode" in st.session_state:
            st.divider()
            mode = st.session_state.interview_mode
            
            st.markdown(f"<h3 style='color: #38ef7d;'>{mode} Interview Mode Selected ✓</h3>", unsafe_allow_html=True)
            
            # ============ HR INTERVIEW ============
            if mode == "HR":
                st.markdown("**Behavioral questions to assess soft skills, communication, and teamwork**")
                
                col_lang, col_count = st.columns(2)
                
                with col_lang:
                    language = st.selectbox("Language", ["English", "Hindi", "Hinglish"], key="hr_lang")
                
                with col_count:
                    q_count = st.slider("Questions", 1, 20, 5, key="hr_count")
                
                if st.button("🎲 Generate HR Questions", use_container_width=True, key="gen_hr"):
                    with st.spinner("Generating HR questions..."):
                        try:
                            llm = load_llm()
                            resume_text = st.session_state["resume_text"]
                            
                            prompt = f"""Generate {q_count} behavioral HR interview questions in {language}.
These should assess soft skills, communication, teamwork, leadership.
Based on resume context: {resume_text[:500]}

Return as JSON array: [{{ "question": "...", "difficulty": "easy/medium/hard" }}]
Return ONLY JSON, no markdown."""
                            
                            response = llm.generate_content(prompt)
                            text = response.text.strip()
                            
                            if "```" in text:
                                text = text.split("```")[1]
                                if text.startswith("json"):
                                    text = text[4:]
                                text = text.split("```")[0]
                            
                            questions = json.loads(text)
                            st.session_state.hr_questions = questions
                            st.success(f"✅ Generated {len(questions)} HR questions!")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                # Display HR questions
                if "hr_questions" in st.session_state:
                    st.markdown("---")
                    st.write("### 📋 HR Question Bank")
                    
                    for idx, q in enumerate(st.session_state.hr_questions, 1):
                        with st.expander(f"Q{idx}: {q.get('question', 'N/A')[:60]}..."):
                            st.write(q.get('question', 'N/A'))
                            st.write(f"Difficulty: {q.get('difficulty', 'N/A')}")
                    
                    st.markdown("---")
                    
                    # Practice section
                    st.write("### 🎤 Practice a Question")
                    q_select = st.selectbox("Select:", [f"Q{i+1}: {q['question'][:50]}..." for i, q in enumerate(st.session_state.hr_questions)], key="hr_select")
                    q_idx = int(q_select.split(":")[0][1:]) - 1
                    current_q = st.session_state.hr_questions[q_idx]["question"]
                    
                    st.write(f"**Question:** {current_q}")
                    
                    answer = st.text_area("Your Answer:", height=120, key=f"hr_answer_{q_idx}")
                    
                    if st.button("✅ Submit & Evaluate", use_container_width=True, key=f"hr_submit_{q_idx}"):
                        if answer:
                            with st.spinner("Evaluating..."):
                                try:
                                    llm = load_llm()
                                    docs = st.session_state["vector_db"].similarity_search(current_q, k=3)
                                    context = "\n".join([d.page_content for d in docs])
                                    
                                    eval_prompt = f"""Evaluate this interview answer on 4 dimensions (0-100 each):

Question: {current_q}
Answer: {answer}
Resume Context: {context[:300]}

Return JSON:
{{ "relevance": 0-100, "star": 0-100, "accuracy": 0-100, "communication": 0-100, 
   "feedback": "Brief feedback", "tip": "Interview tip" }}

Return ONLY JSON."""
                                    
                                    response = llm.generate_content(eval_prompt)
                                    text = response.text.strip()
                                    
                                    if "```" in text:
                                        text = text.split("```")[1]
                                        if text.startswith("json"):
                                            text = text[4:]
                                        text = text.split("```")[0]
                                    
                                    eval_data = json.loads(text)
                                    
                                    st.markdown("---")
                                    st.markdown("### 📊 Evaluation Results")
                                    
                                    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                                    with col_e1:
                                        st.metric("Relevance", f"{eval_data.get('relevance', 0)}/100")
                                    with col_e2:
                                        st.metric("STAR", f"{eval_data.get('star', 0)}/100")
                                    with col_e3:
                                        st.metric("Accuracy", f"{eval_data.get('accuracy', 0)}/100")
                                    with col_e4:
                                        st.metric("Communication", f"{eval_data.get('communication', 0)}/100")
                                    
                                    st.divider()
                                    st.info(f"💬 {eval_data.get('feedback', 'N/A')}")
                                    st.success(f"🎯 Tip: {eval_data.get('tip', 'N/A')}")
                                    
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                        else:
                            st.error("Please provide an answer!")
            
            # ============ TECHNICAL INTERVIEW ============
            elif mode == "Technical":
                st.markdown("**Technical questions based on your skills and experience**")
                
                resume_text = st.session_state["resume_text"]
                skills = extract_skills(resume_text)
                top_skills = get_top_skills(skills, top_n=10)
                skill_names = [s[0] for s in top_skills]
                
                col_skill, col_lang, col_count = st.columns(3)
                
                with col_skill:
                    selected_skill = st.selectbox("Skill", skill_names, key="tech_skill")
                
                with col_lang:
                    language = st.selectbox("Language", ["English", "Hindi", "Hinglish"], key="tech_lang")
                
                with col_count:
                    q_count = st.slider("Questions", 1, 20, 5, key="tech_count")
                
                if st.button("🎲 Generate Technical Questions", use_container_width=True, key="gen_tech"):
                    with st.spinner(f"Generating {selected_skill} questions..."):
                        try:
                            llm = load_llm()
                            
                            prompt = f"""Generate {q_count} technical interview questions about {selected_skill} in {language}.
Based on resume: {resume_text[:500]}

Questions should cover: concepts, practical application, problem-solving.
Return as JSON: [{{ "question": "...", "difficulty": "easy/medium/hard" }}]
Return ONLY JSON."""
                            
                            response = llm.generate_content(prompt)
                            text = response.text.strip()
                            
                            if "```" in text:
                                text = text.split("```")[1]
                                if text.startswith("json"):
                                    text = text[4:]
                                text = text.split("```")[0]
                            
                            questions = json.loads(text)
                            st.session_state.tech_questions = questions
                            st.success(f"✅ Generated {len(questions)} {selected_skill} questions!")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                # Display technical questions
                if "tech_questions" in st.session_state:
                    st.markdown("---")
                    st.write(f"### 📋 {selected_skill} Question Bank")
                    
                    for idx, q in enumerate(st.session_state.tech_questions, 1):
                        with st.expander(f"Q{idx}: {q.get('question', 'N/A')[:60]}..."):
                            st.write(q.get('question', 'N/A'))
                            st.write(f"Difficulty: {q.get('difficulty', 'N/A')}")
                    
                    st.markdown("---")
                    st.write("### 🎤 Practice a Question")
                    q_select = st.selectbox("Select:", [f"Q{i+1}: {q['question'][:50]}..." for i, q in enumerate(st.session_state.tech_questions)], key="tech_select")
                    q_idx = int(q_select.split(":")[0][1:]) - 1
                    current_q = st.session_state.tech_questions[q_idx]["question"]
                    
                    st.write(f"**Question:** {current_q}")
                    answer = st.text_area("Your Answer:", height=120, key=f"tech_answer_{q_idx}")
                    
                    if st.button("✅ Submit & Evaluate", use_container_width=True, key=f"tech_submit_{q_idx}"):
                        if answer:
                            with st.spinner("Evaluating..."):
                                try:
                                    llm = load_llm()
                                    docs = st.session_state["vector_db"].similarity_search(current_q, k=3)
                                    context = "\n".join([d.page_content for d in docs])
                                    
                                    eval_prompt = f"""Evaluate this technical answer (0-100 each):

Question: {current_q}
Answer: {answer}
Resume Context: {context[:300]}

Return JSON:
{{ "relevance": 0-100, "technical_depth": 0-100, "clarity": 0-100, "completeness": 0-100,
   "feedback": "Brief feedback", "tip": "Interview tip" }}

Return ONLY JSON."""
                                    
                                    response = llm.generate_content(eval_prompt)
                                    text = response.text.strip()
                                    
                                    if "```" in text:
                                        text = text.split("```")[1]
                                        if text.startswith("json"):
                                            text = text[4:]
                                        text = text.split("```")[0]
                                    
                                    eval_data = json.loads(text)
                                    
                                    st.markdown("---")
                                    st.markdown("### 📊 Evaluation Results")
                                    
                                    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                                    with col_e1:
                                        st.metric("Relevance", f"{eval_data.get('relevance', 0)}/100")
                                    with col_e2:
                                        st.metric("Depth", f"{eval_data.get('technical_depth', 0)}/100")
                                    with col_e3:
                                        st.metric("Clarity", f"{eval_data.get('clarity', 0)}/100")
                                    with col_e4:
                                        st.metric("Completeness", f"{eval_data.get('completeness', 0)}/100")
                                    
                                    st.divider()
                                    st.info(f"💬 {eval_data.get('feedback', 'N/A')}")
                                    st.success(f"🎯 Tip: {eval_data.get('tip', 'N/A')}")
                                    
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                        else:
                            st.error("Please provide an answer!")
            
            # ============ MOCK INTERVIEW ============
            elif mode == "Mock":
                st.markdown("**Mixed HR and Technical questions for complete interview simulation**")
                
                col_lang, col_count = st.columns(2)
                
                with col_lang:
                    language = st.selectbox("Language", ["English", "Hindi", "Hinglish"], key="mock_lang")
                
                with col_count:
                    q_count = st.slider("Questions", 1, 20, 5, key="mock_count")
                
                if st.button("🎲 Generate Mock Interview Questions", use_container_width=True, key="gen_mock"):
                    with st.spinner("Generating mock interview questions..."):
                        try:
                            llm = load_llm()
                            resume_text = st.session_state["resume_text"]
                            
                            prompt = f"""Generate {q_count} mixed interview questions (HR + Technical) in {language}.
Based on resume: {resume_text[:500]}

Include both behavioral (HR) and technical questions.
Return as JSON: [{{ "question": "...", "type": "hr/technical", "difficulty": "easy/medium/hard" }}]
Return ONLY JSON."""
                            
                            response = llm.generate_content(prompt)
                            text = response.text.strip()
                            
                            if "```" in text:
                                text = text.split("```")[1]
                                if text.startswith("json"):
                                    text = text[4:]
                                text = text.split("```")[0]
                            
                            questions = json.loads(text)
                            st.session_state.mock_questions = questions
                            st.success(f"✅ Generated {len(questions)} mock interview questions!")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                # Display mock questions
                if "mock_questions" in st.session_state:
                    st.markdown("---")
                    st.write("### 📋 Mock Interview Question Bank")
                    
                    for idx, q in enumerate(st.session_state.mock_questions, 1):
                        q_type = q.get('type', 'general').upper()
                        with st.expander(f"Q{idx} ({q_type}): {q.get('question', 'N/A')[:60]}..."):
                            st.write(q.get('question', 'N/A'))
                            st.write(f"Type: {q_type} | Difficulty: {q.get('difficulty', 'N/A')}")
                    
                    st.markdown("---")
                    st.write("### 🎤 Practice a Question")
                    q_select = st.selectbox("Select:", [f"Q{i+1}: {q['question'][:50]}..." for i, q in enumerate(st.session_state.mock_questions)], key="mock_select")
                    q_idx = int(q_select.split(":")[0][1:]) - 1
                    current_q = st.session_state.mock_questions[q_idx]["question"]
                    
                    st.write(f"**Question:** {current_q}")
                    answer = st.text_area("Your Answer:", height=120, key=f"mock_answer_{q_idx}")
                    
                    if st.button("✅ Submit & Evaluate", use_container_width=True, key=f"mock_submit_{q_idx}"):
                        if answer:
                            with st.spinner("Evaluating..."):
                                try:
                                    llm = load_llm()
                                    docs = st.session_state["vector_db"].similarity_search(current_q, k=3)
                                    context = "\n".join([d.page_content for d in docs])
                                    
                                    eval_prompt = f"""Evaluate this interview answer (0-100 each):

Question: {current_q}
Answer: {answer}
Resume Context: {context[:300]}

Return JSON:
{{ "score1": 0-100, "score2": 0-100, "score3": 0-100, "score4": 0-100,
   "feedback": "Brief feedback", "tip": "Interview tip" }}

Return ONLY JSON."""
                                    
                                    response = llm.generate_content(eval_prompt)
                                    text = response.text.strip()
                                    
                                    if "```" in text:
                                        text = text.split("```")[1]
                                        if text.startswith("json"):
                                            text = text[4:]
                                        text = text.split("```")[0]
                                    
                                    eval_data = json.loads(text)
                                    
                                    st.markdown("---")
                                    st.markdown("### 📊 Evaluation Results")
                                    
                                    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                                    with col_e1:
                                        st.metric("Relevance", f"{eval_data.get('score1', 0)}/100")
                                    with col_e2:
                                        st.metric("Structure", f"{eval_data.get('score2', 0)}/100")
                                    with col_e3:
                                        st.metric("Accuracy", f"{eval_data.get('score3', 0)}/100")
                                    with col_e4:
                                        st.metric("Communication", f"{eval_data.get('score4', 0)}/100")
                                    
                                    st.divider()
                                    st.info(f"💬 {eval_data.get('feedback', 'N/A')}")
                                    st.success(f"🎯 Tip: {eval_data.get('tip', 'N/A')}")
                                    
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                        else:
                            st.error("Please provide an answer!")

# ================== TAB 4: PROGRESS ==================
with tab4:
    st.markdown("<h2 style='color: #00d4ff;'>📈 Progress Tracking</h2>", unsafe_allow_html=True)
    st.markdown("Track your interview preparation journey.")
    st.divider()
    
    metrics = db.get_progress_metrics(st.session_state.user_id)
    
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.metric("📚 Sessions", metrics["total_sessions"])
    with col_p2:
        st.metric("🎤 Questions", metrics["total_answers"])
    with col_p3:
        st.metric("⭐ Avg Score", f"{metrics['average_score']}%")
    with col_p4:
        st.metric("👤 User", st.session_state.user_id[:10] + "...")
    
    st.divider()
    st.info("📊 Detailed analytics coming soon!")

# ================== TAB 5: ABOUT ==================
with tab5:
    st.markdown("<h2 style='color: #00d4ff;'>ℹ️ About CodeGuru</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
        <h3 style='margin-top: 0; color: white;'>🎯 Our Mission</h3>
        <p>CodeGuru transforms interview preparation with AI-powered coaching that understands YOUR resume, YOUR skills, and YOUR goals.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='success-card'>
        <h3 style='margin-top: 0; color: white;'>✨ Key Features</h3>
        <ul style='color: white; margin: 10px 0 0 0;'>
            <li>🧠 AI-powered resume analysis</li>
            <li>📊 STAR method evaluation</li>
            <li>🎯 Job description gap analysis</li>
            <li>📥 Downloadable question banks</li>
            <li>📈 Adaptive difficulty system</li>
            <li>📋 Progress tracking</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='color: #00d4ff; font-weight: 600;'>🚀 CodeGuru - AI Interview Preparation</p>
</div>
""", unsafe_allow_html=True)