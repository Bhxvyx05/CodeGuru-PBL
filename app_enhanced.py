import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils import extract_text_from_pdf, chunk_text
import google.generativeai as genai

# Import new modules
from skill_extractor import extract_skills, get_top_skills, categorize_skills
from gap_analyzer import analyze_gap, get_gap_recommendations
from star_evaluator import evaluate_answer_star, get_evaluation_color
from difficulty_adapter import DifficultyAdapter
from database import SessionDatabase

# ================== CONFIG ==================
st.set_page_config(
    page_title="CodeGuru - AI Interview Coach",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Initialize database and difficulty adapter
db = SessionDatabase()
difficulty_adapter = DifficultyAdapter()

# ================== INITIALIZE SESSION STATE ==================
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"

if "current_score" not in st.session_state:
    st.session_state.current_score = 50

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "answers_history" not in st.session_state:
    st.session_state.answers_history = []

# ================== UI LAYOUT ==================
# Professional header
col_title, col_logo = st.columns([3, 1])
with col_title:
    st.markdown("""
    <h1 style='margin: 0; font-size: 2.5em;'>CodeGuru</h1>
    <p style='margin: 0; font-size: 1.2em; color: #888;'>AI-Powered Interview Preparation System</p>
    <p style='margin: 5px 0 0 0; font-size: 0.9em; color: #666;'>Personalized Resume-Based Interview Coaching with Adaptive Learning</p>
    """, unsafe_allow_html=True)

st.divider()

# Sidebar
# Sidebar - Professional styling
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    language = st.selectbox(
        "Language Preference",
        ["English", "Hindi", "Hinglish"],
        key="language"
    )
    
    mode = st.radio(
        "Interview Mode",
        ["HR Interview", "Technical Interview", "Mock Interview"],
        key="mode",
        help="Select the type of interview you want to practice"
    )
    
    # Divider
    st.divider()
    
    # Session metrics with better styling
    st.markdown("### 📊 Current Session Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        difficulty = difficulty_adapter.get_difficulty(st.session_state.current_score)
        st.metric("Difficulty Level", difficulty.upper(), delta=None)
    with col2:
        st.metric("Current Score", f"{st.session_state.current_score}%")
    
    st.metric("Questions Answered", len(st.session_state.answers_history))
    
    st.divider()
    
    # Progress bar
    progress_pct = st.session_state.current_score / 100
    st.markdown("**Overall Progress**")
    st.progress(progress_pct)
    
    st.divider()
    
    # Help section
    with st.expander("❓ Quick Help"):
        st.markdown("""
        **How to use CodeGuru:**
        
        1. **Upload Resume** - PDF format only
        2. **Analyze Skills** - See detected skills
        3. **Gap Analysis** - Compare with job description
        4. **Practice** - Answer interview questions
        5. **Track Progress** - View improvement over time
        """)

# ================== MAIN TABS ==================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Resume Analysis",
    "🎯 Skill Gap Analysis",
    "🎤 Interview Practice",
    "📊 Progress Tracking",
    "ℹ️ How It Works"
])

# ================== TAB 1: RESUME ANALYSIS ==================
with tab1:
    st.header("📝 Resume Analysis & Skill Extraction")
    
    uploaded_resume = st.file_uploader(
        "Upload your Resume (PDF)",
        type=["pdf"],
        key="resume_uploader"
    )
    
    if uploaded_resume:
        st.success("✅ Resume uploaded successfully!")
        
        if st.button("🔍 Analyze Resume", key="analyze_btn"):
            with st.spinner("Extracting resume content..."):
                resume_text = extract_text_from_pdf(uploaded_resume)
                
                # Store in session
                st.session_state["resume_text"] = resume_text
                st.session_state["vector_db"] = process_resume(resume_text)
                
                st.success("✅ Resume processed and indexed!")
        
        # Display skill analysis if available
        if "resume_text" in st.session_state:
            st.divider()
            st.subheader("🏆 Extracted Skills & Confidence")
            
            resume_text = st.session_state["resume_text"]
            skills = extract_skills(resume_text)
            top_skills = get_top_skills(skills, top_n=12)
            
            # Display in columns with progress bars
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Skills Detected:**")
                for i, (skill, score) in enumerate(top_skills[:6], 1):
                    st.progress(score/100, text=f"{i}. {skill} - {score}%")
            
            with col2:
                categorized = categorize_skills(skills)
                st.write("**Skills by Category:**")
                for category, skill_list in categorized.items():
                    with st.expander(f"{category} ({len(skill_list)})"):
                        for skill, score in skill_list:
                            st.write(f"• {skill}: {score}%")
            
            # Display resume summary
            st.divider()
            st.subheader("📄 Resume Content")
            with st.expander("View Full Resume Text"):
                st.text_area("Resume Content:", value=resume_text, height=300, disabled=True)

# ================== TAB 2: SKILL GAP ANALYSIS ==================
with tab2:
    st.header("🎯 Job Description Gap Analysis")
    st.write("Paste your target job description to see how well your resume aligns with it.")
    
    if "resume_text" in st.session_state:
        jd_text = st.text_area(
            "📋 Paste Job Description Here",
            height=200,
            placeholder="Paste the job description to analyze skill gaps..."
        )
        
        if jd_text and st.button("📊 Analyze Gap", key="gap_btn"):
            resume_text = st.session_state["resume_text"]
            skills = extract_skills(resume_text)
            gap_analysis = analyze_gap(skills, jd_text)
            
            # Store in session
            st.session_state["gap_analysis"] = gap_analysis
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Match %", f"{gap_analysis['match_percentage']}%")
            with col2:
                st.metric("Matched Skills", gap_analysis['total_matched'])
            with col3:
                st.metric("Missing Skills", gap_analysis['gap_count'])
            with col4:
                st.metric("Extra Skills", len(gap_analysis['extra_skills']))
            
            # Detailed breakdown
            st.divider()
            
            col_match, col_missing, col_extra = st.columns(3)
            
            with col_match:
                st.success("✅ Matched Skills")
                if gap_analysis['matched_skills']:
                    for skill in gap_analysis['matched_skills']:
                        st.write(f"• {skill}")
                else:
                    st.write("None")
            
            with col_missing:
                st.error("❌ Missing Skills")
                if gap_analysis['missing_skills']:
                    for skill in gap_analysis['missing_skills']:
                        st.write(f"• {skill}")
                else:
                    st.write("All covered!")
            
            with col_extra:
                st.info("ℹ️ Extra Skills")
                if gap_analysis['extra_skills']:
                    for skill in gap_analysis['extra_skills']:
                        st.write(f"• {skill}")
                else:
                    st.write("None")
            
            # Recommendations
            st.divider()
            st.subheader("💡 Recommendations")
            recommendations = get_gap_recommendations(gap_analysis)
            for rec in recommendations:
                st.info(rec)
    else:
        st.warning("⚠️ Please upload and analyze your resume first (Tab 1)")

# ================== TAB 3: INTERVIEW PRACTICE ==================
with tab3:
    st.header("🎤 Interview Practice")
    
    if "vector_db" not in st.session_state:
        st.warning("⚠️ Please upload and analyze your resume first (Tab 1)")
    else:
        # Start session if not already started
        if st.session_state.session_id is None:
            if st.button("🚀 Start Practice Session"):
                skill = st.session_state.get("selected_skill", "General")
                st.session_state.session_id = db.create_session(
                    st.session_state.user_id,
                    st.session_state.mode,
                    skill
                )
                st.success(f"✅ Session started! Difficulty: {difficulty_adapter.get_difficulty(st.session_state.current_score).upper()}")
                st.rerun()
        
        if st.session_state.session_id:
            # Display current difficulty
            current_difficulty = difficulty_adapter.get_difficulty(st.session_state.current_score)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"📊 Difficulty: **{current_difficulty.upper()}**")
            with col2:
                st.info(f"💯 Your Score: **{st.session_state.current_score}%**")
            with col3:
                st.info(f"🎯 {difficulty_adapter.get_next_difficulty_recommendation(st.session_state.current_score)}")
            
            st.divider()
            
            # Question generation
            st.subheader("Get Next Question")
            
            skill_choice = st.selectbox(
                "Select Skill to Practice",
                ["Python", "SQL", "React", "System Design", "Problem Solving", "Communication"],
                key="selected_skill"
            )
            
            if st.button("🎲 Generate Question", key="gen_q_btn"):
                llm = load_llm()
                
                # Build prompt with difficulty
                difficulty_instruction = difficulty_adapter.build_difficulty_prompt_instruction(
                    current_difficulty,
                    skill_choice
                )
                
                prompt = f"""
You are an expert interviewer. {difficulty_instruction}

Resume Context:
{st.session_state.get('resume_text', 'No resume context')[:500]}...

Generate ONE interview question. Just the question, nothing else.
"""
                
                with st.spinner("Generating question..."):
                    response = llm.generate_content(prompt)
                    question = response.text.strip()
                    st.session_state.current_question = question
            
            # Display and answer question
            if "current_question" in st.session_state:
                st.subheader("📌 Question")
                st.write(st.session_state.current_question)
                
                st.divider()
                
                user_answer = st.text_area(
                    "Your Answer:",
                    height=150,
                    placeholder="Type your answer here...",
                    key="user_answer"
                )
                
                if st.button("✅ Submit Answer", key="submit_ans_btn"):
                    if not user_answer:
                        st.error("Please provide an answer!")
                    else:
                        with st.spinner("Evaluating your answer with STAR method..."):
                            llm = load_llm()
                            
                            # Get relevant context from resume
                            docs = st.session_state["vector_db"].similarity_search(
                                st.session_state.current_question, k=3
                            )
                            context = "\n\n".join([d.page_content for d in docs])
                            
                            # Evaluate with STAR
                            evaluation = evaluate_answer_star(
                                st.session_state.current_question,
                                user_answer,
                                context,
                                llm
                            )
                            
                            # Save to database
                            db.save_answer(
                                st.session_state.session_id,
                                st.session_state.current_question,
                                user_answer,
                                current_difficulty,
                                evaluation
                            )
                            
                            # Update running score
                            st.session_state.current_score = difficulty_adapter.update_score(
                                st.session_state.current_score,
                                int(evaluation["overall_score"])
                            )
                            
                            # Store in history
                            st.session_state.answers_history.append(evaluation)
                            
                            # Display evaluation
                            st.success("✅ Answer Evaluated!")
                            
                            st.divider()
                            st.subheader("📊 STAR Method Evaluation")
                            
                            # Score grid
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                color = get_evaluation_color(evaluation["relevance_score"])
                                st.metric(f"{color} Relevance", f"{evaluation['relevance_score']}/100")
                            with col2:
                                color = get_evaluation_color(evaluation["star_score"])
                                st.metric(f"{color} STAR", f"{evaluation['star_score']}/100")
                            with col3:
                                color = get_evaluation_color(evaluation["accuracy_score"])
                                st.metric(f"{color} Accuracy", f"{evaluation['accuracy_score']}/100")
                            with col4:
                                color = get_evaluation_color(evaluation["communication_score"])
                                st.metric(f"{color} Communication", f"{evaluation['communication_score']}/100")
                            
                            st.divider()
                            
                            # Overall score
                            overall = evaluation["overall_score"]
                            col_overall = st.columns(1)[0]
                            st.metric("🎯 Overall Score", f"{overall}/100")
                            
                            # Feedback breakdown
                            st.subheader("💬 Detailed Feedback")
                            
                            feedback_col1, feedback_col2 = st.columns(2)
                            
                            with feedback_col1:
                                st.write("**STAR Breakdown:**")
                                st.write(f"🟣 Situation: {evaluation['situation_feedback']}")
                                st.write(f"🟠 Task: {evaluation['task_feedback']}")
                            
                            with feedback_col2:
                                st.write("**Strengths & Areas:**")
                                st.write(f"💪 Strength: {evaluation['top_strength']}")
                                st.write(f"📈 Improve: {evaluation['improvement_area']}")
                            
                            st.divider()
                            st.subheader("🎯 Interview Tip for Next Answer")
                            st.info(evaluation["interview_tip"])
                            
                            # Clear question for next one
                            del st.session_state.current_question
                            
                            st.divider()
                            if st.button("🔄 Next Question"):
                                st.rerun()

# ================== TAB 4: PROGRESS TRACKING ==================
with tab4:
    st.header("📈 Progress Tracking & Analytics")
    
    metrics = db.get_progress_metrics(st.session_state.user_id)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Total Sessions", metrics["total_sessions"])
    with col2:
        st.metric("🎤 Total Questions", metrics["total_answers"])
    with col3:
        st.metric("⭐ Average Score", f"{metrics['average_score']}%")
    with col4:
        st.metric("👤 User ID", st.session_state.user_id[:12] + "...")
    
    st.divider()
    
    # Session history
    st.subheader("📋 Recent Sessions")
    history_df = db.get_user_history(st.session_state.user_id)
    
    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No sessions yet. Start practicing in the 'Interview Practice' tab!")
    
    # Score trend visualization
    if metrics["recent_trend"]:
        st.divider()
        st.subheader("📊 Score Trend (Last 5 Sessions)")
        
        import pandas as pd
        import plotly.graph_objects as go
        
        trend_data = metrics["recent_trend"]
        dates = [row[0] for row in trend_data]
        scores = [row[1] if row[1] else 0 for row in trend_data]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=scores,
            mode='lines+markers',
            name='Average Score',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Your Improvement Over Time",
            xaxis_title="Session Date",
            yaxis_title="Average Score (%)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ================== TAB 5: HOW IT WORKS ==================
with tab5:
    st.header("ℹ️ How CodeGuru Works")
    
    st.subheader("🎯 Why CodeGuru is Different")
    st.write("""
    Unlike existing interview tools that just answer questions, CodeGuru:
    
    1. **📊 Analyzes Your Resume** - Extracts skills and creates a semantic understanding
    2. **🎯 Matches Job Requirements** - Compares your skills against job description
    3. **⭐ Evaluates on STAR Method** - Scores answers on 4 dimensions (Relevance, STAR, Accuracy, Communication)
    4. **📈 Adapts Difficulty** - Questions get harder/easier based on YOUR performance
    5. **📚 Tracks Progress** - Remembers all sessions and shows improvement over time
    """)
    
    st.divider()
    
    st.subheader("⚙️ Architecture")
    st.write("""
    **Resume Processing Pipeline:**
    - Upload PDF → Extract Text → Chunk Content → Generate Embeddings → Store in FAISS
    
    **Question Generation:**
    - User Question → Retrieve Relevant Resume Context → Generate with Gemini
    
    **Answer Evaluation:**
    - Answer → STAR-based Rubric → Multi-dimensional Scoring → Feedback
    
    **Adaptive Learning:**
    - Track Performance → Adjust Difficulty → Update Progress History
    """)
    
    st.divider()
    
    st.subheader("🛠️ Tech Stack")
    st.write("""
    - **Frontend:** Streamlit (UI/UX)
    - **LLM:** Google Gemini API (Answer Generation & Evaluation)
    - **RAG:** LangChain + FAISS (Resume Context Retrieval)
    - **Embeddings:** HuggingFace Transformers (Semantic Understanding)
    - **Database:** SQLite (Session History)
    - **Visualization:** Plotly (Progress Charts)
    """)

# ================== FOOTER ==================
st.divider()
st.caption("🚀 CodeGuru - AI Interview Coach | Prep2Place | Made with ❤️")