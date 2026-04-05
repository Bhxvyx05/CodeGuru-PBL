# CodeGuru: AI-Powered Interview Preparation System

<div align="center">
  
![CodeGuru Logo](https://img.shields.io/badge/CodeGuru-AI%20Interview%20Coach-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9%2B-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge)

**Transform Your Interview Preparation with AI-Powered Personalized Coaching**

[Features](#-features) • [Usage](#-usage) • [Tech Stack](#-tech-stack) • [Architecture](#-architecture) 

</div>

---

## 🎯 Overview

**CodeGuru** is an intelligent interview preparation system that combines **Resume-Based AI Coaching**, **STAR Method Evaluation**, and **Adaptive Learning** to help students and job seekers ace their interviews.

Unlike generic interview prep tools like ChatGPT or LeetCode, CodeGuru:
- 🧠 **Understands YOUR Resume** - Generates personalized questions based on your experience
- 📊 **Evaluates with STAR Method** - 4-dimensional scoring (Relevance, STAR, Accuracy, Communication)
- 🎯 **Analyzes Job Gaps** - Shows exactly which skills you need to practice
- 📈 **Tracks Progress** - Visualize improvement with charts and analytics
- 🌍 **Multi-Language Support** - Practice in English, Hindi, or Hinglish
- 💰 **Completely FREE** - No subscription fees, open-source

---

## ✨ Key Features

### 1. 📝 Resume Analysis
- Extracts skills, experience, projects, and education from PDF resumes
- AI-powered skill proficiency detection
- Comprehensive profile strength scoring (0-100)
- Detailed improvement suggestions

### 2. 🎯 Job Description Gap Analysis
- Compare your resume against job requirements
- Identify matched, missing, and extra skills
- Get match percentage and recommendations
- Prioritize skills to practice based on gaps

### 3. 🎤 Interview Practice (3 Modes)

#### HR Interview Mode
- Behavioral questions assessing soft skills, communication, teamwork
- Resume-grounded personalized questions
- Real-time STAR method evaluation

#### Technical Interview Mode
- Questions tailored to your technical skills
- Deep dive into concepts, practical application, problem-solving
- Skill-specific question generation

#### Mock Interview Mode
- Complete interview simulation with mixed questions
- Combines HR and Technical assessment
- End-to-end interview experience

### 4. 📊 STAR Method Evaluation
Each answer is evaluated on 4 dimensions:
- **Relevance** (0-100): Does it use YOUR actual experience?
- **STAR Alignment** (0-100): Situation→Task→Action→Result structure?
- **Accuracy** (0-100): No contradictions with resume?
- **Communication** (0-100): Clear, confident, easy to follow?

### 5. 📈 Progress Tracking & Analytics
- Real-time progress visualization
- Score progression charts
- Performance breakdown by dimension
- Session history and statistics
- Improvement trends

---

## 🛠️ Tech Stack

### Frontend
- **Streamlit** (v1.28.1) - Web UI framework
- **HTML/CSS** - Styling and animations
- **Plotly** (v5.17.0) - Interactive data visualization

### Backend & AI/ML
- **Google Gemini API** - LLM for intelligent generation & evaluation
- **LangChain** (v0.1.1) - RAG (Retrieval-Augmented Generation) framework
- **FAISS** (v1.7.4) - Vector database for semantic search
- **HuggingFace Transformers** (v2.2.2) - Embedding models

### Data Processing
- **PyPDF** (v3.17.1) - PDF text extraction
- **Pandas** (v2.0.3) - Data manipulation
- **ReportLab** (v4.0.9) - PDF generation

### Database
- **SQLite3** - Session persistence and history

### Development
- **Python** (3.9+)
- **Git** - Version control
- **Virtual Environment** - Dependency isolation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                      (Streamlit Web UI)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Resume Analysis  │ │ Question Gen     │ │ Answer Eval      │
│ - PDF Extract    │ │ - Gemini LLM     │ │ - STAR Scoring   │
│ - Skill Extract  │ │ - HR Questions   │ │ - 4D Evaluation  │
│ - Profile Score  │ │ - Tech Questions │ │ - Feedback Gen   │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                     │                    │
         ▼                     ▼                    ▼
    ┌─────────────────────────────────────────────────────┐
    │          Semantic Intelligence Layer                │
    │  - LangChain (RAG Framework)                        │
    │  - HuggingFace Embeddings                           │
    │  - FAISS Vector Database                            │
    └──────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Google   │ │ FAISS    │ │ SQLite   │
    │ Gemini   │ │ Vector   │ │ Database │
    │ API      │ │ Store    │ │ (History)│
    └──────────┘ └──────────┘ └──────────┘
```

---
## 📖 Usage Guide

### 1️⃣ Upload & Analyze Resume

**Step 1:** Go to **Resume Analysis** tab

**Step 2:** Click "📄 Upload Resume (PDF)"

**Step 3:** Click "🔬 Analyze Resume"

**Step 4:** View analysis:
- Professional summary
- Key strengths
- Experience breakdown
- Projects portfolio
- Technical skills
- Soft skills
- Improvement suggestions
- Overall profile score (0-100)

### 2️⃣ Compare Job Description (Optional)

**Step 1:** Go to **Gap Analysis** tab

**Step 2:** Paste job description in text area

**Step 3:** Click "📊 Analyze Gap"

**Step 4:** View results:
- Match percentage
- Matched skills ✅
- Missing skills ❌
- Extra skills ℹ️

### 3️⃣ Practice Interview Questions

**Step 1:** Go to **Interview Practice** tab

**Step 2:** Choose interview mode:
```
🗣️ HR Interview        (Behavioral questions)
💻 Technical Interview (Skill-based technical)
🎭 Mock Interview      (Mixed HR + Technical)
```

**Step 3:** Configure settings:
- Select language (English/Hindi/Hinglish)
- Choose number of questions (1-20)

**Step 4:** Click "🎲 Generate Questions"

**Step 5:** Select a question from the list

**Step 6:** Type your answer in text area

**Step 7:** Click "✅ Submit & Evaluate"

### 4️⃣ View Evaluation Results

After answering:
- See 4 dimension scores (0-100 each):
  - Relevance
  - STAR Method
  - Accuracy
  - Communication
- Read detailed feedback
- Get interview tips

**Important:** Your answer is automatically saved to Progress!

### 5️⃣ Track Progress & Analytics

**Step 1:** Go to **Progress** tab

**Step 2:** View metrics:
- Total sessions
- Questions answered
- Average score
- Your user ID

**Step 3:** See all practice sessions:
- Question number
- Individual scores
- Overall score
- Feedback and tips

**Step 4:** Check charts:
- Score progression chart
- Performance by dimension
- Statistics (highest, lowest, average, trend)

---

## 🔬 STAR Method Explained

**STAR** is an industry-standard behavioral interview technique:

| Component | Definition | Example |
|-----------|-----------|---------|
| **Situation** | Context of the experience | "I was assigned to lead a team project..." |
| **Task** | Your responsibility | "My role was to ensure on-time delivery..." |
| **Action** | What you specifically did | "I implemented a tracking system..." |
| **Result** | Quantifiable outcome | "This increased efficiency by 40%" |

### CodeGuru's 4-Dimensional Scoring

1. **Relevance Score (0-100)**
   - Does your answer use YOUR actual experience?
   - Does it reference YOUR resume/projects?
   - Avoids generic or made-up stories

2. **STAR Alignment Score (0-100)**
   - Is there clear Situation?
   - Is Task well-defined?
   - Are Actions specific and detailed?
   - Are Results quantifiable?

3. **Accuracy Score (0-100)**
   - No contradictions with your resume
   - Technical details are correct
   - Timeline makes sense

4. **Communication Score (0-100)**
   - Clear and easy to understand
   - Confident tone
   - Proper pacing and structure
   - Professional language

---

## 📊 Competitive Analysis

| Feature | ChatGPT | Pramp | LeetCode | InterviewBit | **CodeGuru** |
|---------|---------|-------|----------|--------------|-------------|
| Resume Understanding | ❌ | ❌ | ❌ | ⚠️ | ✅ |
| STAR Evaluation | ❌ | ⚠️ | ❌ | ❌ | ✅ |
| Gap Analysis | ❌ | ❌ | ❌ | ❌ | ✅ |
| Offline Practice | ❌ | ❌ | ✅ | ⚠️ | ✅ |
| Free/Open-Source | ⚠️ | ❌ | ⚠️ | ❌ | ✅ |
| Multi-Language | ❌ | ❌ | ❌ | ❌ | ✅ |
| Progress Analytics | ❌ | ⚠️ | ✅ | ⚠️ | ✅ |
| Price | Free | $50+/session | Free/Paid | $9-99/mo | **FREE** |

---

## 📈 Performance

- ⚡ Resume Analysis: < 10 seconds
- ⚡ Question Generation: < 5 seconds
- ⚡ Answer Evaluation: < 15 seconds
- 📊 UI Response: < 1 second
- 💾 Database Queries: < 100ms

---

## 🎯 Roadmap

### Version 1.0 (Current) ✅
- Resume analysis
- Gap analysis
- HR/Technical/Mock interviews
- STAR method evaluation
- Progress tracking

### Version 2.0 (Planned)
- 🔄 Live video interview simulation
- 🔄 Computer vision for body language
- 🔄 Pronunciation feedback
- 🔄 Company-specific questions
- 🔄 Mobile app

### Version 3.0 (Future)
- 🔄 Multimodal AI (video + speech + text)
- 🔄 Predictive success scoring
- 🔄 Recruiter integration
- 🔄 B2B licensing
- 🔄 20+ language support

---



<div align="center">

### Made with ❤️ by Bhavya Dhingra


</div>

---

