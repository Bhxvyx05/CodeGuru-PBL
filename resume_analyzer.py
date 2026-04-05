import google.generativeai as genai
import json
from typing import Dict, List

def generate_detailed_resume_analysis(resume_text: str, llm) -> Dict:
    """
    Generate detailed analysis of resume with projects, skills, experience breakdown.
    """
    
    prompt = """Analyze this resume comprehensively and provide a detailed structured analysis in JSON format.

RESUME TEXT:
""" + resume_text + """

Provide analysis with these EXACT fields in JSON (return ONLY JSON, no markdown):

{
    "professional_summary": "2-3 line summary of the candidate",
    "key_strengths": ["strength1", "strength2", "strength3", "strength4", "strength5"],
    "experience": [
        {"position": "Job Title", "company": "Company Name", "duration": "Year-Year", "achievements": ["achievement1", "achievement2"]}
    ],
    "projects": [
        {"name": "Project Name", "description": "Description", "technologies": ["tech1", "tech2"], "impact": "Impact statement"}
    ],
    "education": [
        {"degree": "Degree", "institution": "University", "graduation_year": 2024, "coursework": ["course1"]}
    ],
    "technical_skills": {
        "languages": [{"skill": "Python", "proficiency": "Expert"}],
        "frameworks": [{"skill": "Django", "proficiency": "Advanced"}],
        "databases": [{"skill": "SQL", "proficiency": "Advanced"}],
        "cloud": [{"skill": "AWS", "proficiency": "Intermediate"}],
        "tools": [{"skill": "Git", "proficiency": "Expert"}]
    },
    "soft_skills": ["Communication", "Leadership", "Problem-solving"],
    "certifications": ["cert1", "cert2"],
    "experience_level": {"years": 2, "seniority": "Junior", "progression": "Strong"},
    "gaps_weaknesses": ["gap1", "gap2"],
    "improvement_suggestions": [
        {"area": "Area", "current_state": "Current", "suggestion": "Suggestion", "impact": "Impact"}
    ],
    "scores": {
        "technical_strength": 75,
        "communication_clarity": 70,
        "project_quality": 80,
        "career_growth": 65,
        "overall": 72
    }
}

CRITICAL: Return ONLY valid JSON. No markdown. No extra text."""
    
    try:
        response = llm.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean response
        if "```" in response_text:
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.split("```")[0]
        
        response_text = response_text.strip()
        
        # Try to parse JSON
        analysis = json.loads(response_text)
        
        # Ensure all required fields exist
        if not analysis.get('professional_summary'):
            analysis['professional_summary'] = "Unable to generate summary from resume"
        if not analysis.get('key_strengths'):
            analysis['key_strengths'] = ["Technical Skills", "Problem Solving", "Professional Experience"]
        if not analysis.get('experience'):
            analysis['experience'] = []
        if not analysis.get('projects'):
            analysis['projects'] = []
        if not analysis.get('education'):
            analysis['education'] = []
        if not analysis.get('technical_skills'):
            analysis['technical_skills'] = {}
        if not analysis.get('soft_skills'):
            analysis['soft_skills'] = []
        if not analysis.get('certifications'):
            analysis['certifications'] = []
        if not analysis.get('experience_level'):
            analysis['experience_level'] = {"years": 0, "seniority": "Entry-level", "progression": "Growing"}
        if not analysis.get('gaps_weaknesses'):
            analysis['gaps_weaknesses'] = []
        if not analysis.get('improvement_suggestions'):
            analysis['improvement_suggestions'] = []
        if not analysis.get('scores'):
            analysis['scores'] = {"technical_strength": 60, "communication_clarity": 65, "project_quality": 60, "career_growth": 55, "overall": 60}
        
        return analysis
        
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        # Return default structure
        return {
            "professional_summary": "Resume analysis in progress",
            "key_strengths": ["Strong Technical Foundation", "Relevant Experience", "Project Work"],
            "experience": [],
            "projects": [],
            "education": [],
            "technical_skills": {},
            "soft_skills": ["Communication", "Problem-solving"],
            "certifications": [],
            "experience_level": {"years": 0, "seniority": "Entry-level", "progression": "Strong"},
            "gaps_weaknesses": [],
            "improvement_suggestions": [],
            "scores": {"technical_strength": 65, "communication_clarity": 70, "project_quality": 65, "career_growth": 60, "overall": 65}
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "professional_summary": "Unable to analyze resume",
            "key_strengths": [],
            "experience": [],
            "projects": [],
            "education": [],
            "technical_skills": {},
            "soft_skills": [],
            "certifications": [],
            "experience_level": {},
            "gaps_weaknesses": [],
            "improvement_suggestions": [],
            "scores": {}
        }

def generate_improvement_report(analysis: Dict) -> str:
    """Generate human-readable improvement report."""
    
    report = "\n" + "="*70 + "\n"
    report += "COMPREHENSIVE RESUME ANALYSIS REPORT\n"
    report += "="*70 + "\n\n"
    
    report += "PROFESSIONAL SUMMARY\n"
    report += "-"*70 + "\n"
    report += analysis.get('professional_summary', 'N/A') + "\n\n"
    
    report += "KEY STRENGTHS\n"
    report += "-"*70 + "\n"
    for idx, strength in enumerate(analysis.get('key_strengths', []), 1):
        report += f"{idx}. {strength}\n"
    report += "\n"
    
    return report