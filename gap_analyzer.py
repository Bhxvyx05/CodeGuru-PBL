from typing import Dict, Set, List, Tuple
from skill_extractor import extract_skills, SKILL_DATABASE

def extract_jd_skills(jd_text: str) -> Set[str]:
    """
    Extract skills mentioned in job description.
    
    Args:
        jd_text: Text of job description
        
    Returns:
        Set of skill names found in JD
    """
    jd_lower = jd_text.lower()
    found_skills = set()
    
    for skill, keywords in SKILL_DATABASE.items():
        for keyword in keywords:
            if keyword in jd_lower:
                found_skills.add(skill)
                break
    
    return found_skills

def analyze_gap(resume_skills: Dict[str, int], jd_text: str) -> Dict:
    """
    Analyze gap between resume and job description.
    
    Args:
        resume_skills: Skills extracted from resume (with scores)
        jd_text: Job description text
        
    Returns:
        Dictionary with matched, missing, and gap analysis
    """
    jd_skills = extract_jd_skills(jd_text)
    resume_skill_names = set(resume_skills.keys())
    
    matched = resume_skill_names & jd_skills  # Intersection
    missing = jd_skills - resume_skill_names  # Skills required but missing
    extra = resume_skill_names - jd_skills    # Skills you have but not required
    
    # Calculate match percentage
    match_percentage = (len(matched) / len(jd_skills) * 100) if jd_skills else 0
    
    return {
        "matched_skills": sorted(list(matched)),
        "missing_skills": sorted(list(missing)),
        "extra_skills": sorted(list(extra)),
        "total_jd_skills": len(jd_skills),
        "total_matched": len(matched),
        "match_percentage": round(match_percentage, 1),
        "gap_count": len(missing),
        "jd_skill_names": sorted(list(jd_skills)),
    }

def get_gap_recommendations(gap_analysis: Dict) -> List[str]:
    """
    Generate recommendations based on gap analysis.
    """
    recommendations = []
    
    if gap_analysis["gap_count"] == 0:
        recommendations.append("✅ Perfect match! Your resume covers all JD requirements.")
    elif gap_analysis["gap_count"] <= 2:
        recommendations.append(f"🟡 Minor gaps detected. Focus on: {', '.join(gap_analysis['missing_skills'])}")
    else:
        recommendations.append(f"🔴 Significant gaps ({gap_analysis['gap_count']} skills). Prepare to discuss these in interview.")
    
    if gap_analysis["match_percentage"] >= 80:
        recommendations.append("Your skills are well-aligned with the job.")
    elif gap_analysis["match_percentage"] >= 60:
        recommendations.append("You're moderately aligned - highlight transferable skills.")
    else:
        recommendations.append("Consider upskilling in missing areas before interview.")
    
    return recommendations