import re
from collections import Counter
from typing import Dict, List, Tuple

# Comprehensive skill database
SKILL_DATABASE = {
    # Programming Languages
    "Python": ["python", "py"],
    "Java": ["java"],
    "JavaScript": ["javascript", "js", "node.js", "nodejs"],
    "React": ["react", "react.js"],
    "SQL": ["sql", "mysql", "postgresql", "oracle"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp"],
    
    # Cloud & DevOps
    "AWS": ["aws", "amazon web services"],
    "Docker": ["docker", "containerization"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud"],
    
    # Databases
    "MongoDB": ["mongodb", "mongo", "nosql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "Redis": ["redis"],
    
    # Frameworks
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "Spring": ["spring", "spring boot"],
    
    # ML/AI
    "Machine Learning": ["machine learning", "ml", "scikit-learn"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "NLP": ["nlp", "natural language processing"],
    
    # Other Tools
    "Git": ["git", "github", "gitlab"],
    "Linux": ["linux", "unix"],
    "REST API": ["rest", "api", "restful"],
    "System Design": ["system design", "architecture"],
}

def extract_skills(resume_text: str) -> Dict[str, int]:
    """
    Extract skills from resume and return confidence scores.
    
    Args:
        resume_text: Full text of resume
        
    Returns:
        Dictionary with skill names and confidence scores (0-100)
    """
    text_lower = resume_text.lower()
    found_skills = {}
    
    for skill, keywords in SKILL_DATABASE.items():
        max_count = 0
        
        for keyword in keywords:
            # Count occurrences of each keyword
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
            max_count = max(max_count, count)
        
        if max_count > 0:
            # Calculate confidence: base 40 + increment per mention
            confidence = min(100, 40 + (max_count * 15))
            found_skills[skill] = confidence
    
    return found_skills

def get_top_skills(skills: Dict[str, int], top_n: int = 10) -> List[Tuple[str, int]]:
    """
    Get top N skills sorted by confidence.
    
    Args:
        skills: Dictionary of skills and scores
        top_n: Number of top skills to return
        
    Returns:
        List of (skill, score) tuples sorted by score
    """
    return sorted(skills.items(), key=lambda x: x[1], reverse=True)[:top_n]

def categorize_skills(skills: Dict[str, int]) -> Dict[str, List[Tuple[str, int]]]:
    """
    Organize skills into categories for better visualization.
    """
    categories = {
        "Programming": ["Python", "Java", "JavaScript", "C++", "C#"],
        "Frontend": ["React", "JavaScript"],
        "Backend": ["Django", "Flask", "FastAPI", "Spring"],
        "Cloud & DevOps": ["AWS", "Docker", "Kubernetes", "Azure", "GCP"],
        "Databases": ["MongoDB", "PostgreSQL", "MySQL", "Redis", "SQL"],
        "AI/ML": ["Machine Learning", "TensorFlow", "PyTorch", "NLP"],
        "Tools": ["Git", "Linux", "REST API", "System Design"],
    }
    
    categorized = {cat: [] for cat in categories}
    
    for category, skill_list in categories.items():
        for skill in skill_list:
            if skill in skills:
                categorized[category].append((skill, skills[skill]))
    
    return {k: v for k, v in categorized.items() if v}  # Remove empty categories