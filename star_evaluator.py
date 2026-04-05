import json
import google.generativeai as genai
from typing import Dict

def build_star_evaluation_prompt(question: str, answer: str, resume_context: str) -> str:
    """
    Build prompt for STAR-based answer evaluation.
    """
    return f"""
You are an expert interview coach evaluating a candidate's answer using the STAR method.

RESUME CONTEXT:
{resume_context}

QUESTION ASKED:
{question}

CANDIDATE'S ANSWER:
{answer}

Evaluate this answer on FOUR dimensions (each out of 100). Be strict but fair.

SCORING CRITERIA:

1. RELEVANCE (0-100): Does the answer use their actual resume experience? No made-up stories?
2. STAR ALIGNMENT (0-100): Is there clear Situation → Task → Action → Result structure?
3. FACTUAL ACCURACY (0-100): No contradictions with resume? Believable claims?
4. COMMUNICATION (0-100): Clear, concise, confident delivery? Easy to follow?

Return ONLY valid JSON (no markdown, no code blocks):
{{
    "relevance_score": <number>,
    "star_score": <number>,
    "accuracy_score": <number>,
    "communication_score": <number>,
    "overall_score": <number>,
    "situation_feedback": "<one sentence>",
    "task_feedback": "<one sentence>",
    "action_feedback": "<one sentence>",
    "result_feedback": "<one sentence>",
    "top_strength": "<one strength>",
    "improvement_area": "<one area to improve>",
    "interview_tip": "<actionable tip for next answer>"
}}
"""

def evaluate_answer_star(question: str, answer: str, resume_context: str, llm) -> Dict:
    """
    Evaluate answer using STAR method with Gemini.
    
    Args:
        question: Interview question asked
        answer: Candidate's answer
        resume_context: Relevant resume information
        llm: Gemini LLM model instance
        
    Returns:
        Dictionary with evaluation scores and feedback
    """
    prompt = build_star_evaluation_prompt(question, answer, resume_context)
    
    try:
        response = llm.generate_content(prompt)
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        evaluation = json.loads(response_text)
        
        # Calculate overall score as average
        evaluation["overall_score"] = round(
            (evaluation["relevance_score"] + evaluation["star_score"] + 
             evaluation["accuracy_score"] + evaluation["communication_score"]) / 4, 1
        )
        
        return evaluation
        
    except json.JSONDecodeError:
        return {
            "relevance_score": 0,
            "star_score": 0,
            "accuracy_score": 0,
            "communication_score": 0,
            "overall_score": 0,
            "situation_feedback": "Error processing evaluation",
            "task_feedback": "Please try again",
            "action_feedback": "N/A",
            "result_feedback": "N/A",
            "top_strength": "Unable to evaluate",
            "improvement_area": "Unable to evaluate",
            "interview_tip": "Please rephrase your answer more clearly"
        }

def get_evaluation_color(score: int) -> str:
    """Get color coding for scores."""
    if score >= 80:
        return "🟢"  # Green
    elif score >= 60:
        return "🟡"  # Yellow
    else:
        return "🔴"  # Red