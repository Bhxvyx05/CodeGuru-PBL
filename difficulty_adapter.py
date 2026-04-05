from typing import Literal
from enum import Enum

class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class DifficultyAdapter:
    """
    Manages adaptive difficulty based on user performance.
    Uses Streamlit session state to persist across interactions.
    """
    
    def __init__(self):
        self.initial_score = 50  # Start at medium difficulty
    
    def get_difficulty(self, current_score: int) -> str:
        """
        Get difficulty level based on current performance score.
        
        Args:
            current_score: User's average score (0-100)
            
        Returns:
            Difficulty level: "easy", "medium", or "hard"
        """
        if current_score >= 75:
            return DifficultyLevel.HARD.value
        elif current_score >= 50:
            return DifficultyLevel.MEDIUM.value
        else:
            return DifficultyLevel.EASY.value
    
    def get_difficulty_description(self, difficulty: str) -> str:
        """Get description for difficulty level."""
        descriptions = {
            "easy": "Easy - Basic concepts and foundational questions",
            "medium": "Medium - Balanced mix of concepts and real-world scenarios",
            "hard": "Hard - Complex scenarios, edge cases, and advanced topics"
        }
        return descriptions.get(difficulty, "Medium")
    
    def get_next_difficulty_recommendation(self, current_score: int) -> str:
        """Get recommendation for next difficulty."""
        if current_score >= 80:
            return "🚀 Excellent! Ready for harder questions"
        elif current_score >= 65:
            return "📈 Good progress! Slight difficulty increase"
        elif current_score >= 50:
            return "⚖️ Maintaining medium difficulty"
        else:
            return "📉 Take easier questions to build confidence"
    
    def update_score(self, old_score: int, new_answer_score: int) -> int:
        """
        Update running average score.
        
        Args:
            old_score: Previous average score
            new_answer_score: Score for latest answer
            
        Returns:
            New average score
        """
        return round((old_score + new_answer_score) / 2, 1)
    
    def build_difficulty_prompt_instruction(self, difficulty: str, skill: str) -> str:
        """
        Build prompt instruction for generating questions at specific difficulty.
        """
        if difficulty == "easy":
            return f"""
Generate an EASY interview question about {skill}.
Focus on: Basic definitions, fundamental concepts, simple scenarios.
The candidate should be able to answer in 1-2 minutes.
"""
        elif difficulty == "medium":
            return f"""
Generate a MEDIUM difficulty interview question about {skill}.
Focus on: Practical applications, real-world scenarios, problem-solving.
Requires some experience but not advanced knowledge.
"""
        else:  # hard
            return f"""
Generate a HARD interview question about {skill}.
Focus on: Complex scenarios, edge cases, architectural decisions, trade-offs.
Requires deep knowledge and critical thinking.
Expect a 3-5 minute answer.
"""

def initialize_difficulty_adapter():
    """Initialize difficulty adapter for Streamlit session."""
    return DifficultyAdapter()