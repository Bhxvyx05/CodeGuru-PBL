import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

class SessionDatabase:
    """
    Manages SQLite database for session history and progress tracking.
    """
    
    DB_FILE = "codeguru_sessions.db"
    
    def __init__(self):
        self.init_database()
    
    def init_database(self):
        """Initialize database tables if they don't exist."""
        conn = sqlite3.connect(self.DB_FILE)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_date TEXT NOT NULL,
                interview_mode TEXT,
                skill_practiced TEXT,
                questions_count INTEGER DEFAULT 0,
                average_score REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Answers table - stores each answer evaluation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT,
                difficulty TEXT,
                relevance_score INTEGER,
                star_score INTEGER,
                accuracy_score INTEGER,
                communication_score INTEGER,
                overall_score REAL,
                feedback TEXT,
                answered_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_session(self, user_id: str, interview_mode: str, skill: str = "") -> int:
        """
        Create a new practice session.
        
        Returns:
            Session ID
        """
        conn = sqlite3.connect(self.DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sessions (user_id, session_date, interview_mode, skill_practiced)
            VALUES (?, ?, ?, ?)
        """, (user_id, datetime.now().strftime("%Y-%m-%d"), interview_mode, skill))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def save_answer(self, session_id: int, question: str, answer: str, 
                   difficulty: str, scores: Dict) -> int:
        """
        Save answer evaluation to database.
        
        Returns:
            Answer ID
        """
        conn = sqlite3.connect(self.DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO answers 
            (session_id, question, answer, difficulty, 
             relevance_score, star_score, accuracy_score, communication_score, overall_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            question,
            answer,
            difficulty,
            scores.get("relevance_score", 0),
            scores.get("star_score", 0),
            scores.get("accuracy_score", 0),
            scores.get("communication_score", 0),
            scores.get("overall_score", 0)
        ))
        
        answer_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return answer_id
    
    def get_user_history(self, user_id: str) -> pd.DataFrame:
        """Get all sessions for a user."""
        conn = sqlite3.connect(self.DB_FILE)
        
        query = """
            SELECT id, session_date, interview_mode, skill_practiced, 
                   questions_count, average_score, created_at
            FROM sessions
            WHERE user_id = ?
            ORDER BY created_at DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(user_id,))
        conn.close()
        
        return df
    
    def get_session_details(self, session_id: int) -> pd.DataFrame:
        """Get all answers for a specific session."""
        conn = sqlite3.connect(self.DB_FILE)
        
        query = """
            SELECT question, difficulty, overall_score, relevance_score, 
                   star_score, accuracy_score, communication_score, answered_at
            FROM answers
            WHERE session_id = ?
            ORDER BY answered_at
        """
        
        df = pd.read_sql_query(query, conn, params=(session_id,))
        conn.close()
        
        return df
    
    def get_progress_metrics(self, user_id: str) -> Dict:
        """Get overall progress metrics for a user."""
        conn = sqlite3.connect(self.DB_FILE)
        cursor = conn.cursor()
        
        # Total sessions
        cursor.execute("SELECT COUNT(*) FROM sessions WHERE user_id = ?", (user_id,))
        total_sessions = cursor.fetchone()[0]
        
        # Total answers
        cursor.execute("""
            SELECT COUNT(*) FROM answers a
            JOIN sessions s ON a.session_id = s.id
            WHERE s.user_id = ?
        """, (user_id,))
        total_answers = cursor.fetchone()[0]
        
        # Average score
        cursor.execute("""
            SELECT AVG(overall_score) FROM answers a
            JOIN sessions s ON a.session_id = s.id
            WHERE s.user_id = ?
        """, (user_id,))
        avg_score = cursor.fetchone()[0] or 0
        
        # Score trend (last 5 sessions)
        cursor.execute("""
            SELECT s.session_date, AVG(a.overall_score) as avg_score
            FROM sessions s
            LEFT JOIN answers a ON s.id = a.session_id
            WHERE s.user_id = ?
            GROUP BY s.id
            ORDER BY s.created_at DESC
            LIMIT 5
        """, (user_id,))
        
        trend = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_sessions": total_sessions,
            "total_answers": total_answers,
            "average_score": round(avg_score, 1),
            "recent_trend": trend
        }