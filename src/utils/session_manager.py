"""
Session Manager — Local Persistence (SQLite/JSON)
Stores user preferences, history, analytics locally.
"""

import json
import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import streamlit as st


class SessionManager:
    """Manages local session persistence."""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Use ~/.raphaid for cross-platform
            home = Path.home()
            data_dir = home / ".raphaid"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.data_dir / "sessions.db"
        self.analytics_path = self.data_dir / "analytics.json"
        self.preferences_path = self.data_dir / "preferences.json"

        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    start_time TEXT,
                    end_time TEXT,
                    module TEXT,
                    analysis_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp TEXT,
                    module TEXT,
                    submodule TEXT,
                    image_name TEXT,
                    result_json TEXT,
                    verified BOOLEAN DEFAULT FALSE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    detection_index INTEGER,
                    decision TEXT,  -- 'accept' or 'reject'
                    timestamp TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    report_type TEXT,  -- 'pdf', 'csv'
                    file_path TEXT,
                    timestamp TEXT
                )
            """)

    def start_session(self, module: str = "dashboard") -> str:
        """Start a new session."""
        session_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, start_time, module) VALUES (?, ?, ?)",
                (session_id, datetime.now().isoformat(), module)
            )
        return session_id

    def end_session(self, session_id: str) -> None:
        """End a session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sessions SET end_time = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id)
            )

    def log_analysis(
        self,
        session_id: str,
        module: str,
        submodule: str,
        image_name: str,
        result: Dict,
        verified: bool = False,
    ) -> int:
        """Log an analysis result."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO analyses (session_id, timestamp, module, submodule, image_name, result_json, verified)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, datetime.now().isoformat(), module, submodule, image_name, json.dumps(result), verified)
            )
            return cursor.lastrowid

    def log_verification(self, analysis_id: int, detection_index: int, decision: str) -> None:
        """Log a verification decision."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO verifications (analysis_id, detection_index, decision, timestamp) VALUES (?, ?, ?, ?)",
                (analysis_id, detection_index, decision, datetime.now().isoformat())
            )

    def log_report(self, analysis_id: int, report_type: str, file_path: str) -> None:
        """Log a generated report."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO reports (analysis_id, report_type, file_path, timestamp) VALUES (?, ?, ?, ?)",
                (analysis_id, report_type, file_path, datetime.now().isoformat())
            )

    def get_history(self, session_id: str = None, limit: int = 50) -> List[Dict]:
        """Get analysis history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if session_id:
                cursor = conn.execute(
                    "SELECT * FROM analyses WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (session_id, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM analyses ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_analytics(self) -> Dict:
        """Get aggregate analytics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Total analyses per module
            cursor = conn.execute("""
                SELECT module, submodule, COUNT(*) as count
                FROM analyses
                GROUP BY module, submodule
            """)
            by_module = {}
            for row in cursor.fetchall():
                key = f"{row['module']}_{row['submodule']}"
                by_module[key] = row['count']

            # Total verifications
            cursor = conn.execute("SELECT COUNT(*) as count FROM verifications")
            verifications = cursor.fetchone()['count']

            # Total reports
            cursor = conn.execute("SELECT report_type, COUNT(*) as count FROM reports GROUP BY report_type")
            reports = {row['report_type']: row['count'] for row in cursor.fetchall()}

            return {
                "by_module": by_module,
                "total_verifications": verifications,
                "reports": reports,
            }

    def save_preferences(self, preferences: Dict) -> None:
        """Save user preferences."""
        with open(self.preferences_path, 'w') as f:
            json.dump(preferences, f, indent=2)

    def load_preferences(self) -> Dict:
        """Load user preferences."""
        if self.preferences_path.exists():
            with open(self.preferences_path, 'r') as f:
                return json.load(f)
        return {}

    def save_analytics_json(self, analytics: Dict) -> None:
        """Save analytics to JSON file for quick access."""
        with open(self.analytics_path, 'w') as f:
            json.dump(analytics, f, indent=2)

    def load_analytics_json(self) -> Dict:
        """Load analytics from JSON file."""
        if self.analytics_path.exists():
            with open(self.analytics_path, 'r') as f:
                return json.load(f)
        return {}


@st.cache_resource
def get_session_manager(data_dir: str = None) -> SessionManager:
    """Get cached SessionManager instance."""
    return SessionManager(data_dir)