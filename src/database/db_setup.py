"""Database setup for video analysis system"""
import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_database():
    """Initialize the database and create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute('PRAGMA foreign_keys = ON')

    # Create analysis_run table - stores each analysis run
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_run (
        run_id VARCHAR(36) PRIMARY KEY,
        case_id VARCHAR(255) NOT NULL,
        video_path TEXT NOT NULL,
        label VARCHAR(20),
        source VARCHAR(50),
        config JSON,
        analysts JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Create agent_result table - stores results from each agent
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS agent_result (
        id VARCHAR(36) PRIMARY KEY,
        run_id VARCHAR(36) NOT NULL,
        agent VARCHAR(50) NOT NULL,
        status VARCHAR(20) NOT NULL,
        score_fake REAL,
        confidence REAL,
        error TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (run_id) REFERENCES analysis_run(run_id) ON DELETE CASCADE
    )
    ''')

    # Create evidence table - stores individual evidence items from agents
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS evidence (
        id VARCHAR(36) PRIMARY KEY,
        agent_result_id VARCHAR(36) NOT NULL,
        type VARCHAR(50) NOT NULL,
        detail TEXT NOT NULL,
        score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (agent_result_id) REFERENCES agent_result(id) ON DELETE CASCADE
    )
    ''')

    # Create verdict table - stores final verdict from judge
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS verdict (
        id VARCHAR(36) PRIMARY KEY,
        run_id VARCHAR(36) NOT NULL UNIQUE,
        label VARCHAR(20) NOT NULL,
        score_fake REAL,
        confidence REAL,
        rationale TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (run_id) REFERENCES analysis_run(run_id) ON DELETE CASCADE
    )
    ''')

    # Create verdict_evidence table - stores evidence items associated with verdict
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS verdict_evidence (
        id VARCHAR(36) PRIMARY KEY,
        verdict_id VARCHAR(36) NOT NULL,
        type VARCHAR(50) NOT NULL,
        detail TEXT NOT NULL,
        score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (verdict_id) REFERENCES verdict(id) ON DELETE CASCADE
    )
    ''')

    # Create indices for better query performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_run_case_id ON analysis_run(case_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_run_label ON analysis_run(label)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_run_created_at ON analysis_run(created_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_result_run_id ON agent_result(run_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_result_agent ON agent_result(agent)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_result_status ON agent_result(status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_evidence_agent_result_id ON evidence(agent_result_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_evidence_type ON evidence(type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_verdict_run_id ON verdict(run_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_verdict_label ON verdict(label)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_verdict_evidence_verdict_id ON verdict_evidence(verdict_id)')

    conn.commit()
    conn.close()
    print(f"Database initialized: {DB_PATH}")


if __name__ == "__main__":
    init_database()
