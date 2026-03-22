"""Database helper for video analysis system"""
import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from database.db_setup import DB_PATH, init_database
from util.logger import logger
from graph.schema import VideoCase, AgentResult, Verdict, EvidenceItem


class ViagentDB:
    """SQLite database helper for the Viagent video analysis system."""

    _initialized = False

    def __init__(self):
        self.db_path = DB_PATH
        if not ViagentDB._initialized:
            init_database()
            ViagentDB._initialized = True

    def _get_connection(self):
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA journal_mode = WAL')  # Enable WAL mode for better concurrency
        return conn

    def _parse_json(self, json_str: Optional[str]) -> Any:
        """Parse JSON string safely."""
        if not json_str or not json_str.strip():
            return None
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse JSON: '{json_str}' - {exc}")
            return None

    # ==================== Analysis Run Methods ====================

    def save_analysis_run(
        self,
        run_id: str,
        case: VideoCase,
        config: Dict[str, Any],
        analysts: List[str]
    ) -> bool:
        """Save an analysis run record."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO analysis_run (
                    run_id, case_id, video_path, label, source,
                    config, analysts, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                case.case_id,
                case.video_path,
                case.label,
                case.source,
                json.dumps(config),
                json.dumps(analysts),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))

            conn.commit()
            logger.info(f"Saved analysis run: {run_id}")
            return True
        except (sqlite3.Error, ValueError, TypeError) as exc:
            logger.error(f"Error saving analysis run: {exc}")
            return False
        finally:
            if conn:
                conn.close()

    def get_analysis_run(self, run_id: str) -> Optional[Dict]:
        """Get an analysis run by run_id."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('SELECT * FROM analysis_run WHERE run_id = ?', (run_id,))
            row = cursor.fetchone()

            if row:
                return {
                    'run_id': row['run_id'],
                    'case_id': row['case_id'],
                    'video_path': row['video_path'],
                    'label': row['label'],
                    'source': row['source'],
                    'config': self._parse_json(row['config']),
                    'analysts': self._parse_json(row['analysts']),
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                }
            return None
        except (sqlite3.Error, ValueError) as exc:
            logger.error(f"Error getting analysis run: {exc}")
            return None
        finally:
            if conn:
                conn.close()

    def get_analysis_runs_by_case_id(
        self,
        case_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get analysis runs by case_id."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = '''
                SELECT * FROM analysis_run
                WHERE case_id = ?
                ORDER BY created_at DESC
            '''
            params = [case_id]
            if limit:
                query += ' LIMIT ?'
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                {
                    'run_id': row['run_id'],
                    'case_id': row['case_id'],
                    'video_path': row['video_path'],
                    'label': row['label'],
                    'source': row['source'],
                    'config': self._parse_json(row['config']),
                    'analysts': self._parse_json(row['analysts']),
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                }
                for row in rows
            ]
        except (sqlite3.Error, ValueError) as exc:
            logger.error(f"Error getting analysis runs by case_id: {exc}")
            return []
        finally:
            if conn:
                conn.close()

    # ==================== Agent Result Methods ====================

    def save_agent_result(
        self,
        run_id: str,
        agent_result: AgentResult
    ) -> Optional[str]:
        """Save an agent result."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            result_id = str(uuid.uuid4())

            cursor.execute('''
                INSERT INTO agent_result (
                    id, run_id, agent, status, score_fake,
                    confidence, error, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result_id,
                run_id,
                agent_result.agent,
                agent_result.status,
                agent_result.score_fake,
                agent_result.confidence,
                agent_result.error,
                datetime.now().isoformat()
            ))

            # Save evidence items in the same transaction
            for evidence in agent_result.evidence:
                evidence_id = str(uuid.uuid4())
                cursor.execute('''
                    INSERT INTO evidence (
                        id, agent_result_id, type, detail,
                        score, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    evidence_id,
                    result_id,
                    evidence.type,
                    evidence.detail,
                    evidence.score,
                    datetime.now().isoformat()
                ))

            conn.commit()
            logger.info(f"Saved agent result: {result_id} for agent {agent_result.agent}")
            return result_id
        except (sqlite3.Error, ValueError, TypeError) as exc:
            logger.error(f"Error saving agent result: {exc}")
            return None
        finally:
            if conn:
                conn.close()

    def get_agent_results_by_run_id(self, run_id: str) -> List[Dict]:
        """Get all agent results for a run."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM agent_result
                WHERE run_id = ?
                ORDER BY created_at
            ''', (run_id,))

            rows = cursor.fetchall()
            results = []

            for row in rows:
                # Get evidence for this result
                evidence = self.get_evidence_by_agent_result_id(row['id'])

                results.append({
                    'id': row['id'],
                    'run_id': row['run_id'],
                    'agent': row['agent'],
                    'status': row['status'],
                    'score_fake': row['score_fake'],
                    'confidence': row['confidence'],
                    'error': row['error'],
                    'evidence': evidence,
                    'created_at': row['created_at']
                })

            return results
        except (sqlite3.Error, ValueError) as exc:
            logger.error(f"Error getting agent results: {exc}")
            return []
        finally:
            if conn:
                conn.close()

    # ==================== Evidence Methods ====================

    def save_evidence(
        self,
        agent_result_id: str,
        evidence: EvidenceItem
    ) -> Optional[str]:
        """Save an evidence item."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            evidence_id = str(uuid.uuid4())

            cursor.execute('''
                INSERT INTO evidence (
                    id, agent_result_id, type, detail,
                    score, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                evidence_id,
                agent_result_id,
                evidence.type,
                evidence.detail,
                evidence.score,
                datetime.now().isoformat()
            ))

            conn.commit()
            return evidence_id
        except (sqlite3.Error, ValueError, TypeError) as exc:
            logger.error(f"Error saving evidence: {exc}")
            return None
        finally:
            if conn:
                conn.close()

    def get_evidence_by_agent_result_id(self, agent_result_id: str) -> List[Dict]:
        """Get evidence items for an agent result."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM evidence
                WHERE agent_result_id = ?
                ORDER BY created_at
            ''', (agent_result_id,))

            rows = cursor.fetchall()
            return [
                {
                    'id': row['id'],
                    'type': row['type'],
                    'detail': row['detail'],
                    'score': row['score'],
                    'created_at': row['created_at']
                }
                for row in rows
            ]
        except (sqlite3.Error, ValueError) as exc:
            logger.error(f"Error getting evidence: {exc}")
            return []
        finally:
            if conn:
                conn.close()

    # ==================== Verdict Methods ====================

    def save_verdict(self, run_id: str, verdict: Verdict) -> Optional[str]:
        """Save a verdict."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            verdict_id = str(uuid.uuid4())

            cursor.execute('''
                INSERT INTO verdict (
                    id, run_id, label, score_fake,
                    confidence, rationale, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                verdict_id,
                run_id,
                verdict.label,
                verdict.score_fake,
                verdict.confidence,
                verdict.rationale,
                datetime.now().isoformat()
            ))

            # Save verdict evidence in the same transaction
            for evidence in verdict.evidence:
                evidence_id = str(uuid.uuid4())
                cursor.execute('''
                    INSERT INTO verdict_evidence (
                        id, verdict_id, type, detail,
                        score, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    evidence_id,
                    verdict_id,
                    evidence.type,
                    evidence.detail,
                    evidence.score,
                    datetime.now().isoformat()
                ))

            conn.commit()
            logger.info(f"Saved verdict: {verdict_id} for run {run_id}")
            return verdict_id
        except (sqlite3.Error, ValueError, TypeError) as exc:
            logger.error(f"Error saving verdict: {exc}")
            return None
        finally:
            if conn:
                conn.close()

    def get_verdict_by_run_id(self, run_id: str) -> Optional[Dict]:
        """Get verdict for a run."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('SELECT * FROM verdict WHERE run_id = ?', (run_id,))
            row = cursor.fetchone()

            if row:
                # Get verdict evidence
                evidence = self.get_verdict_evidence_by_verdict_id(row['id'])

                return {
                    'id': row['id'],
                    'run_id': row['run_id'],
                    'label': row['label'],
                    'score_fake': row['score_fake'],
                    'confidence': row['confidence'],
                    'rationale': row['rationale'],
                    'evidence': evidence,
                    'created_at': row['created_at']
                }
            return None
        except (sqlite3.Error, ValueError) as exc:
            logger.error(f"Error getting verdict: {exc}")
            return None
        finally:
            if conn:
                conn.close()

    def save_verdict_evidence(
        self,
        verdict_id: str,
        evidence: EvidenceItem
    ) -> Optional[str]:
        """Save verdict evidence item."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            evidence_id = str(uuid.uuid4())

            cursor.execute('''
                INSERT INTO verdict_evidence (
                    id, verdict_id, type, detail,
                    score, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                evidence_id,
                verdict_id,
                evidence.type,
                evidence.detail,
                evidence.score,
                datetime.now().isoformat()
            ))

            conn.commit()
            return evidence_id
        except (sqlite3.Error, ValueError, TypeError) as exc:
            logger.error(f"Error saving verdict evidence: {exc}")
            return None
        finally:
            if conn:
                conn.close()

    def get_verdict_evidence_by_verdict_id(self, verdict_id: str) -> List[Dict]:
        """Get evidence items for a verdict."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM verdict_evidence
                WHERE verdict_id = ?
                ORDER BY created_at
            ''', (verdict_id,))

            rows = cursor.fetchall()
            return [
                {
                    'id': row['id'],
                    'type': row['type'],
                    'detail': row['detail'],
                    'score': row['score'],
                    'created_at': row['created_at']
                }
                for row in rows
            ]
        except (sqlite3.Error, ValueError) as exc:
            logger.error(f"Error getting verdict evidence: {exc}")
            return []
        finally:
            if conn:
                conn.close()

    # ==================== Complete Analysis Save ====================

    def save_complete_analysis(
        self,
        run_id: str,
        case: VideoCase,
        analysis_data: Dict[str, Any]
    ) -> bool:
        """Save a complete analysis run with all results and verdict.
        
        Args:
            run_id: Analysis run ID
            case: VideoCase object
            analysis_data: Dict containing:
                - config: Analysis configuration
                - analysts: List of agent names
                - results: Dict of AgentResult objects
                - verdict: Optional Verdict object
        """
        try:
            config = analysis_data['config']
            analysts = analysis_data['analysts']
            results = analysis_data['results']
            verdict = analysis_data.get('verdict')

            # Save analysis run
            if not self.save_analysis_run(run_id, case, config, analysts):
                return False

            # Save agent results
            for agent_result in results.values():
                self.save_agent_result(run_id, agent_result)

            # Save verdict if available
            if verdict:
                self.save_verdict(run_id, verdict)

            logger.info(f"Saved complete analysis: {run_id}")
            return True
        except (sqlite3.Error, ValueError, TypeError, KeyError) as exc:
            logger.error(f"Error saving complete analysis: {exc}")
            return False

    def get_complete_analysis(self, run_id: str) -> Optional[Dict]:
        """Get a complete analysis run with all results and verdict."""
        try:
            # Get analysis run
            run = self.get_analysis_run(run_id)
            if not run:
                return None

            # Get agent results
            agent_results = self.get_agent_results_by_run_id(run_id)

            # Get verdict
            verdict = self.get_verdict_by_run_id(run_id)

            return {
                'run': run,
                'agent_results': agent_results,
                'verdict': verdict
            }
        except (sqlite3.Error, ValueError) as exc:
            logger.error(f"Error getting complete analysis: {exc}")
            return None
