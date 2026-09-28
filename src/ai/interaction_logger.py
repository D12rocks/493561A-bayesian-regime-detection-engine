"""
AI Interaction Audit Logger.

Logs every Copilot and Scenario Lab interaction with:
  - timestamp (UTC)
  - interaction_type: COPILOT | SCENARIO_LAB
  - date (analysis date used)
  - provider: provider class name
  - model_name: model identifier (e.g. llama3.2, MockProvider)
  - user_query
  - evidence_bundle_hash: SHA-256[:16] of evidence bundle JSON
  - response_hash: SHA-256[:16] of response text
  - evidence_source_statuses: {tool_name: VERIFIED|UNAVAILABLE|...}
  - grounding_trusted: bool
  - violations: list of grounding violations
  - scenario_id (optional, for Scenario Lab)

CONFIDENTIALITY: Logs stay on local disk only. Never transmitted externally.
Raw evidence values are NOT stored; only the cryptographic hash provides linkage.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOG_DIR = Path("logs/ai_interactions")


def _sha256_preview(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _bundle_hash(bundle: Optional[Dict[str, Any]]) -> str:
    if not bundle:
        return "NO_BUNDLE"
    try:
        return _sha256_preview(json.dumps(bundle, sort_keys=True, default=str))
    except Exception:
        return "HASH_ERROR"


class InteractionLogger:
    """Append-only, on-disk interaction audit log."""

    def __init__(self, log_dir: Path = _LOG_DIR) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_dir / f"ai_log_{datetime.now().strftime('%Y%m%d')}.jsonl"

    def log(
        self,
        interaction_type: str,
        query: str,
        response: str,
        provider_name: str,
        model_name: str = "unknown",
        evidence_bundle: Optional[Dict[str, Any]] = None,
        evidence_source_statuses: Optional[Dict[str, str]] = None,
        grounding_trusted: bool = True,
        violations: Optional[List[str]] = None,
        analysis_date: Optional[str] = None,
        scenario_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a single interaction record to the daily JSONL log."""
        record: Dict[str, Any] = {
            "date": analysis_date or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "interaction_type": interaction_type,
            "provider": provider_name,
            "model_name": model_name,
            "model_version": model_name,
            "user_query": query,
            "query": query,
            # Hashes for linkage — no raw data stored
            "evidence_bundle_hash": _bundle_hash(evidence_bundle),
            "response_hash": _sha256_preview(response),
            # Audit fields
            "evidence_source_statuses": evidence_source_statuses or {},
            "grounding_trusted": grounding_trusted,
            "violations": violations or [],
            # Optional
            "scenario_id": scenario_id,
            "extra": extra or {},
        }
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning("Failed to write AI interaction log: %s", exc)

    def get_session_count(self) -> int:
        """Return number of interactions logged today."""
        if not self._log_file.exists():
            return 0
        with open(self._log_file, encoding="utf-8") as f:
            return sum(1 for _ in f)

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the N most recent log records (for admin review)."""
        if not self._log_file.exists():
            return []
        records: List[Dict[str, Any]] = []
        with open(self._log_file, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
        return records[-n:]
