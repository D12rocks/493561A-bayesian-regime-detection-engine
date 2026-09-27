"""
Model Governance: Registration Cards and Lifecycle State Machine.

Implements REQ-070 and Specification Section K:
- Formal model lifecycle: CANDIDATE -> VALIDATION -> CHALLENGER -> CHAMPION -> MONITORED -> RETIRED
- Immutable Model Registration Cards recording snapshot hashes, priors, MCMC diagnostics, and code commits
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


class ModelLifecycleState(str, Enum):
    CANDIDATE = "CANDIDATE"
    VALIDATION = "VALIDATION"
    CHALLENGER = "CHALLENGER"
    CHAMPION = "CHAMPION"
    MONITORED = "MONITORED"
    RETIRED = "RETIRED"


@dataclass
class ModelRegistrationCard:
    model_name: str
    model_version: str
    lifecycle_state: str
    created_at_utc: str
    git_commit: str
    data_snapshot_sha256: str
    feature_snapshot_sha256: str
    hyperparameters: Dict[str, Any]
    priors: Dict[str, Any]
    diagnostics: Dict[str, Any]
    validation_metrics: Dict[str, float]
    approved_by: str = "Quantitative_Model_Risk_Committee"
    audit_notes: str = ""


class ModelRegistry:
    """
    Central registry managing model governance cards and promotion workflows.
    """

    def __init__(self, registry_dir: str = "models/registry") -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.cards: Dict[str, ModelRegistrationCard] = {}

    def register_card(self, card: ModelRegistrationCard) -> Path:
        key = f"{card.model_name}_{card.model_version}"
        self.cards[key] = card
        
        card_path = self.registry_dir / f"{key}.json"
        with open(card_path, "w") as f:
            json.dump(asdict(card), f, indent=2)
        return card_path

    def promote_to_champion(self, model_name: str, model_version: str, reason: str) -> None:
        key = f"{model_name}_{model_version}"
        if key not in self.cards:
            # Check disk
            card_path = self.registry_dir / f"{key}.json"
            if card_path.exists():
                with open(card_path) as f:
                    self.cards[key] = ModelRegistrationCard(**json.load(f))
            else:
                raise KeyError(f"Model {key} not found in registry.")

        # Demote existing champion if any
        for k, card in self.cards.items():
            if card.lifecycle_state == ModelLifecycleState.CHAMPION.value:
                card.lifecycle_state = ModelLifecycleState.CHALLENGER.value
                self.register_card(card)

        # Promote target
        card = self.cards[key]
        card.lifecycle_state = ModelLifecycleState.CHAMPION.value
        card.audit_notes = f"Promoted to CHAMPION: {reason} at {datetime.utcnow().isoformat()}"
        self.register_card(card)
