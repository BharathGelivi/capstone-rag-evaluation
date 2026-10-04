"""
Claim Representation Framework (Phase 2.1)

Defines the foundational data structures for the diagnostic framework by defining 
how extracted claims will be represented, completely separated from evaluation metrics.
"""

import os
import json
import hashlib
from enum import Enum
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

class ClaimType(str, Enum):
    """Categories of claims."""
    ENTITY = "ENTITY"
    ATTRIBUTE = "ATTRIBUTE"
    RELATIONSHIP = "RELATIONSHIP"
    NUMERICAL = "NUMERICAL"
    TEMPORAL = "TEMPORAL"
    CONDITIONAL = "CONDITIONAL"
    PROCEDURAL = "PROCEDURAL"

class VerificationComplexity(str, Enum):
    """Levels of complexity for verification."""
    VCL1 = "VCL1"
    VCL2 = "VCL2"
    VCL3 = "VCL3"
    VCL4 = "VCL4"
    VCL5 = "VCL5"

@dataclass
class Claim:
    """
    A factual record of an atomic claim extracted from the LLM's generated answer.
    Does NOT contain evaluation fields (e.g. confidence, support score).
    """
    # Required fields
    claim_id: str
    trace_id: str
    claim_text: str
    sentence_id: str
    claim_index: int
    character_start: int
    character_end: int

    # Optional fields (populated by downstream modules)
    normalized_text: Optional[str] = None
    claim_type: Optional[ClaimType] = None
    verification_complexity: Optional[VerificationComplexity] = None
    # SHA-256 hex digest of claim_text (lowercase, stripped) for deduplication.
    # Populated by ClaimFactory.create_claim() at construction time.
    claim_hash: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClaimSet:
    """
    A collection of Claim objects belonging to a single RAGTrace.
    """
    trace_id: str
    claims: List[Claim] = field(default_factory=list)
    total_claims: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_claim(self, claim: Claim) -> None:
        """Adds a claim to the set and updates the total count."""
        if claim.trace_id != self.trace_id:
            raise ValueError(f"Claim trace_id '{claim.trace_id}' does not match ClaimSet trace_id '{self.trace_id}'")
        self.claims.append(claim)
        self.total_claims = len(self.claims)

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Retrieves a claim by its ID."""
        for claim in self.claims:
            if claim.claim_id == claim_id:
                return claim
        return None

    @classmethod
    def from_candidates(cls, candidates) -> 'ClaimSet':
        result = cls(trace_id=candidates.trace_id, metadata=dict(candidates.metadata))
        for candidate in candidates.candidate_claims:
            result.add_claim(Claim(
                claim_id=candidate.candidate_id, trace_id=candidate.trace_id,
                claim_text=candidate.claim_text, sentence_id=candidate.sentence_id,
                claim_index=candidate.claim_index, character_start=candidate.character_start,
                character_end=candidate.character_end, metadata=dict(candidate.metadata)))
        return result

    def to_json(self, file_path: str) -> None:
        """Serializes the ClaimSet to a JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Custom serialization for Enums
        def default_serializer(obj):
            if isinstance(obj, Enum):
                return obj.value
            raise TypeError(f"Type {type(obj)} not serializable")
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=4, default=default_serializer)

    @classmethod
    def from_json(cls, file_path: str) -> 'ClaimSet':
        """Deserializes a ClaimSet from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Reconstruct Claim objects, handling Enums correctly
        reconstructed_claims = []
        for claim_data in data.get('claims', []):
            if claim_data.get('claim_type'):
                claim_data['claim_type'] = ClaimType(claim_data['claim_type'])
            if claim_data.get('verification_complexity'):
                claim_data['verification_complexity'] = VerificationComplexity(claim_data['verification_complexity'])
                
            reconstructed_claims.append(Claim(**claim_data))
            
        data['claims'] = reconstructed_claims
        return cls(**data)


class ClaimFactory:
    """Standardizes creation of Claim objects with deterministic IDs and hashes."""

    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self._counter = 0

    def create_claim(
        self,
        claim_text: str,
        sentence_id: str,
        character_start: int,
        character_end: int,
        **kwargs,
    ) -> Claim:
        """Create a new Claim with a deterministic ID (TRACE_C001) and SHA-256 hash.

        The claim_hash enables O(1) deduplication across ClaimSets.
        """
        self._counter += 1
        claim_id = f"{self.trace_id}_C{self._counter:03d}"
        claim_hash = hashlib.sha256(claim_text.lower().strip().encode()).hexdigest()

        return Claim(
            claim_id=claim_id,
            trace_id=self.trace_id,
            claim_text=claim_text,
            sentence_id=sentence_id,
            claim_index=self._counter - 1,
            character_start=character_start,
            character_end=character_end,
            claim_hash=claim_hash,
            **kwargs,
        )
