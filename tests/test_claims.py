import os
import tempfile
import unittest

from src.claims import Claim, ClaimSet, ClaimFactory, ClaimType, VerificationComplexity


class TestClaims(unittest.TestCase):
    def test_claim_factory_creates_deterministic_ids(self):
        factory = ClaimFactory(trace_id="TRACE1")
        c1 = factory.create_claim("Claim one.", "S1", 0, 10)
        c2 = factory.create_claim("Claim two.", "S2", 10, 20)

        self.assertEqual(c1.claim_id, "TRACE1_C001")
        self.assertEqual(c2.claim_id, "TRACE1_C002")
        self.assertEqual(c1.claim_index, 0)
        self.assertEqual(c2.claim_index, 1)

    def test_claim_set_add_and_count(self):
        claim_set = ClaimSet(trace_id="TRACE1")
        factory = ClaimFactory(trace_id="TRACE1")
        claim_set.add_claim(factory.create_claim("Claim.", "S1", 0, 5))

        self.assertEqual(claim_set.total_claims, 1)
        self.assertEqual(claim_set.get_claim("TRACE1_C001").claim_text, "Claim.")
        self.assertIsNone(claim_set.get_claim("MISSING"))

    def test_add_claim_rejects_mismatched_trace_id(self):
        claim_set = ClaimSet(trace_id="TRACE1")
        other_trace_claim = Claim(
            claim_id="OTHER_C001",
            trace_id="OTHER_TRACE",
            claim_text="Claim.",
            sentence_id="S1",
            claim_index=0,
            character_start=0,
            character_end=5,
        )
        with self.assertRaises(ValueError):
            claim_set.add_claim(other_trace_claim)

    def test_save_and_load_roundtrip_equality(self):
        claim_set = ClaimSet(trace_id="TRACE1")
        factory = ClaimFactory(trace_id="TRACE1")
        claim = factory.create_claim(
            "The punishment for murder is death or life imprisonment.",
            "S001",
            0,
            58,
            claim_type=ClaimType.PROCEDURAL,
            verification_complexity=VerificationComplexity.VCL1,
            metadata={"confidence": "High"},
        )
        claim_set.add_claim(claim)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "claim_set.json")
            claim_set.to_json(path)
            reloaded = ClaimSet.from_json(path)

            self.assertEqual(claim_set, reloaded)
            self.assertEqual(reloaded.claims[0].claim_type, ClaimType.PROCEDURAL)
            self.assertEqual(reloaded.claims[0].verification_complexity, VerificationComplexity.VCL1)


if __name__ == "__main__":
    unittest.main()
