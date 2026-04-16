Return ONLY valid JSON with exactly these keys:
{
  "overall_winner": "A" | "B" | "Tie",
  "category_winners": {
    "soundness": "A" | "B" | "Tie",
    "presentation": "A" | "B" | "Tie",
    "contribution": "A" | "B" | "Tie"
  },
  "confidence": float,
  "rationale_bullets": ["short bullet", "short bullet"]
}

Confidence must be between 0.0 and 1.0 and should reflect certainty in the overall comparative decision.
Use the ICLR-style paper attributes for category winners: soundness, presentation, and contribution.
Do not emit a winner for reviewer confidence; that is a judge property, not a paper property.
