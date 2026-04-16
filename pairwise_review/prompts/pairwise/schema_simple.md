Return ONLY valid JSON with exactly these keys:
{
  "overall_winner": "A" | "B" | "Tie",
  "confidence": float,
  "rationale": "one short sentence"
}

Confidence must be between 0.0 and 1.0 and should reflect certainty in the overall comparative decision.
