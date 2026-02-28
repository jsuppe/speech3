"""
Argument Mapping Analysis for SpeakFit Oratory.
Extracts claim-evidence-impact structure from speech transcripts using LLM.
"""
import json
import logging
import requests
from typing import Optional

logger = logging.getLogger("speechscore.argument")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

EXTRACTION_PROMPT = """Analyze this speech transcript and extract the argument structure.

TRANSCRIPT:
{transcript}

Extract ALL arguments found. For each argument, identify:
1. CLAIM: The main assertion or position being made
2. EVIDENCE: Supporting facts, examples, statistics, or reasoning (if any)
3. IMPACT: The consequence or why it matters (if stated)

Return ONLY valid JSON in this exact format:
{{
  "arguments": [
    {{
      "claim": "The exact claim being made",
      "evidence": ["List of supporting evidence points"],
      "impact": "The stated consequence or importance",
      "strength": "strong|moderate|weak"
    }}
  ],
  "argument_flow": "linear|circular|building|scattered",
  "gaps": ["List of claims without evidence or evidence without claims"],
  "summary": "One sentence summary of the overall argument"
}}

If a field is not present in the speech, use null. Be precise and extract actual quotes when possible."""


def analyze_arguments(transcript: str, timeout: int = 30) -> Optional[dict]:
    """
    Extract argument structure from a transcript using LLM.
    
    Returns:
        - arguments: List of claim-evidence-impact structures
        - argument_flow: How arguments connect (linear, building, etc.)
        - gaps: Identified weaknesses in argumentation
        - summary: Overall argument summary
        - metrics: Computed metrics for scoring
    """
    if not transcript or len(transcript.strip()) < 50:
        return {
            "arguments": [],
            "argument_flow": "none",
            "gaps": ["Transcript too short for argument analysis"],
            "summary": None,
            "metrics": {
                "argument_count": 0,
                "evidence_ratio": 0.0,
                "impact_ratio": 0.0,
                "strength_score": 0,
            }
        }
    
    # Truncate very long transcripts
    truncated = transcript[:4000] if len(transcript) > 4000 else transcript
    
    prompt = EXTRACTION_PROMPT.format(transcript=truncated)
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower for more consistent extraction
                    "num_predict": 1500,
                }
            },
            timeout=timeout
        )
        response.raise_for_status()
        
        result = response.json()
        raw_output = result.get("response", "")
        
        # Extract JSON from response
        parsed = _extract_json(raw_output)
        if not parsed:
            logger.warning("Failed to parse argument analysis JSON")
            return _empty_result("Could not parse LLM response")
        
        # Compute metrics
        arguments = parsed.get("arguments", [])
        metrics = _compute_metrics(arguments)
        parsed["metrics"] = metrics
        
        return parsed
        
    except requests.exceptions.Timeout:
        logger.warning("Argument analysis timed out")
        return _empty_result("Analysis timed out")
    except Exception as e:
        logger.exception(f"Argument analysis failed: {e}")
        return _empty_result(str(e))


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON object from LLM response text."""
    # Try direct parse first
    try:
        return json.loads(text)
    except:
        pass
    
    # Look for JSON block
    import re
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'(\{[\s\S]*\})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                continue
    
    return None


def _compute_metrics(arguments: list) -> dict:
    """Compute scoring metrics from extracted arguments."""
    if not arguments:
        return {
            "argument_count": 0,
            "evidence_ratio": 0.0,
            "impact_ratio": 0.0,
            "strength_score": 0,
        }
    
    count = len(arguments)
    with_evidence = sum(1 for a in arguments if a.get("evidence") and len(a["evidence"]) > 0)
    with_impact = sum(1 for a in arguments if a.get("impact"))
    
    strength_map = {"strong": 3, "moderate": 2, "weak": 1}
    strength_sum = sum(strength_map.get(a.get("strength", "weak"), 1) for a in arguments)
    
    return {
        "argument_count": count,
        "evidence_ratio": round(with_evidence / count, 2) if count > 0 else 0.0,
        "impact_ratio": round(with_impact / count, 2) if count > 0 else 0.0,
        "strength_score": round((strength_sum / (count * 3)) * 100) if count > 0 else 0,
    }


def _empty_result(error: str) -> dict:
    """Return empty result structure with error."""
    return {
        "arguments": [],
        "argument_flow": "none",
        "gaps": [error],
        "summary": None,
        "metrics": {
            "argument_count": 0,
            "evidence_ratio": 0.0,
            "impact_ratio": 0.0,
            "strength_score": 0,
        }
    }


# --- Visualization helpers for the app ---

def format_for_display(analysis: dict) -> dict:
    """
    Format argument analysis for app display.
    Returns structure optimized for Flutter UI.
    """
    if not analysis or not analysis.get("arguments"):
        return {
            "cards": [],
            "flow_diagram": None,
            "summary_card": {
                "title": "No Arguments Detected",
                "subtitle": "Try recording a speech with clear claims and supporting evidence.",
                "score": 0,
            }
        }
    
    arguments = analysis["arguments"]
    metrics = analysis.get("metrics", {})
    
    # Build argument cards for list view
    cards = []
    for i, arg in enumerate(arguments, 1):
        card = {
            "index": i,
            "claim": arg.get("claim", ""),
            "evidence": arg.get("evidence", []),
            "impact": arg.get("impact"),
            "strength": arg.get("strength", "weak"),
            "strength_color": {
                "strong": "#22C55E",  # green
                "moderate": "#F59E0B",  # amber
                "weak": "#EF4444",  # red
            }.get(arg.get("strength", "weak"), "#9CA3AF"),
        }
        cards.append(card)
    
    # Flow diagram data (for visual representation)
    flow_diagram = {
        "type": analysis.get("argument_flow", "linear"),
        "nodes": [
            {"id": i, "label": f"Arg {i}", "strength": arg.get("strength")}
            for i, arg in enumerate(arguments, 1)
        ],
        "description": {
            "linear": "Arguments build sequentially, each leading to the next.",
            "building": "Arguments accumulate toward a climax.",
            "circular": "Arguments reinforce each other in a loop.",
            "scattered": "Arguments are independent, not clearly connected.",
        }.get(analysis.get("argument_flow", "linear"), "")
    }
    
    # Summary card
    summary_card = {
        "title": f"{len(arguments)} Argument{'s' if len(arguments) != 1 else ''} Detected",
        "subtitle": analysis.get("summary") or "Analysis complete",
        "score": metrics.get("strength_score", 0),
        "stats": [
            {"label": "With Evidence", "value": f"{int(metrics.get('evidence_ratio', 0) * 100)}%"},
            {"label": "With Impact", "value": f"{int(metrics.get('impact_ratio', 0) * 100)}%"},
            {"label": "Overall Strength", "value": f"{metrics.get('strength_score', 0)}%"},
        ],
        "gaps": analysis.get("gaps", []),
    }
    
    return {
        "cards": cards,
        "flow_diagram": flow_diagram,
        "summary_card": summary_card,
    }


if __name__ == "__main__":
    # Test with a sample speech
    test_transcript = """
    We must act now on climate change. The science is clear: global temperatures have risen 
    1.1 degrees Celsius since pre-industrial times. The IPCC reports that we have less than 
    a decade to prevent catastrophic warming. If we fail to act, we face rising seas, 
    extreme weather, and mass extinction. 
    
    But there is hope. Renewable energy costs have dropped 90% in the last decade. 
    Countries that invest in green technology are seeing economic growth. Germany created 
    300,000 jobs in solar alone. The transition is not just possible—it's profitable.
    
    We cannot wait for perfect solutions. We cannot let the perfect be the enemy of the good.
    Every day of delay costs lives. The choice is ours: act now, or explain to our children 
    why we didn't.
    """
    
    result = analyze_arguments(test_transcript)
    print(json.dumps(result, indent=2))
    
    display = format_for_display(result)
    print("\n--- Display Format ---")
    print(json.dumps(display, indent=2))
