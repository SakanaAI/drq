"""LLM bridge for Janus interop."""

import asyncio
from pathlib import Path

import sys
sys.path.insert(0, '.')

# Load prompts
PROMPTS_DIR = Path("src/prompts")


def load_prompt(name: str) -> str:
    """Load prompt template from file."""
    path = PROMPTS_DIR / f"{name}.txt"
    if path.exists():
        return path.read_text()
    return ""


SYSTEM_PROMPT = load_prompt("system_prompt_0")
NEW_PROMPT = load_prompt("new_prompt_0")
MUTATE_PROMPT = load_prompt("mutate_prompt_0")


def build_prompt(hints: dict) -> str:
    """Build prompt from constraint hints."""
    constraints = hints.get("constraints", [])

    # Check for mutation
    parent_constraint = next(
        (c for c in constraints if c.get("type") == "parent"), None
    )

    if parent_constraint:
        base = MUTATE_PROMPT or "Mutate this warrior to improve performance."
        return f"{base}\n\nParent ID: {parent_constraint['id']}"

    # Build generation prompt with constraints
    base = NEW_PROMPT or "Generate a new Core War warrior."

    constraint_lines = []
    for c in constraints:
        ctype = c.get("type")
        if ctype == "min_length":
            constraint_lines.append(f"- Minimum length: {c['value']} instructions")
        elif ctype == "required_opcode":
            constraint_lines.append(f"- Must use opcode: {c['value']}")
        elif ctype == "target_bc":
            constraint_lines.append(f"- Target behavior: TSP bin {c['tsp']}, MC bin {c['mc']}")

    if constraint_lines:
        return f"{base}\n\nConstraints:\n" + "\n".join(constraint_lines)

    return base


def _get_completion(prompt: str) -> str:
    """Get completion from LLM (sync wrapper)."""
    # This would normally call OpenAI async client
    # For now, return placeholder
    return "ORG start\nstart: MOV 0, 1\nEND"


def generate_with_hints(hints: dict) -> dict:
    """Generate warrior source from hints.

    Returns:
        {"status": "ok", "source": str} on success
        {"status": "error", "error_type": str, "message": str} on failure
    """
    try:
        prompt = build_prompt(hints)
        source = _get_completion(prompt)
        return {"status": "ok", "source": source}
    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e),
        }
