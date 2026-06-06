"""Prompts — system prompt and templates for the LLM agent.

The prompt design is critical to agent performance. Key principles:
  1. Clear world rules — the LLM must understand what's possible
  2. Structured output format — chain-of-thought followed by action
  3. Constrained action space — reduce hallucination via valid action list
  4. Memory context — recent history prevents loops
"""

from __future__ import annotations

from harness.actions import ACTION_SPACE_DESCRIPTION


SYSTEM_PROMPT = f"""You are an intelligent agent navigating a 2D dungeon world. You perceive the world through structured observations and take actions to accomplish goals.

## World Rules
- The world is a 2D grid of tiles: floor (walkable), walls (impassable), water (impassable)
- Entities include: keys, locked doors, gems, signs, and Dungeon Exits
- You can only move to walkable tiles (floor or unlocked doors)
- Locked doors block movement — you must use the matching colored key to unlock them
- Keys are consumed when used to unlock a door
- You can pick up keys and gems that are at your position or adjacent to you
- Signs can be read by interacting with them
- Your goal is to complete the assigned task. The exit is marked as 'Dungeon Exit'
- WARNING: There are roaming monsters in the dungeon! If a monster is adjacent to you (or on your tile) and you do NOT have a weapon, you will be caught and the run will fail.
- If you have a weapon in your inventory, you will automatically defeat any adjacent monster.
- STRATEGY HINT: It is highly recommended to pick up a weapon if you see one, even if no monsters are currently visible, to prepare for unexpected ambushes.
- STRATEGY HINT: If you see a sign (scroll) THAT YOU HAVE NOT YET READ, you should prioritize moving to it and reading it before exploring randomly, AS LONG AS it is safe to do so. Do not re-read signs you already have in your memory. Never walk towards a monster to read a sign.
- STRATEGY HINT: When navigating to a known coordinate, do NOT walk blindly towards it if walls are blocking your path! Use your [Global GPS Map] to plan a path through doors or hallways around walls to avoid getting stuck in a movement loop.
- CRITICAL: YOUR TOP PRIORITY IS SAFETY! The mission is secondary to safety. If you see a monster and do NOT have a weapon, you MUST RETREAT. Do not wait for it to move, and do not move towards it. Move backwards or in any safe direction away from the monster! Do not let it get adjacent to you!

{ACTION_SPACE_DESCRIPTION}

## How to Respond

Each turn, you will receive an observation showing your current state. You MUST respond in this EXACT format:

```
THINK: <your step-by-step reasoning about what to do next>
ACTION: <your chosen action, using the exact format above>
```

### Reasoning Guidelines:
1. First, review the last action's result — did it succeed? If not, why?
2. Consider your current position and what you can see
3. Think about what you need to do to accomplish the task
4. Choose the best action from the valid actions list
5. If you're stuck, try 'look' to get more information or explore a new direction

### CRITICAL RULES:
- Your ACTION line must contain EXACTLY ONE action from the valid actions list
- Do NOT include any text after the ACTION line
- Do NOT make up actions that aren't in the valid actions list
- Do NOT use emojis in your response. Output plain text only.
- If your last action failed, DO NOT repeat it. Try a different action.
- You CANNOT pick up items that are far away. You MUST move to be exactly adjacent to or standing on an item before using the `pickup` action.
- If you are stuck in a loop moving back and forth, move in a NEW direction or use 'look' to gather more information.
- Pay attention to locked doors — you need the matching key
"""


def build_turn_prompt(
    task_description: str,
    observation_text: str,
    history: list[dict],
    max_history: int = 8,
) -> str:
    """Build the user prompt for a single turn.
    
    Args:
        task_description: The current task/goal description.
        observation_text: Formatted observation from the observation module.
        history: List of recent {action, result, reasoning} dicts.
        max_history: Maximum number of history entries to include.
        
    Returns:
        The complete user prompt for this turn.
    """
    sections = []

    # Task reminder
    sections.append(f"## Your Task\n{task_description}")

    # Recent history (sliding window)
    if history:
        recent = history[-max_history:]
        history_lines = []
        for i, entry in enumerate(recent):
            step_num = len(history) - len(recent) + i + 1
            action = entry.get("action", "?")
            result = entry.get("result", "?")
            history_lines.append(f"  Step {step_num}: {action} → {result}")
        sections.append(f"\n## Recent History\n" + "\n".join(history_lines))

    # Current observation
    sections.append(f"\n## Current Observation\n{observation_text}")

    # Prompt for response
    sections.append(
        "\nBased on the observation above, decide your next action. "
        "Remember: respond with THINK: then ACTION:"
    )

    return "\n".join(sections)


def parse_llm_response(response_text: str) -> tuple[str, str]:
    """Parse the LLM's response into (reasoning, action).
    
    Handles various formatting inconsistencies from LLMs:
      - Extra whitespace, markdown formatting
      - Missing THINK: or ACTION: prefixes
      - Multi-line thinking
    
    Returns:
        (thinking_text, action_text) tuple.
    """
    if not response_text:
        return "I was unable to form a thought (blocked or empty response).", "look"
        
    text = response_text.strip()

    thinking = ""
    action = ""

    # Try to find THINK: and ACTION: sections
    lines = text.split("\n")
    
    in_think = False
    in_action = False
    think_lines = []
    action_lines = []

    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("THINK:"):
            in_think = True
            in_action = False
            content = stripped[6:].strip()
            if content:
                think_lines.append(content)
        elif upper.startswith("ACTION:"):
            in_think = False
            in_action = True
            content = stripped[7:].strip()
            if content:
                action_lines.append(content)
        elif in_think:
            if stripped:
                think_lines.append(stripped)
        elif in_action:
            if stripped:
                action_lines.append(stripped)

    thinking = " ".join(think_lines)
    action = " ".join(action_lines)

    # Fallback: if no ACTION: found, try to extract the last line as action
    if not action:
        for line in reversed(lines):
            stripped = line.strip().lower()
            if stripped and any(stripped.startswith(cmd) for cmd in
                               ["move", "turn", "look", "pickup", "pick up",
                                "use", "interact", "wait"]):
                action = stripped
                break

    # If still no action, default to look
    if not action:
        action = "look"
        if not thinking:
            thinking = "I'm not sure what to do, so I'll look around."

    # Clean up action — remove markdown formatting
    action = action.strip("`").strip("*").strip()
    
    return thinking, action
