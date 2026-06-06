"""Actions — defines the action space and provides parsing/validation.

The action space is deliberately kept simple and discrete to maximize
the LLM's ability to reliably choose and execute actions. Each action
has a clear text format that the LLM can output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """All possible action types the agent can take."""
    MOVE = "move"
    TURN = "turn"
    LOOK = "look"
    PICKUP = "pickup"
    USE = "use"
    INTERACT = "interact"
    WAIT = "wait"


@dataclass
class ParsedAction:
    """A parsed and validated action from LLM output."""
    action_type: ActionType
    raw_text: str
    direction: str = ""
    target: str = ""
    item: str = ""
    valid: bool = True
    error: str = ""


# Action format documentation for the LLM
ACTION_SPACE_DESCRIPTION = """
## Available Actions

You can take the following actions each turn. Use EXACTLY these formats:

| Action | Format | Description |
|--------|--------|-------------|
| Move | `move <direction>` | Move one tile. Directions: north, south, east, west |
| Turn | `turn <direction>` | Face a direction without moving |
| Look | `look` | Get a detailed description of your surroundings |
| Pick up | `pickup <item name>` | Pick up an item at your position or adjacent to you |
| Use | `use <item> on <target>` | Use an inventory item on a nearby object |
| Interact | `interact <object>` | Interact with a sign or other object |
| Wait | `wait` | Do nothing for one turn |

### Important Rules:
- You can only pick up keys and gems
- You must be adjacent to (or standing on) an item to pick it up
- You can only use items that are in your inventory
- To unlock a door, use the matching colored key: `use blue key on blue door`
- You can only move to walkable tiles (floors and unlocked doors)
- Locked doors block movement — unlock them first
""".strip()


def parse_action(raw_text: str) -> ParsedAction:
    """Parse a raw action string from LLM output into a structured action.
    
    This is intentionally lenient in parsing to handle minor LLM formatting
    variations while still validating the core action structure.
    """
    text = raw_text.strip().lower()

    # Handle empty input
    if not text:
        return ParsedAction(
            action_type=ActionType.WAIT,
            raw_text=raw_text,
            valid=False,
            error="No action provided.",
        )

    parts = text.split()
    verb = parts[0]

    # Match against known action types
    if verb == "move":
        if len(parts) < 2:
            return ParsedAction(
                action_type=ActionType.MOVE, raw_text=raw_text,
                valid=False, error="Move needs a direction: move north/south/east/west",
            )
        direction = parts[1]
        if direction not in ("north", "south", "east", "west"):
            return ParsedAction(
                action_type=ActionType.MOVE, raw_text=raw_text,
                valid=False, error=f"Invalid direction '{direction}'. Use: north, south, east, west",
            )
        return ParsedAction(ActionType.MOVE, raw_text, direction=direction)

    elif verb == "turn":
        if len(parts) < 2:
            return ParsedAction(
                action_type=ActionType.TURN, raw_text=raw_text,
                valid=False, error="Turn needs a direction: turn north/south/east/west",
            )
        direction = parts[1]
        if direction not in ("north", "south", "east", "west"):
            return ParsedAction(
                action_type=ActionType.TURN, raw_text=raw_text,
                valid=False, error=f"Invalid direction '{direction}'.",
            )
        return ParsedAction(ActionType.TURN, raw_text, direction=direction)

    elif verb == "look":
        return ParsedAction(ActionType.LOOK, raw_text)

    elif verb == "pickup" or verb == "pick":
        # Handle both "pickup X" and "pick up X"
        if verb == "pick" and len(parts) > 1 and parts[1] == "up":
            item_name = " ".join(parts[2:])
        else:
            item_name = " ".join(parts[1:])
        
        if not item_name:
            return ParsedAction(
                action_type=ActionType.PICKUP, raw_text=raw_text,
                valid=False, error="Pick up what? Specify: pickup <item name>",
            )
        return ParsedAction(ActionType.PICKUP, raw_text, target=item_name)

    elif verb == "use":
        full = " ".join(parts[1:])
        if " on " not in full:
            return ParsedAction(
                action_type=ActionType.USE, raw_text=raw_text,
                valid=False, error="Use format: use <item> on <target>. Example: use blue key on blue door",
            )
        item, target = full.split(" on ", 1)
        return ParsedAction(ActionType.USE, raw_text, item=item.strip(), target=target.strip())

    elif verb == "interact":
        target = " ".join(parts[1:])
        if not target:
            return ParsedAction(
                action_type=ActionType.INTERACT, raw_text=raw_text,
                valid=False, error="Interact with what? Specify: interact <target>",
            )
        return ParsedAction(ActionType.INTERACT, raw_text, target=target)

    elif verb == "throw":
        if len(parts) < 3:
            return ParsedAction(
                action_type=ActionType.THROW, raw_text=raw_text,
                valid=False, error="Throw what where? Format: throw <item> <direction>",
            )
        direction = parts[-1]
        item = " ".join(parts[1:-1])
        if direction not in ("north", "south", "east", "west"):
            return ParsedAction(
                action_type=ActionType.THROW, raw_text=raw_text,
                valid=False, error=f"Invalid throw direction '{direction}'. Use: north, south, east, west",
            )
        return ParsedAction(ActionType.THROW, raw_text, item=item, direction=direction)

    elif verb == "wait":
        return ParsedAction(ActionType.WAIT, raw_text)

    else:
        return ParsedAction(
            action_type=ActionType.WAIT, raw_text=raw_text,
            valid=False,
            error=f"Unknown action '{verb}'. Valid: move, turn, look, pickup, use, interact, wait",
        )
