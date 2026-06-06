"""Observations — formats raw environment state into structured text for the LLM.

This module is the key interface design challenge. It determines WHAT the agent
knows and HOW it knows it. The observation format must be:
  1. Information-dense enough for the agent to make good decisions
  2. Structured enough for the LLM to parse reliably
  3. Not so verbose that it wastes context window

Design philosophy: Provide dual-modality perception (visual grid + semantic text),
similar to how a real robot might combine camera feeds with processed sensor data.
"""

from __future__ import annotations

from typing import Optional
from engine.environment import Observation
from engine.entities import Door, Gem, Key, Sign, GoalMarker


def format_observation(
    obs: Observation,
    include_grid: bool = True,
    spatial_memory: Optional[dict] = None,
) -> str:
    """Format an observation into a complete text representation for the LLM.
    
    The output includes:
      1. Visual grid — a local 7×7 ASCII map centered on the agent
      2. Semantic description — natural language description of surroundings
      3. Inventory — what the agent is carrying
      4. Valid actions — constrained list of legal actions
      5. Task progress — current goal status
      6. Loop warnings / Alerts (from memory)
      7. Read sign messages (from memory)
    
    Args:
        obs: The raw observation from the environment.
        include_grid: Whether to include the ASCII grid view.
        spatial_memory: Optional dictionary containing agent's spatial memory context.
        
    Returns:
        Formatted observation string.
    """
    sections = []

    # ── Header ──
    sections.append(f"═══ Step {obs.step_number}/{obs.max_steps} ═══")

    # ── Spatial Memory Alerts ──
    if spatial_memory and "loop_warning" in spatial_memory and spatial_memory["loop_warning"]:
        sections.append(f"\n[WARNING]  [Spatial Memory Alert]\n{spatial_memory['loop_warning']}")

    # ── Turn-Based Tactical Alerts ──
    from engine.entities import Monster, Weapon
    has_weapon = any(isinstance(i, Weapon) for i in obs.agent.inventory)
    
    tactical_alerts = []
    if not has_weapon:
        for entity, dx, dy in obs.visible_entities:
            if isinstance(entity, Monster) and entity.stun_duration <= 0:
                manhattan_dist = abs(dx) + abs(dy)
                if manhattan_dist == 2:
                    tactical_alerts.append(f"[WARNING]  [Tactical Alert] A {entity.name} is exactly 2 tiles away (Manhattan distance)! Because it is your turn first, if you move towards it, you will become adjacent. Then it will be the monster's turn, and if it wanders onto your tile, you will DIE. Be very careful!")
                elif manhattan_dist == 1:
                    tactical_alerts.append(f"[WARNING]  [CRITICAL DANGER] A {entity.name} is ADJACENT to you! If it randomly wanders onto your tile next turn, you will DIE! You MUST retreat NOW!")
                    
    if tactical_alerts:
        sections.append("\n" + "\n".join(tactical_alerts))

    # ── Last Action Result ──
    if obs.last_action_result:
        status = "[PASS]" if obs.last_action_result.success else "[FAIL]"
        sections.append(f"\n[Last Action Result] {status} {obs.last_action_result.message}")

    # ── Location ──
    sections.append(f"\n[Location] Position ({obs.agent.x}, {obs.agent.y}) in \"{obs.current_room}\" | Facing {obs.agent.facing.value}")

    # ── Local Grid View ──
    if include_grid:
        grid_str = _format_grid(obs.local_grid)
        sections.append(f"\n[Vision — 7×7 local grid]\n{grid_str}")
        sections.append(_grid_legend())

    # ── Visible Entities ──
    if obs.visible_entities:
        entity_lines = []
        for entity, dx, dy in sorted(obs.visible_entities, key=lambda e: abs(e[1]) + abs(e[2])):
            desc = _describe_entity(entity)
            direction = _offset_to_words(dx, dy)
            dist = max(abs(dx), abs(dy))
            entity_lines.append(f"  • {desc} — {dist} tile(s) {direction}")
        sections.append(f"\n[Visible Entities]\n" + "\n".join(entity_lines))
    else:
        sections.append("\n[Visible Entities] Nothing notable in sight.")

    # ── Global GPS Map (Architecture Only) ──
    global_grid = obs.world.get_global_map_blind(obs.agent.x, obs.agent.y)
    global_grid_str = _format_grid(global_grid)
    sections.append(f"\n[Global GPS Map — Layout Only]\n{global_grid_str}")
    sections.append("  Legend: @ = you, # = wall, . = floor, ~ = water, D = door")

    # ── Adjacent (Interactable) ──
    if obs.adjacent_entities:
        adj_lines = []
        for entity, direction in obs.adjacent_entities:
            desc = _describe_entity(entity)
            adj_lines.append(f"  • {desc} — {direction}")
        sections.append(f"\n[Adjacent / Interactable]\n" + "\n".join(adj_lines))

    # ── Known Sign Messages (from Memory) ──
    sign_memory_text = _get_sign_memory_text(spatial_memory)
    if sign_memory_text:
        sections.append(sign_memory_text)

    # ── Seen Items (from Memory) ──
    seen_items_text = _get_seen_items_text(spatial_memory)
    if seen_items_text:
        sections.append(seen_items_text)

    # ── Blocked Doors (Persistent Memory) ──
    blocked_doors_text = _get_blocked_doors_text(spatial_memory)
    if blocked_doors_text:
        sections.append(blocked_doors_text)

    # ── Inventory ──
    inv_str = obs.agent.inventory_str()
    sections.append(f"\n[Inventory] {inv_str}")

    # ── Task ──
    sections.append(f"\n[Task] {obs.task_progress.to_str() if obs.task_progress.objectives else 'No objectives yet.'}")
    if obs.task_progress.message:
        sections.append(f"  → {obs.task_progress.message}")

    # ── Valid Actions ──
    actions_str = ", ".join(obs.valid_actions)
    sections.append(f"\n[Valid Actions] {actions_str}")

    return "\n".join(sections)





def _get_sign_memory_text(spatial_memory: Optional[dict]) -> str:
    """Format known sign contents from spatial memory."""
    if not spatial_memory or "known_signs" not in spatial_memory:
        return ""
    known_signs = spatial_memory["known_signs"]
    if not known_signs:
        return ""
    lines = []
    for loc_desc, message in sorted(known_signs.items()):
        lines.append(f"  • {loc_desc}: \"{message}\"")
    return "\n[Known Sign Messages (from Memory)]\n" + "\n".join(lines)


def _get_seen_items_text(spatial_memory: Optional[dict]) -> str:
    """Format known item locations from spatial memory."""
    if not spatial_memory or "seen_items" not in spatial_memory:
        return ""
    seen_items = spatial_memory["seen_items"]
    nav_hints = spatial_memory.get("navigation_hints", {})
    if not seen_items and not nav_hints:
        return ""
    lines = []
    
    # Track what we've displayed to avoid duplicates
    displayed = set()
    
    for item_name, coord in sorted(seen_items.items()):
        displayed.add(item_name)
        if item_name in nav_hints:
            lines.append(f"  • {item_name} {nav_hints[item_name]}")
        else:
            lines.append(f"  • {item_name} at {coord}")
            
    for hint_name, hint_text in sorted(nav_hints.items()):
        if hint_name not in displayed:
            lines.append(f"  • {hint_name} {hint_text}")

    return "\n[Remembered Item Locations (from Memory)]\n" + "\n".join(lines)


def _get_blocked_doors_text(spatial_memory: Optional[dict]) -> str:
    """Format known locked doors from persistent memory."""
    if not spatial_memory or "blocked_doors" not in spatial_memory:
        return ""
    blocked_doors = spatial_memory["blocked_doors"]
    if not blocked_doors:
        return ""
    lines = []
    for door_key, door_info in sorted(blocked_doors.items()):
        lines.append(f"  ⛔ {door_key} — REQUIRES: {door_info['needs']}. DO NOT go there without the key!")
    return (
        "\n[WARNING]  [BLOCKED PATHS — LOCKED DOORS] [WARNING]\n"
        "You have encountered these locked doors. Do NOT walk towards them unless you have the matching key!\n"
        + "\n".join(lines)
    )


def format_observation_compact(obs: Observation) -> str:
    """A more compact observation format — useful for conserving context window.
    
    Omits the grid view and provides only essential information.
    """
    return format_observation(obs, include_grid=False)


def _format_grid(grid: list[list[str]]) -> str:
    """Format a 2D grid into a bordered ASCII display."""
    lines = []
    width = len(grid[0]) if grid else 0
    border = "+" + "-" * (width * 2 + 1) + "+"
    lines.append(border)
    for row in grid:
        line = "| " + " ".join(row) + " |"
        lines.append(line)
    lines.append(border)
    return "\n".join(lines)


def _grid_legend() -> str:
    """Legend for grid symbols."""
    return (
        "  Legend: @ = you, # = wall, . = floor, ~ = water, "
        "K = key, D = locked door, d = open door, G = gem, ! = sign, ★ = Dungeon Exit"
    )


def _describe_entity(entity) -> str:
    """Generate a concise description of an entity."""
    if isinstance(entity, Door):
        state = "locked" if entity.locked else "unlocked"
        return f"{entity.name} ({state})"
    elif isinstance(entity, Gem):
        return f"{entity.name}"
    elif isinstance(entity, Key):
        return f"{entity.name}"
    elif isinstance(entity, Sign):
        return "sign"
    elif isinstance(entity, GoalMarker):
        return "Dungeon Exit ★"
    else:
        return getattr(entity, "name", "unknown")


def _offset_to_words(dx: int, dy: int) -> str:
    """Convert a (dx, dy) offset to directional words."""
    parts = []
    if dy < 0:
        parts.append("north")
    elif dy > 0:
        parts.append("south")
    if dx > 0:
        parts.append("east")
    elif dx < 0:
        parts.append("west")
    return "-".join(parts) if parts else "here"
