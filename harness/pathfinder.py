"""Pathfinder — BFS implementation for calculating optimal routes to objectives."""

from __future__ import annotations

from collections import deque
from typing import Optional

from engine.world import World, Tile
from engine.entities import Door


def find_shortest_path(
    world: World,
    start: tuple[int, int],
    end: tuple[int, int],
    ignore_locked_doors: bool = False,
) -> Optional[list[str]]:
    """Find the shortest path from start to end using Breadth-First Search (BFS).

    Args:
        world: The game world instance.
        start: (x, y) starting coordinate.
        end: (x, y) target coordinate.
        ignore_locked_doors: If True, treats locked doors as walkable tiles.

    Returns:
        List of movement commands (e.g. ['move north', 'move east']) or None if unreachable.
    """
    if start == end:
        return []

    queue = deque([(start, [])])
    visited = {start}

    directions = [
        ((0, -1), "north"),
        ((0, 1), "south"),
        ((1, 0), "east"),
        ((-1, 0), "west"),
    ]

    while queue:
        (x, y), path = queue.popleft()

        for (dx, dy), dir_name in directions:
            nx, ny = x + dx, y + dy

            if (nx, ny) == end:
                return path + [f"move {dir_name}"]

            # Check boundaries
            if nx < 0 or nx >= world.width or ny < 0 or ny >= world.height:
                continue

            # Check walkability based on ignore_locked_doors flag
            tile = world.get_tile(nx, ny)
            if not tile.walkable:
                continue

            # Check entities at this tile
            blocked = False
            for entity in world.get_entities_at(nx, ny):
                if isinstance(entity, Door) and entity.locked:
                    if not ignore_locked_doors:
                        blocked = True
                        break

            if blocked:
                continue

            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [f"move {dir_name}"]))

    return None
