"""Agent state — tracks the agent's position, facing direction, and inventory."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from engine.entities import Entity, EntityType


class Direction(Enum):
    """Cardinal directions the agent can face and move."""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

    @property
    def delta(self) -> tuple[int, int]:
        """Returns (dx, dy) for this direction. Y increases downward."""
        deltas = {
            Direction.NORTH: (0, -1),
            Direction.SOUTH: (0, 1),
            Direction.EAST: (1, 0),
            Direction.WEST: (-1, 0),
        }
        return deltas[self]

    @property
    def opposite(self) -> Direction:
        opposites = {
            Direction.NORTH: Direction.SOUTH,
            Direction.SOUTH: Direction.NORTH,
            Direction.EAST: Direction.WEST,
            Direction.WEST: Direction.EAST,
        }
        return opposites[self]


@dataclass
class AgentState:
    """Represents the full state of the agent in the world.
    
    Attributes:
        x: Horizontal position on the grid.
        y: Vertical position on the grid (increases downward).
        facing: The direction the agent is currently facing.
        inventory: Items the agent is carrying.
        steps_taken: Number of actions executed so far.
    """
    x: int = 0
    y: int = 0
    facing: Direction = Direction.NORTH
    inventory: list[Entity] = field(default_factory=list)
    steps_taken: int = 0

    @property
    def position(self) -> tuple[int, int]:
        return (self.x, self.y)

    def move(self, direction: Direction) -> tuple[int, int]:
        """Move one step in the given direction. Returns the new position."""
        dx, dy = direction.delta
        self.x += dx
        self.y += dy
        self.facing = direction
        self.steps_taken += 1
        return self.position

    def turn(self, direction: Direction) -> None:
        """Face a new direction without moving."""
        self.facing = direction
        self.steps_taken += 1

    def add_to_inventory(self, entity: Entity) -> None:
        """Add an entity to the agent's inventory."""
        self.inventory.append(entity)

    def remove_from_inventory(self, entity_name: str) -> Optional[Entity]:
        """Remove and return an entity from inventory by name. Returns None if not found."""
        for i, entity in enumerate(self.inventory):
            if entity.name.lower() == entity_name.lower():
                return self.inventory.pop(i)
        return None

    def has_item(self, entity_name: str) -> bool:
        """Check if the agent has an item in inventory."""
        return any(e.name.lower() == entity_name.lower() for e in self.inventory)

    def has_key_for_color(self, color) -> bool:
        """Check if the agent has a key of the specified color."""
        return any(
            e.entity_type == EntityType.KEY and e.color == color
            for e in self.inventory
        )

    def get_key_for_color(self, color) -> Optional[Entity]:
        """Get and remove a key of the specified color from inventory."""
        for i, entity in enumerate(self.inventory):
            if entity.entity_type == EntityType.KEY and entity.color == color:
                return self.inventory.pop(i)
        return None

    def inventory_str(self) -> str:
        """Human-readable inventory listing."""
        if not self.inventory:
            return "empty"
        return ", ".join(e.name for e in self.inventory)

    def to_dict(self) -> dict:
        """Serialize agent state to dictionary."""
        return {
            "position": {"x": self.x, "y": self.y},
            "facing": self.facing.value,
            "inventory": [e.name for e in self.inventory],
            "steps_taken": self.steps_taken,
        }
