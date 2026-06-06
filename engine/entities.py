"""Game entities — objects that exist in the world and can be interacted with."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EntityType(Enum):
    """All entity types in the game world."""
    KEY = "key"
    DOOR = "door"
    GEM = "gem"
    SIGN = "sign"
    GOAL_MARKER = "goal_marker"
    WEAPON = "weapon"
    MONSTER = "monster"
    POTION = "potion"


class Color(Enum):
    """Colors used to match keys to doors and distinguish collectibles."""
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    GOLD = "gold"


@dataclass
class Entity:
    """Base class for all world entities."""
    entity_type: EntityType
    x: int
    y: int
    color: Optional[Color] = None
    name: str = ""

    def __post_init__(self):
        if not self.name:
            color_prefix = f"{self.color.value} " if self.color else ""
            self.name = f"{color_prefix}{self.entity_type.value}"

    @property
    def position(self) -> tuple[int, int]:
        return (self.x, self.y)

    @property
    def symbol(self) -> str:
        """Single character for grid rendering."""
        symbols = {
            EntityType.KEY: "K",
            EntityType.DOOR: "D",
            EntityType.GEM: "G",
            EntityType.SIGN: "!",
            EntityType.GOAL_MARKER: "★",
            EntityType.WEAPON: "W",
            EntityType.MONSTER: "M",
            EntityType.POTION: "P",
        }
        return symbols.get(self.entity_type, "?")


@dataclass
class Key(Entity):
    """A key that can unlock a matching colored door."""
    def __init__(self, x: int, y: int, color: Color):
        super().__init__(
            entity_type=EntityType.KEY,
            x=x, y=y,
            color=color,
            name=f"{color.value} key",
        )


@dataclass
class Door(Entity):
    """A door that blocks passage until unlocked with the matching key."""
    locked: bool = True

    def __init__(self, x: int, y: int, color: Color, locked: bool = True):
        super().__init__(
            entity_type=EntityType.DOOR,
            x=x, y=y,
            color=color,
            name=f"{color.value} door",
        )
        self.locked = locked

    @property
    def symbol(self) -> str:
        return "D" if self.locked else "d"

    def unlock(self) -> str:
        """Unlock the door. Returns a description of what happened."""
        if not self.locked:
            return f"The {self.name} is already unlocked."
        self.locked = False
        return f"You unlock the {self.name}. It swings open."


@dataclass
class Gem(Entity):
    """A collectible gem."""
    collected: bool = False

    def __init__(self, x: int, y: int, color: Color):
        super().__init__(
            entity_type=EntityType.GEM,
            x=x, y=y,
            color=color,
            name=f"{color.value} gem",
        )
        self.collected = False

    def collect(self) -> str:
        """Mark gem as collected. Returns description."""
        self.collected = True
        return f"You pick up the {self.name}. It gleams brightly."





@dataclass
class Sign(Entity):
    """A readable sign with a text message."""
    message: str = ""

    def __init__(self, x: int, y: int, message: str):
        super().__init__(
            entity_type=EntityType.SIGN,
            x=x, y=y,
            name="sign",
        )
        self.message = message

    def read(self) -> str:
        """Read the sign. Returns the message."""
        return f'The sign reads: "{self.message}"'


@dataclass
class GoalMarker(Entity):
    """Marks the goal/exit location."""
    def __init__(self, x: int, y: int):
        super().__init__(
            entity_type=EntityType.GOAL_MARKER,
            x=x, y=y,
            name="Dungeon Exit",
        )

@dataclass
class Weapon(Entity):
    """A weapon that can be picked up to fight monsters."""
    collected: bool = False

    def __init__(self, x: int, y: int):
        super().__init__(
            entity_type=EntityType.WEAPON,
            x=x, y=y,
            name="sword",
        )
        self.collected = False

    def collect(self) -> str:
        """Mark weapon as collected."""
        self.collected = True
        return f"You pick up the {self.name}. You feel ready for battle."

@dataclass
class Monster(Entity):
    """A hostile monster that roams the dungeon."""
    alive: bool = True
    stun_duration: int = 0

    def __init__(self, x: int, y: int, color: Optional[Color] = None):
        super().__init__(
            entity_type=EntityType.MONSTER,
            x=x, y=y,
            name="monster",
            color=color,
        )
        self.alive = True

    def die(self) -> str:
        """Kill the monster."""
        self.alive = False
        return f"The {self.name} dies with a terrible groan."

@dataclass
class Potion(Entity):
    """A thrown potion to stun monsters."""
    collected: bool = False

    def __init__(self, x: int, y: int):
        super().__init__(
            entity_type=EntityType.POTION,
            x=x, y=y,
            name="stun potion",
        )
        self.collected = False

    def collect(self) -> str:
        self.collected = True
        return f"You pick up the {self.name}."
