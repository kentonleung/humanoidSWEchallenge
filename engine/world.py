"""World — the 2D grid world with tiles, rooms, and entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from engine.entities import (
    Entity, EntityType, Color,
    Key, Door, Gem, Sign, GoalMarker, Weapon, Monster
)
import random


class Tile(Enum):
    """Types of tiles in the world grid."""
    FLOOR = "."
    WALL = "#"
    WATER = "~"

    @property
    def walkable(self) -> bool:
        return self == Tile.FLOOR

    @property
    def description(self) -> str:
        descriptions = {
            Tile.FLOOR: "floor",
            Tile.WALL: "wall",
            Tile.WATER: "water",
        }
        return descriptions[self]


@dataclass
class Room:
    """A named region of the world for spatial reasoning."""
    name: str
    x1: int  # Top-left corner
    y1: int
    x2: int  # Bottom-right corner
    y2: int

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class World:
    """The complete world state: grid, entities, and rooms.
    
    The world is a 2D grid where:
      - Each cell has a tile type (floor, wall, water)
      - Entities (keys, doors, gems, etc.) occupy positions on the grid
      - Rooms provide named regions for spatial descriptions
    """
    width: int = 0
    height: int = 0
    grid: list[list[Tile]] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    name: str = "Unknown"
    description: str = ""

    def get_tile(self, x: int, y: int) -> Tile:
        """Get the tile at position (x, y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return Tile.WALL  # Out of bounds = wall

    def is_walkable(self, x: int, y: int) -> bool:
        """Check if the position is walkable (floor and no blocking entities)."""
        tile = self.get_tile(x, y)
        if not tile.walkable:
            return False

        # Check for blocking entities (locked doors)
        for entity in self.entities:
            if entity.x == x and entity.y == y:
                if isinstance(entity, Door) and entity.locked:
                    return False
        return True

    def get_entities_at(self, x: int, y: int) -> list[Entity]:
        """Get all entities at a given position."""
        return [e for e in self.entities if e.x == x and e.y == y]

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Find an entity by name (case-insensitive)."""
        for entity in self.entities:
            if entity.name.lower() == name.lower():
                return entity
        return None

    def get_adjacent_entities(self, x: int, y: int) -> list[tuple[Entity, str]]:
        """Get entities adjacent to (x, y) with their relative direction."""
        adjacent = []
        directions = [
            ((0, -1), "north"), ((0, 1), "south"),
            ((1, 0), "east"), ((-1, 0), "west"),
        ]
        for (dx, dy), dir_name in directions:
            for entity in self.get_entities_at(x + dx, y + dy):
                adjacent.append((entity, dir_name))
        return adjacent

    def get_room_at(self, x: int, y: int) -> Optional[Room]:
        """Get the room containing the given position."""
        for room in self.rooms:
            if room.contains(x, y):
                return room
        return None

    def has_line_of_sight(self, x0: int, y0: int, x1: int, y1: int) -> bool:
        """Check if there is a clear line of sight between two points using Bresenham's line algorithm."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            # Exclude the exact start point from blocking vision, check path
            if (x0 != x1 or y0 != y1):
                tile = self.get_tile(x0, y0)
                if tile == Tile.WALL:
                    return False

            if x0 == x1 and y0 == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
                
        return True

    def get_visible_entities(self, x: int, y: int, radius: int = 5) -> list[tuple[Entity, int, int]]:
        """Get entities visible within a radius, with relative offsets (dx, dy)."""
        visible = []
        for entity in self.entities:
            dx = entity.x - x
            dy = entity.y - y
            if abs(dx) <= radius and abs(dy) <= radius:
                # Skip entities on the agent's exact position (they're in inventory or underfoot)
                if dx == 0 and dy == 0:
                    continue
                if self.has_line_of_sight(x, y, entity.x, entity.y):
                    visible.append((entity, dx, dy))
        return visible

    def remove_entity(self, entity: Entity) -> None:
        """Remove an entity from the world."""
        self.entities = [e for e in self.entities if e is not entity]

    def get_local_grid(self, cx: int, cy: int, radius: int = 3) -> list[list[str]]:
        """Get a local grid view centered on (cx, cy) with given radius.
        
        Returns a (2*radius+1) x (2*radius+1) grid of characters.
        """
        size = 2 * radius + 1
        local = []
        for dy in range(-radius, radius + 1):
            row = []
            for dx in range(-radius, radius + 1):
                wx, wy = cx + dx, cy + dy
                if dx == 0 and dy == 0:
                    row.append("@")  # Agent position
                    continue
                    
                if not self.has_line_of_sight(cx, cy, wx, wy):
                    row.append(" ")
                    continue

                # Check for entities first
                entities_here = self.get_entities_at(wx, wy)
                if entities_here:
                    # Show the most important entity
                    row.append(entities_here[0].symbol)
                else:
                    tile = self.get_tile(wx, wy)
                    row.append(tile.value)
            local.append(row)
        return local

    def to_full_grid(self, agent_x: int, agent_y: int) -> list[list[str]]:
        """Render the full world grid as characters, including agent position."""
        result = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if x == agent_x and y == agent_y:
                    row.append("@")
                    continue
                entities_here = self.get_entities_at(x, y)
                if entities_here:
                    row.append(entities_here[0].symbol)
                else:
                    row.append(self.grid[y][x].value)
            result.append(row)
        return result

    def get_global_map_blind(self, agent_x: int, agent_y: int) -> list[list[str]]:
        """Return the full map showing only architecture and generic doors (no items/monsters)."""
        from engine.entities import Door
        result = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if x == agent_x and y == agent_y:
                    row.append("@")
                    continue
                door = next((e for e in self.get_entities_at(x, y) if isinstance(e, Door)), None)
                if door:
                    row.append("D")
                else:
                    row.append(self.grid[y][x].value)
            result.append(row)
        return result

    def to_dict(self) -> dict:
        """Serialize world state for logging and visualization."""
        grid_chars = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                row.append(self.grid[y][x].value)
            grid_chars.append(row)
        
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "grid": grid_chars,
            "entities": [
                {
                    "type": e.entity_type.value,
                    "name": e.name,
                    "x": e.x,
                    "y": e.y,
                    "symbol": e.symbol,
                    "color": e.color.value if e.color else None,
                    "locked": e.locked if isinstance(e, Door) else None,
                    "collected": getattr(e, 'collected', None),
                    "message": e.message if isinstance(e, Sign) else None,
                    "alive": getattr(e, 'alive', None),
                }
                for e in self.entities
            ],
            "rooms": [
                {"name": r.name, "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2}
                for r in self.rooms
            ],
        }


# ─── Level Definitions ──────────────────────────────────────────────────────────

def _parse_level(ascii_map: str) -> tuple[list[list[Tile]], int, int]:
    """Parse an ASCII map string into a tile grid."""
    lines = [line for line in ascii_map.strip().split("\n")]
    height = len(lines)
    width = max(len(line) for line in lines)

    grid = []
    for line in lines:
        row = []
        for ch in line.ljust(width):
            if ch == "#":
                row.append(Tile.WALL)
            elif ch == "~":
                row.append(Tile.WATER)
            else:
                row.append(Tile.FLOOR)
        grid.append(row)
    return grid, width, height


def _spawn_random_entities(world: World, start_x: int, start_y: int, num_monsters: int, num_weapons: int):
    """Randomly spawn monsters and weapons away from the agent."""
    valid_positions = []
    for y in range(world.height):
        for x in range(world.width):
            if world.is_walkable(x, y) and not world.get_entities_at(x, y):
                dist = abs(x - start_x) + abs(y - start_y)
                if dist >= 6:
                    valid_positions.append((x, y))
                    
    random.shuffle(valid_positions)
    
    spawned_weapon_rooms = set()
    for _ in range(num_weapons):
        spawned = False
        for i, pos in enumerate(valid_positions):
            x, y = pos
            room = world.get_room_at(x, y)
            room_name = room.name if room else f"no_room_{x}_{y}"
            if room_name not in spawned_weapon_rooms:
                spawned_weapon_rooms.add(room_name)
                world.entities.append(Weapon(x, y))
                valid_positions.pop(i)
                spawned = True
                break
        if not spawned and valid_positions:
            x, y = valid_positions.pop()
            world.entities.append(Weapon(x, y))
            
    from engine.entities import Color
    
    spawned_monster_rooms = set()
    monster_colors = [Color.RED, Color.BLUE, Color.GREEN, Color.GOLD]
    color_idx = 0
    for _ in range(num_monsters):
        spawned = False
        for i, pos in enumerate(valid_positions):
            x, y = pos
            room = world.get_room_at(x, y)
            room_name = room.name if room else f"no_room_{x}_{y}"
            if room_name not in spawned_monster_rooms:
                spawned_monster_rooms.add(room_name)
                world.entities.append(Monster(x, y, color=monster_colors[color_idx % len(monster_colors)]))
                valid_positions.pop(i)
                color_idx += 1
                spawned = True
                break
        if not spawned and valid_positions:
            x, y = valid_positions.pop()
            world.entities.append(Monster(x, y, color=monster_colors[color_idx % len(monster_colors)]))
            color_idx += 1

def create_level_1() -> tuple[World, int, int]:
    """Level 1: Navigation — Go to the red gem in the east room.
    
    Simple two-room layout connected by an open passage.
    Returns (world, agent_start_x, agent_start_y).
    """
    ascii_map = """\
################
#......#.......#
#......#.......#
#......#.......#
#......#.......#
#......._.......#
#......#.......#
#......#.......#
#......#.......#
################"""

    # Actually, let's make a cleaner level
    ascii_map = """\
####################
#........#.........#
#........#.........#
#........#.........#
#........#.........#
#........._.........#
#........#.........#
#........#.........#
#........#.........#
####################"""

    # Hand-craft a clean grid instead
    width, height = 20, 10
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                row.append(Tile.WALL)
            elif x == 9 and y != 5:
                row.append(Tile.WALL)  # Dividing wall with passage at y=5
            else:
                row.append(Tile.FLOOR)
        grid.append(row)

    world = World(
        width=width,
        height=height,
        grid=grid,
        name="Twin Rooms",
        description="Two rooms connected by a passage.",
        rooms=[
            Room("West Room", 1, 1, 8, 8),
            Room("East Room", 10, 1, 18, 8),
        ],
        entities=[
            Gem(15, 3, Color.RED),
            Sign(3, 3, "The red gem is in the East Room. Go through the passage to find it."),
        ],
    )
    return world, 2, 5  # Agent starts in West Room


def create_level_2() -> tuple[World, int, int]:
    """Level 2: Key & Door — Find the blue key and unlock the blue door.
    
    Three rooms: start room, key room, and goal room behind locked door.
    Returns (world, agent_start_x, agent_start_y).
    """
    width, height = 25, 12
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                row.append(Tile.WALL)
            elif x == 8 and y != 3:
                row.append(Tile.WALL)  # Wall between start and key room
            elif x == 16 and y != 6:
                row.append(Tile.WALL)  # Wall between start and goal room
            else:
                row.append(Tile.FLOOR)
        grid.append(row)

    world = World(
        width=width,
        height=height,
        grid=grid,
        name="Key & Door Challenge",
        description="Find the key, then unlock the door to reach the goal.",
        rooms=[
            Room("Starting Room", 1, 1, 7, 10),
            Room("Key Chamber", 9, 1, 15, 10),
            Room("Treasure Room", 17, 1, 23, 10),
        ],
        entities=[
            Key(12, 8, Color.BLUE),
            Door(16, 6, Color.BLUE, locked=True),
            GoalMarker(20, 6),
            Sign(3, 5, "A blue key lies in the chamber to the north-east. Use it to unlock the blue door."),
        ],
    )
    _spawn_random_entities(world, 4, 8, num_monsters=1, num_weapons=1)
    return world, 4, 8  # Agent starts in Starting Room


def create_level_3() -> tuple[World, int, int]:
    """Level 3: Dungeon Escape — Collect all gems, unlock the exit, escape.
    
    A larger dungeon with 4 rooms, multiple keys, doors, and 3 gems to collect.
    Returns (world, agent_start_x, agent_start_y).
    """
    width, height = 30, 16
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                row.append(Tile.WALL)
            # Horizontal wall dividing top and bottom
            elif y == 8 and not (x == 5 or x == 22):
                row.append(Tile.WALL)
            # Vertical wall dividing left and right (top half)
            elif x == 14 and y < 8 and y != 4:
                row.append(Tile.WALL)
            # Vertical wall dividing left and right (bottom half)
            elif x == 14 and y > 8 and y != 12:
                row.append(Tile.WALL)
            # Water hazard in bottom-right room
            elif y in (10, 11) and x in (20, 21, 22) and y > 8:
                row.append(Tile.WATER)
            else:
                row.append(Tile.FLOOR)
        grid.append(row)

    world = World(
        width=width,
        height=height,
        grid=grid,
        name="Dungeon Escape",
        description="Explore the dungeon, collect all gems, and find the exit.",
        rooms=[
            Room("Entry Hall", 1, 1, 13, 7),
            Room("Armory", 15, 1, 28, 7),
            Room("Catacombs", 1, 9, 13, 14),
            Room("Flooded Chamber", 15, 9, 28, 14),
        ],
        entities=[
            # Gems — scattered across rooms
            Gem(4, 3, Color.RED),
            Gem(22, 3, Color.BLUE),
            Gem(6, 12, Color.GREEN),

            # Keys
            Key(25, 5, Color.RED),    # Red key in Armory
            Key(3, 13, Color.GREEN),  # Green key in Catacombs

            # Doors
            Door(14, 4, Color.RED, locked=True),   # Between Entry Hall and Armory
            Door(14, 12, Color.GREEN, locked=True), # Between Catacombs and Flooded Chamber

            # Exit
            GoalMarker(27, 12),

            # Signs
            Sign(7, 2, "Welcome! Locations: Gems: Red(4, 3), Blue(22, 3), Green(6, 12). Keys: Red(25, 5), Green(3, 13). Doors: Red(14, 4), Green(14, 12). Exit(27, 12)."),
        ],
    )
    _spawn_random_entities(world, 7, 5, num_monsters=2, num_weapons=2)
    return world, 7, 5  # Agent starts in Entry Hall


# Registry of all levels
LEVELS = {
    "navigate": (create_level_1, "Go to the red gem in the East Room"),
    "key_door": (create_level_2, "Find the blue key and unlock the blue door to reach the goal marker"),
    "dungeon_escape": (create_level_3, "Collect all 3 gems (red, blue, green) and reach the goal marker to escape"),
}
