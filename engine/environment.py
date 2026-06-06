"""Environment — the central coordination layer between agent and world.

Provides a Gym-style interface: reset(), step(), observe().
Manages world state, validates actions, and checks task completion.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from engine.world import World, LEVELS
from engine.entities import (
    Entity, EntityType, Door, Key, Gem, Sign, GoalMarker, Weapon, Monster
)
import random
from engine.agent_state import AgentState, Direction
from engine.tasks import Task, TaskProgress, TASKS


@dataclass
class ActionResult:
    """Result of executing an action in the environment."""
    success: bool
    message: str
    action_str: str = ""


@dataclass
class Observation:
    """Everything the agent perceives after a step.
    
    This is the raw observation data — the harness module formats it
    for the LLM in different representation styles.
    """
    agent: AgentState
    world: World
    local_grid: list[list[str]]         # 7x7 grid centered on agent
    visible_entities: list[tuple[Entity, int, int]]  # (entity, dx, dy)
    adjacent_entities: list[tuple[Entity, str]]      # (entity, direction)
    current_room: Optional[str]
    task_progress: TaskProgress
    last_action_result: Optional[ActionResult]
    valid_actions: list[str]
    step_number: int
    max_steps: int


class Environment:
    """The virtual world environment.
    
    Orchestrates the game loop: creates the world, manages the agent,
    processes actions, and checks for task completion.
    """

    def __init__(self):
        self.world: Optional[World] = None
        self.agent: Optional[AgentState] = None
        self.task: Optional[Task] = None
        self.last_action_result: Optional[ActionResult] = None
        self.step_count: int = 0
        self.done: bool = False
        self.history: list[dict] = []

    def reset(self, task_id: str, seed: Optional[int] = None) -> Observation:
        """Initialize or reset the environment for a specific task.
        
        Args:
            task_id: One of 'navigate', 'key_door', 'dungeon_escape'
            seed: Optional random seed for deterministic generation
            
        Returns:
            The initial observation.
        """
        if task_id not in LEVELS:
            raise ValueError(f"Unknown task: {task_id}. Available: {list(LEVELS.keys())}")

        import random
        if seed is not None:
            random.seed(seed)
            self.seed = seed
        else:
            self.seed = random.randint(0, 1000000)
            random.seed(self.seed)

        # Create the world
        create_fn, _ = LEVELS[task_id]
        self.world, start_x, start_y = create_fn()

        # Create the agent
        self.agent = AgentState(x=start_x, y=start_y, facing=Direction.NORTH)
        
        # Create the task
        self.task = TASKS[task_id]()

        # Reset state
        self.step_count = 0
        self.done = False
        self.last_action_result = None
        self.history = []

        return self.observe()

    def observe(self) -> Observation:
        """Generate a complete observation of the current state."""
        local_grid = self.world.get_local_grid(self.agent.x, self.agent.y, radius=3)
        visible = self.world.get_visible_entities(self.agent.x, self.agent.y, radius=5)
        adjacent = self.world.get_adjacent_entities(self.agent.x, self.agent.y)
        room = self.world.get_room_at(self.agent.x, self.agent.y)
        progress = self.task.check_completion(self.world, self.agent)
        valid_actions = self._get_valid_actions()

        return Observation(
            agent=self.agent,
            world=self.world,
            local_grid=local_grid,
            visible_entities=visible,
            adjacent_entities=adjacent,
            current_room=room.name if room else "Unknown area",
            task_progress=progress,
            last_action_result=self.last_action_result,
            valid_actions=valid_actions,
            step_number=self.step_count,
            max_steps=self.task.max_steps,
        )

    def step(self, action_str: str) -> tuple[Observation, bool, dict]:
        """Execute an action and return (observation, done, info).
        
        Args:
            action_str: The action string from the agent (e.g., "move north")
            
        Returns:
            (observation, done, info) tuple.
        """
        self.step_count += 1
        result = self._execute_action(action_str)
        self.last_action_result = result

        # Combat phase 1
        combat_msg, agent_died = self._resolve_combat()
        if combat_msg:
            result.message += "\n" + combat_msg
        
        if not agent_died:
            self._tick_monsters()
            # Combat phase 2
            combat_msg2, agent_died = self._resolve_combat()
            if combat_msg2:
                result.message += "\n" + combat_msg2

        # Check completion
        progress = self.task.check_completion(self.world, self.agent)
        
        info = {
            "action": action_str,
            "result": result.message,
            "success": result.success,
            "step": self.step_count,
        }

        if agent_died:
            self.done = True
            info["completion_message"] = "Task failed: Agent was killed by a monster."
        elif progress.completed:
            self.done = True
            info["completion_message"] = progress.message
        elif self.step_count >= self.task.max_steps:
            self.done = True
            info["completion_message"] = f"Step limit reached ({self.task.max_steps} steps). Task not completed."

        # Record history
        self.history.append(info)

        observation = self.observe()
        return observation, self.done, info

    def _execute_action(self, action_str: str) -> ActionResult:
        """Parse and execute an action string."""
        parts = action_str.strip().lower().split()
        if not parts:
            return ActionResult(False, "No action provided. Please specify an action.", action_str)

        verb = parts[0]

        try:
            if verb == "move":
                return self._do_move(parts)
            elif verb == "turn":
                return self._do_turn(parts)
            elif verb == "look":
                return self._do_look()
            elif verb == "pickup":
                return self._do_pickup(parts)
            elif verb == "use":
                return self._do_use(parts)
            elif verb == "interact":
                return self._do_interact(parts)
            elif verb == "wait":
                return ActionResult(True, "You wait. Nothing happens.", action_str)
            else:
                return ActionResult(
                    False,
                    f"Unknown action '{verb}'. Valid actions: move, turn, look, pickup, use, interact, wait.",
                    action_str,
                )
        except Exception as e:
            return ActionResult(False, f"Error executing action: {str(e)}", action_str)

    def _do_move(self, parts: list[str]) -> ActionResult:
        """Execute a move action."""
        if len(parts) < 2:
            return ActionResult(False, "Move where? Specify a direction: move north/south/east/west.")

        dir_str = parts[1]
        try:
            direction = Direction(dir_str)
        except ValueError:
            return ActionResult(False, f"Invalid direction '{dir_str}'. Use: north, south, east, west.")

        dx, dy = direction.delta
        new_x, new_y = self.agent.x + dx, self.agent.y + dy

        if not self.world.is_walkable(new_x, new_y):
            # Check if it's a locked door for a more helpful message
            entities_there = self.world.get_entities_at(new_x, new_y)
            for e in entities_there:
                if isinstance(e, Door) and e.locked:
                    return ActionResult(
                        False,
                        f"You can't move {dir_str} — the {e.name} is locked. You need the {e.color.value} key.",
                    )
            tile = self.world.get_tile(new_x, new_y)
            return ActionResult(False, f"You can't move {dir_str} — there's a {tile.description} blocking you.")

        self.agent.move(direction)
        room = self.world.get_room_at(self.agent.x, self.agent.y)
        room_msg = f" You are now in the {room.name}." if room else ""
        return ActionResult(True, f"You move {dir_str}.{room_msg}")

    def _do_turn(self, parts: list[str]) -> ActionResult:
        """Execute a turn action."""
        if len(parts) < 2:
            return ActionResult(False, "Turn which way? Specify: turn north/south/east/west.")

        dir_str = parts[1]
        try:
            direction = Direction(dir_str)
        except ValueError:
            return ActionResult(False, f"Invalid direction '{dir_str}'. Use: north, south, east, west.")

        self.agent.turn(direction)
        return ActionResult(True, f"You turn to face {dir_str}.")

    def _do_look(self) -> ActionResult:
        """Execute a look action — gives detailed observation."""
        room = self.world.get_room_at(self.agent.x, self.agent.y)
        visible = self.world.get_visible_entities(self.agent.x, self.agent.y, radius=5)

        lines = [f"You look around from position ({self.agent.x}, {self.agent.y})."]
        if room:
            lines.append(f"You are in the {room.name}.")
        lines.append(f"You are facing {self.agent.facing.value}.")

        if visible:
            lines.append("You can see:")
            for entity, dx, dy in visible:
                direction = self._offset_to_description(dx, dy)
                extra = ""
                if isinstance(entity, Door):
                    extra = " (locked)" if entity.locked else " (unlocked)"
                lines.append(f"  - {entity.name}{extra} {direction}")
        else:
            lines.append("You don't see anything notable nearby.")

        return ActionResult(True, "\n".join(lines))

    def _do_pickup(self, parts: list[str]) -> ActionResult:
        """Execute a pickup action."""
        if len(parts) < 2:
            return ActionResult(False, "Pick up what? Specify: pickup <item name>.")

        item_name = " ".join(parts[1:])

        # Check entities at current position and adjacent
        for entity in self.world.get_entities_at(self.agent.x, self.agent.y):
            if entity.name.lower() == item_name.lower():
                return self._pickup_entity(entity)

        for entity, direction in self.world.get_adjacent_entities(self.agent.x, self.agent.y):
            if entity.name.lower() == item_name.lower():
                return self._pickup_entity(entity)

        return ActionResult(False, f"There's no '{item_name}' nearby to pick up.")

    def _pickup_entity(self, entity: Entity) -> ActionResult:
        """Actually pick up an entity."""
        if entity.entity_type == EntityType.KEY:
            self.agent.add_to_inventory(entity)
            self.world.remove_entity(entity)
            return ActionResult(True, f"You pick up the {entity.name}.")
        elif entity.entity_type == EntityType.GEM:
            entity.collected = True
            self.agent.add_to_inventory(entity)
            self.world.remove_entity(entity)
            return ActionResult(True, entity.collect())
        elif entity.entity_type == EntityType.WEAPON:
            entity.collect()
            self.agent.add_to_inventory(entity)
            self.world.remove_entity(entity)
            return ActionResult(True, f"You pick up the {entity.name}. You feel ready for battle.")
        else:
            return ActionResult(False, f"You can't pick up the {entity.name}.")

    def _do_use(self, parts: list[str]) -> ActionResult:
        """Execute a use action: 'use <item> on <target>'."""
        full_text = " ".join(parts[1:])
        if " on " not in full_text:
            return ActionResult(
                False,
                "Use what on what? Format: use <item> on <target>. Example: use blue key on blue door.",
            )

        item_name, target_name = full_text.split(" on ", 1)
        item_name = item_name.strip()
        target_name = target_name.strip()

        # Check inventory for item
        if not self.agent.has_item(item_name):
            return ActionResult(False, f"You don't have '{item_name}' in your inventory.")

        # Check for target in adjacent or current tiles
        target = None
        for entity, direction in self.world.get_adjacent_entities(self.agent.x, self.agent.y):
            if entity.name.lower() == target_name.lower():
                target = entity
                break
        
        if not target:
            for entity in self.world.get_entities_at(self.agent.x, self.agent.y):
                if entity.name.lower() == target_name.lower():
                    target = entity
                    break

        if not target:
            return ActionResult(False, f"There's no '{target_name}' nearby.")

        # Handle key-on-door interaction
        if isinstance(target, Door) and target.locked:
            if self.agent.has_key_for_color(target.color):
                key = self.agent.get_key_for_color(target.color)
                msg = target.unlock()
                return ActionResult(True, msg)
            else:
                return ActionResult(
                    False,
                    f"The {item_name} doesn't fit the {target.name}. You need a {target.color.value} key.",
                )

        return ActionResult(False, f"You can't use {item_name} on {target.name}.")

    def _do_interact(self, parts: list[str]) -> ActionResult:
        """Execute an interact action."""
        if len(parts) < 2:
            return ActionResult(False, "Interact with what? Specify: interact <target>.")

        target_name = " ".join(parts[1:])

        # Check adjacent and current entities
        all_nearby = []
        for entity, direction in self.world.get_adjacent_entities(self.agent.x, self.agent.y):
            all_nearby.append(entity)
        for entity in self.world.get_entities_at(self.agent.x, self.agent.y):
            all_nearby.append(entity)

        for entity in all_nearby:
            if entity.name.lower() == target_name.lower():
                if isinstance(entity, Sign):
                    return ActionResult(True, entity.read())
            
                # Add other interactive entity handlers here if needed

                elif isinstance(entity, Door):
                    if entity.locked:
                        return ActionResult(
                            False,
                            f"The {entity.name} is locked. You need the {entity.color.value} key to open it.",
                        )
                    else:
                        return ActionResult(True, f"The {entity.name} is already open. You can walk through.")
                else:
                    return ActionResult(False, f"You can't interact with the {entity.name}.")

        return ActionResult(False, f"There's no '{target_name}' nearby to interact with.")



    def _get_valid_actions(self) -> list[str]:
        """Get all currently valid actions the agent can take."""
        actions = []

        # Movement — check all 4 directions
        for direction in Direction:
            dx, dy = direction.delta
            nx, ny = self.agent.x + dx, self.agent.y + dy
            if self.world.is_walkable(nx, ny):
                actions.append(f"move {direction.value}")

        # Turn — always available
        for direction in Direction:
            if direction != self.agent.facing:
                actions.append(f"turn {direction.value}")

        # Look — always available
        actions.append("look")

        # Pickup — check for pickupable entities at current and adjacent positions
        pickupable_types = {EntityType.KEY, EntityType.GEM, EntityType.WEAPON}
        for entity in self.world.get_entities_at(self.agent.x, self.agent.y):
            if entity.entity_type in pickupable_types:
                actions.append(f"pickup {entity.name}")
        for entity, direction in self.world.get_adjacent_entities(self.agent.x, self.agent.y):
            if entity.entity_type in pickupable_types:
                actions.append(f"pickup {entity.name}")

        # Use — check for usable items in inventory on adjacent targets
        for inv_item in self.agent.inventory:
            if inv_item.entity_type == EntityType.KEY:
                for entity, direction in self.world.get_adjacent_entities(self.agent.x, self.agent.y):
                    if isinstance(entity, Door) and entity.locked and entity.color == inv_item.color:
                        actions.append(f"use {inv_item.name} on {entity.name}")

        # Interact        # Discover interactables (signs, doors, etc.)
        interactable_types = {EntityType.SIGN, EntityType.DOOR}
        for entity, direction in self.world.get_adjacent_entities(self.agent.x, self.agent.y):
            if entity.entity_type in interactable_types:
                actions.append(f"interact {entity.name}")
        for entity in self.world.get_entities_at(self.agent.x, self.agent.y):
            if entity.entity_type in interactable_types:
                actions.append(f"interact {entity.name}")

        # Throw - if agent has potion
        if any(e.entity_type == EntityType.POTION for e in self.agent.inventory):
            for direction in Direction:
                actions.append(f"throw potion {direction.value}")

        # Wait — always available
        actions.append("wait")

        return actions

    def _offset_to_description(self, dx: int, dy: int) -> str:
        """Convert a relative offset to a human-readable direction and distance."""
        distance = max(abs(dx), abs(dy))
        
        parts = []
        if dy < 0:
            parts.append("north")
        elif dy > 0:
            parts.append("south")
        if dx > 0:
            parts.append("east")
        elif dx < 0:
            parts.append("west")

        direction = "-".join(parts) if parts else "here"
        tiles = "tile" if distance == 1 else "tiles"
        return f"({distance} {tiles} {direction})"

    def get_full_state(self) -> dict:
        """Get full serialized state for visualization."""
        full_grid = self.world.to_full_grid(self.agent.x, self.agent.y)
        return {
            "world": self.world.to_dict(),
            "agent": self.agent.to_dict(),
            "task": {
                "id": self.task.task_id,
                "description": self.task.description,
                "progress": self.task.check_completion(self.world, self.agent).to_dict(),
            },
            "full_grid": full_grid,
            "step": self.step_count,
            "max_steps": self.task.max_steps,
            "done": self.done,
        }

    def _tick_monsters(self) -> None:
        """Move all monsters."""
        from engine.entities import Monster
        monsters = [e for e in self.world.entities if isinstance(e, Monster) and e.alive]
        for monster in monsters:
            if monster.stun_duration > 0:
                monster.stun_duration -= 1
                continue
            
            # Random roam (blind)
            valid_moves = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = monster.x + dx, monster.y + dy
                if self.world.is_walkable(nx, ny):
                    valid_moves.append((nx, ny))
            if valid_moves:
                monster.x, monster.y = random.choice(valid_moves)

    def _resolve_combat(self) -> tuple[str, bool]:
        """Check for adjacency to monsters and resolve combat."""
        msg = ""
        agent_died = False
        monsters = [e for e in self.world.entities if isinstance(e, Monster) and e.alive]
        for monster in monsters:
            dist = abs(monster.x - self.agent.x) + abs(monster.y - self.agent.y)
            if dist <= 1:
                has_weapon = any(isinstance(i, Weapon) for i in self.agent.inventory)
                if has_weapon:
                    msg += f" {monster.die()}"
                    self.world.remove_entity(monster)
                elif monster.stun_duration > 0:
                    pass  # The monster is stunned, so it does not attack
                else:
                    msg += f" The {monster.name} caught you! You died."
                    agent_died = True
                    break
        return msg.strip(), agent_died
