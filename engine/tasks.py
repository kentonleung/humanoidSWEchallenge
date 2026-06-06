"""Tasks — goal definitions and completion checking for each level."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from engine.entities import EntityType, Gem, GoalMarker
from engine.agent_state import AgentState
from engine.world import World


@dataclass
class TaskProgress:
    """Tracks progress toward completing a task."""
    objectives: dict[str, bool] = field(default_factory=dict)
    completed: bool = False
    message: str = ""

    def update(self, key: str, done: bool) -> None:
        self.objectives[key] = done

    def to_str(self) -> str:
        lines = []
        for name, done in self.objectives.items():
            status = "[PASS]" if done else "[FAIL]"
            lines.append(f"  [{status}] {name}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "objectives": self.objectives,
            "completed": self.completed,
            "message": self.message,
        }


class Task:
    """Base class for goal-directed tasks."""
    
    def __init__(self, task_id: str, description: str, max_steps: int = 50):
        self.task_id = task_id
        self.description = description
        self.max_steps = max_steps

    def check_completion(self, world: World, agent: AgentState) -> TaskProgress:
        """Check whether the task is complete. Override in subclasses."""
        raise NotImplementedError

    def get_hint(self, world: World, agent: AgentState) -> str:
        """Optional hint based on current state."""
        return ""


class NavigateTask(Task):
    """Task 1: Navigate to the red gem."""

    def __init__(self):
        super().__init__(
            task_id="navigate",
            description="Go to the red gem in the East Room and pick it up.",
            max_steps=100,
        )

    def check_completion(self, world: World, agent: AgentState) -> TaskProgress:
        progress = TaskProgress()

        # Check if the agent has picked up the red gem
        has_gem = agent.has_item("red gem")
        
        # Check if the gem has been collected from the world
        gem_collected = True
        for entity in world.entities:
            if isinstance(entity, Gem) and entity.name == "red gem":
                gem_collected = entity.collected
                break

        # The agent either has it in inventory or is standing on it
        in_east_room = False
        room = world.get_room_at(agent.x, agent.y)
        if room and room.name == "East Room":
            in_east_room = True

        progress.update("Reach the East Room", in_east_room)
        progress.update("Pick up the red gem", has_gem)

        progress.completed = has_gem
        if progress.completed:
            progress.message = "Congratulations! You found and collected the red gem!"
        return progress


class KeyDoorTask(Task):
    """Task 2: Find the blue key and unlock the blue door."""

    def __init__(self):
        super().__init__(
            task_id="key_door",
            description="Find the blue key, unlock the blue door, and reach the Dungeon Exit.",
            max_steps=200,
        )

    def check_completion(self, world: World, agent: AgentState) -> TaskProgress:
        progress = TaskProgress()

        # Check if blue key has been found
        has_key = agent.has_item("blue key")
        key_exists_in_world = any(
            e.name == "blue key" for e in world.entities
        )
        key_found = has_key or not key_exists_in_world

        # Check if blue door is unlocked
        door_unlocked = True
        for e in world.entities:
            if e.name == "blue door":
                door_unlocked = not e.locked
                break

        # Check if agent reached the goal
        at_goal = any(
            isinstance(e, GoalMarker) and e.x == agent.x and e.y == agent.y
            for e in world.entities
        )

        progress.update("Find the blue key", key_found)
        progress.update("Unlock the blue door", door_unlocked)
        progress.update("Reach the Dungeon Exit", at_goal)

        progress.completed = at_goal
        if progress.completed:
            progress.message = "Excellent! You found the key, unlocked the door, and reached the goal!"
        return progress


class DungeonEscapeTask(Task):
    """Task 3: Collect all gems and escape the dungeon."""

    def __init__(self):
        super().__init__(
            task_id="dungeon_escape",
            description=(
                "Explore the dungeon, collect all 3 gems (red, blue, green), "
                "and reach the Dungeon Exit to escape. You may need to find keys "
                "to unlock doors blocking your path."
            ),
            max_steps=300,
        )

    def check_completion(self, world: World, agent: AgentState) -> TaskProgress:
        progress = TaskProgress()

        # Check each gem
        gem_names = ["red gem", "blue gem", "green gem"]
        all_gems = True
        for gem_name in gem_names:
            has_it = agent.has_item(gem_name)
            progress.update(f"Collect {gem_name}", has_it)
            if not has_it:
                all_gems = False

        # Check if at goal
        at_goal = any(
            isinstance(e, GoalMarker) and e.x == agent.x and e.y == agent.y
            for e in world.entities
        )
        progress.update("Reach the exit", at_goal)

        progress.completed = all_gems and at_goal
        if progress.completed:
            progress.message = "Amazing! You collected all gems and escaped the dungeon!"
        elif all_gems and not at_goal:
            progress.message = "All gems collected! Now find the exit."
        return progress


# Registry mapping task IDs to task instances
TASKS = {
    "navigate": NavigateTask,
    "key_door": KeyDoorTask,
    "dungeon_escape": DungeonEscapeTask,
}
