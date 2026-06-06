"""Unit tests for the BFS pathfinder."""

from engine.world import create_level_1, create_level_2, create_level_3
from harness.pathfinder import find_shortest_path


def test_level_1_path():
    print("Testing Level 1 (Navigate) pathfinding...")
    world, sx, sy = create_level_1()
    
    # Red gem is at (15, 3). Agent is at (2, 5).
    gem_pos = (15, 3)
    path = find_shortest_path(world, (sx, sy), gem_pos)
    assert path is not None, "Should find a path to the red gem"
    print(f"  Walkable path to gem: {path} ({len(path)} steps)")
    # Path should start with move east and end near/at (15, 3)
    assert path[0] == "move east"


def test_level_2_path():
    print("Testing Level 2 (Key & Door) pathfinding...")
    world, sx, sy = create_level_2()
    
    # Blue key is at (12, 8). Blue door is at (16, 6). Goal is at (20, 6).
    key_pos = (12, 8)
    door_pos = (16, 6)
    goal_pos = (20, 6)
    
    # Check path to key (walkable)
    path_to_key = find_shortest_path(world, (sx, sy), key_pos)
    assert path_to_key is not None
    print(f"  Walkable path to key: {path_to_key} ({len(path_to_key)} steps)")
    
    # Check path to goal (blocked by locked blue door)
    path_to_goal_blocked = find_shortest_path(world, (sx, sy), goal_pos, ignore_locked_doors=False)
    assert path_to_goal_blocked is None, "Should be blocked by locked door"
    print("  [PASS] Walkable path to goal is correctly blocked")
    
    # Check path to goal (optimistic, ignoring locked doors)
    path_to_goal_optimistic = find_shortest_path(world, (sx, sy), goal_pos, ignore_locked_doors=True)
    assert path_to_goal_optimistic is not None
    print(f"  Optimistic path to goal: {path_to_goal_optimistic} ({len(path_to_goal_optimistic)} steps)")


def test_level_3_path():
    print("Testing Level 3 (Dungeon Escape) pathfinding...")
    world, sx, sy = create_level_3()
    
    # Red gem is at (4, 3) (Entry Hall). Walkable!
    red_gem = (4, 3)
    path_to_red = find_shortest_path(world, (sx, sy), red_gem)
    assert path_to_red is not None
    print(f"  Walkable path to red gem: {path_to_red} ({len(path_to_red)} steps)")
    
    # Green gem at (6, 12) (Catacombs). Accessible via open passage at (5, 8).
    green_gem = (6, 12)
    path_to_green = find_shortest_path(world, (sx, sy), green_gem)
    assert path_to_green is not None
    print(f"  Walkable path to green gem: {path_to_green} ({len(path_to_green)} steps)")
    
    # Blue gem at (22, 3) (Armory). Blocked by locked red door at (14, 4) if we go directly,
    # or locked green door at (14, 12) if we go from Catacombs.
    blue_gem = (22, 3)
    path_to_blue_blocked = find_shortest_path(world, (sx, sy), blue_gem, ignore_locked_doors=False)
    assert path_to_blue_blocked is None, "Should be blocked without keys/doors unlocked"
    print("  [PASS] Walkable path to blue gem is correctly blocked")
    
    path_to_blue_optimistic = find_shortest_path(world, (sx, sy), blue_gem, ignore_locked_doors=True)
    assert path_to_blue_optimistic is not None
    print(f"  Optimistic path to blue gem: {path_to_blue_optimistic} ({len(path_to_blue_optimistic)} steps)")


if __name__ == "__main__":
    test_level_1_path()
    test_level_2_path()
    test_level_3_path()
    print("\n[PASS] All pathfinder tests passed!")
