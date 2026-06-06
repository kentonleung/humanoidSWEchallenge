"""Quick test script to verify all game mechanics work correctly."""

from engine.environment import Environment
from harness.observations import format_observation


def test_navigate():
    """Test level 1: Navigate to the red gem."""
    print("=== Test: Navigate ===")
    env = Environment()
    env.reset("navigate")

    # Move east to the passage (at y=5, wall passage at x=9)
    for _ in range(7):
        env.step("move east")

    # Now at x=9, y=5 (the passage). Continue east
    for _ in range(6):
        env.step("move east")

    # Move north to the gem at (15, 3)
    env.step("move north")
    env.step("move north")

    # Pickup the gem
    obs, done, info = env.step("pickup red gem")
    progress = env.task.check_completion(env.world, env.agent)
    assert progress.completed, f"Navigate task should be complete! Agent at ({env.agent.x}, {env.agent.y})"
    print(f"  [PASS] Navigate PASSED — completed in {env.step_count} steps")


def test_key_door():
    """Test level 2: Find key, unlock door, reach goal."""
    print("=== Test: Key & Door ===")
    env = Environment()
    env.reset("key_door")

    # Agent starts at (4, 8). 
    # First, navigate to the key at (12, 8)
    # Go through passage at (8, 3): north 5 to y=3, east 4 to x=8, continue east
    for _ in range(5):
        env.step("move north")   # (4,3)
    for _ in range(4):
        env.step("move east")    # (8,3), through passage
    for _ in range(4):
        env.step("move east")    # (12,3)
    for _ in range(5):
        env.step("move south")   # (12,8)

    obs, done, info = env.step("pickup blue key")
    assert env.agent.has_item("blue key"), "Should have blue key"
    print(f"  [PASS] Key picked up at step {env.step_count}, agent at ({env.agent.x}, {env.agent.y})")

    # Navigate to door at (16, 6): north 2 to y=6, east to approach door
    for _ in range(2):
        env.step("move north")   # (12,6)
    for _ in range(3):
        env.step("move east")    # (15,6), adjacent to door at (16,6)

    # Use key on adjacent door
    obs, done, info = env.step("use blue key on blue door")
    print(f"  Use key result: {info['result']}")

    # Walk through the now-unlocked door to goal at (20, 6)
    for _ in range(5):
        env.step("move east")    # Through door to (20,6)

    progress = env.task.check_completion(env.world, env.agent)
    print(f"  Agent at ({env.agent.x}, {env.agent.y}), Goal at (20, 6)")
    assert progress.completed, f"Key-door task should be complete! Agent at ({env.agent.x}, {env.agent.y})"
    print(f"  [PASS] Key & Door PASSED — completed in {env.step_count} steps")


def test_action_validation():
    """Test that invalid actions are properly handled."""
    print("=== Test: Action Validation ===")
    env = Environment()
    env.reset("navigate")

    # Try to move into a wall - move to edge first
    env.step("move west")  # (2,5) -> (1,5)
    obs, _, info = env.step("move west")  # (1,5) -> (0,5) is wall
    assert not info["success"], "Moving into wall should fail"

    # Try to pick up something that doesn't exist
    obs, _, info = env.step("pickup gold gem")
    assert not info["success"], "Picking up non-existent item should fail"

    # Try invalid action
    obs, _, info = env.step("fly")
    assert not info["success"], "Invalid action should fail"

    # Valid action
    obs, _, info = env.step("look")
    assert info["success"], "Look should succeed"

    print("  [PASS] Action Validation PASSED")


def test_observation_format():
    """Test that observations are properly formatted."""
    print("=== Test: Observation Format ===")
    env = Environment()
    obs = env.reset("navigate")
    text = format_observation(obs)

    assert "[Location]" in text
    assert "[Vision" in text
    assert "[Inventory]" in text
    assert "[Valid Actions]" in text
    assert "[Task]" in text
    assert "@" in text  # Agent symbol in grid

    print("  [PASS] Observation Format PASSED")


if __name__ == "__main__":
    test_navigate()
    test_key_door()
    test_action_validation()
    test_observation_format()
    print("\n[PASS] All tests passed!")
