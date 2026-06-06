# LLM Agent in a Virtual World

This project explores the design of an intelligent agent **harness** — the critical interface layer between a Large Language Model and a virtual world. It tackles core architectural challenges like dual-modality observation representation, strict action-space constraints, structured chain-of-thought parsing, and infinite loop detection.

While the agent's current testbed is a custom 2D dungeon, the primary focus is on building a robust, predictable pipeline for LLMs to perceive, reason, and act within a structured environment. The project includes a real-time web visualizer to watch the agent's reasoning loop in action.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/License-MIT-green)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Your API Key

```bash
# Cerebras (zai-glm-4.7) — Default & Recommended (Free, Generous Tokens)
export CEREBRAS_API_KEY="csk-..."
```

### 3. Run the Agent

```bash
# Run a specific task (console output)
python main.py --task navigate           # Easy: find the red gem
python main.py --task key_door           # Medium: find key & unlock door
python main.py --task dungeon_escape     # Hard: collect all gems & escape

# Run all tasks
python main.py --all

# Launch the web visualizer (recommended!)
python main.py --web
# Then open http://localhost:8000 in your browser
```

---

## Tasks

| Task | Difficulty | Goal | Skills Tested |
|------|-----------|------|--------------|
| **Navigate** | Easy | Go to the red gem in the East Room | Spatial reasoning, movement |
| **Key & Door** | Medium | Find the blue key, unlock the door, reach the goal | Multi-step planning, item use |
| **Dungeon Escape** | Hard | Collect all 3 gems across 4 rooms, find the exit | Exploration, memory, complex planning |

---

## Architecture

```
llmChallenge/
├── engine/               # Core game engine
│   ├── world.py          # 2D grid, tiles, rooms, level layouts
│   ├── entities.py       # Keys, doors, gems, switches, signs
│   ├── agent_state.py    # Agent position, inventory, facing
│   ├── environment.py    # Gym-style env: reset/step/observe
│   └── tasks.py          # Goal definitions & completion checks
│
├── harness/              # The agent harness (core of the challenge)
│   ├── observations.py   # Observation → structured text for LLM
│   ├── actions.py        # Action space definition & validation
│   ├── prompts.py        # System prompt & response parsing
│   └── llm_agent.py      # LLM reasoning loop
│
├── visualization/        # Real-time web viewer
│   ├── server.py         # FastAPI + WebSocket server
│   └── static/           # HTML/CSS/JS frontend
│
├── main.py               # CLI entry point
└── logs/                 # Auto-generated run logs (JSON)
```

### Separation of Concerns

The architecture is deliberately modular — each component is independent and swappable:

- **Swap the LLM**: The architecture is modular. While configured for Cerebras by default, you can easily plug in other providers (like OpenAI or Anthropic) by implementing the `LLMProvider` interface, as demonstrated in `harness/llm_agent.py`.
- **Change the world**: Modify `world.py` without touching the harness
- **Adjust observations**: Edit `observations.py` without changing the agent loop
- **Add actions**: Extend `actions.py` and `environment.py` independently

---

## Design Notes

### Observation Representation

> *"What does the agent need to know, and how do you tell it?"*

The observation format provides **dual-modality perception** — a visual grid AND semantic text, similar to how a real robot might combine camera feeds with processed sensor data:

#### 1. Local Vision Grid (7x7 ASCII)
```
+---------------+
| # # # # # # # |
| # . . . . . # |
| # . . K . . # |
| # . . @ . . # |
| # . . . D . # |
| # # # # # # # |
+---------------+
```
This gives the LLM spatial layout at a glance — where walls, objects, and passages are relative to the agent.

#### 2. Semantic Description
```
[Location] Position (5, 3) in "Main Hall" | Facing north
[Visible Entities]
  • blue key — 2 tile(s) north
  • blue door (locked) — 2 tile(s) east
```
Natural language descriptions provide named entities, distances, and states that are easier for LLMs to reason about than pure grid data.

#### 3. Constrained Action List
```
[Valid Actions] move north, move east, pickup blue key, look, wait
```
**This is critical.** By providing only valid actions, I eliminate a huge class of LLM errors (invalid movements, picking up non-existent items). The LLM selects from a menu rather than generating free-form commands.

#### 4. Task Progress
```
[Task]
  [x] Find the blue key
  [ ] Unlock the blue door
  [ ] Reach the goal marker
```
Explicit progress tracking helps the agent maintain focus across many steps.

### Action Space Design

The action space is **discrete and validated**:
- `move <direction>` — N/S/E/W, one tile at a time
- `pickup <item>` — pick up adjacent item
- `use <item> on <target>` — use inventory item on nearby object
- `interact <target>` — read signs, flip switches
- `look` / `turn` / `wait` — perception and waiting

**Why discrete?** LLMs are text generators. Discrete, text-based actions with clear formats maximize parse reliability. Every action is validated before execution — invalid actions return error messages that help the LLM self-correct.

### LLM Integration

The agent uses a **structured chain-of-thought** format:
```
THINK: I can see the blue key 2 tiles north. I need it to unlock the door. I should move north.
ACTION: move north
```

**Key design decisions:**
- **Sliding window memory**: Last 8 actions/results prevent loops
- **Error recovery**: Invalid actions get error messages; the agent adapts
- **Response parsing**: Lenient parser handles markdown formatting, missing prefixes, etc.
- **Step limit**: Prevents infinite loops (configurable per task)

### What Worked

1. **Constrained action lists** dramatically improved reliability — the agent almost never generates invalid actions when given a clear menu.
2. **Dual observation modes** (local tactical grid + global semantic map) gave better results than either alone, allowing both detailed interaction and broad navigation.
3. **Stalemate / Loop Detection** in the reasoning loop successfully warns the agent if it's getting stuck repeating actions, prompting it to break out and explore.
4. **Explicit task progress** kept the agent focused on uncompleted objectives.
5. **Error messages** for invalid actions led to quick self-correction.

### What Didn't Work

1. **Active Chasing Monsters**: I originally tried making monsters actively chase the agent. However, LLMs struggle heavily with dynamic spatial-temporal lookahead (adversarial pathfinding on a 2D grid). The agent would get caught constantly, forcing me to downgrade monsters to "blind roamers" (acting as dynamic environmental hazards instead).

### What Could Be Improved

1. **Long-horizon memory**: For the hardest task, the sliding window sometimes causes the agent to revisit areas. A dynamic persistent spatial memory map would help track explored vs unexplored regions.
2. **Multi-turn conversations**: Currently each turn is a fresh API call for most models. Using native conversation history and tool-use directly could improve coherence.
3. **Complex Entities**: Expanding the interactions with pushable blocks, traps, or dynamic lighting.

---

## Example Output

### Task: Navigate (Find the Red Gem)

```
===============================================
  Task: navigate
  Goal: Go to the red gem in the East Room and pick it up.
  Max Steps: 30
  World: Twin Rooms
===============================================

  -- Step 1
  | Think: I need to find the red gem in the East Room. I'm in the West Room.
  |        I should head east through the passage.
  | Action: move east
  | Result: You move east.

  -- Step 2
  | Think: I need to keep moving east to reach the passage that connects
  |        to the East Room.
  | Action: move east
  | Result: You move east.

  ... (agent navigates through passage)

  -- Step 8
  | Think: I can see the red gem nearby. Let me pick it up.
  | Action: pickup red gem
  | Result: You pick up the red gem. It gleams brightly.
  | Inventory: red gem
  | Completed: Reach the East Room, Pick up the red gem

===============================================
  RUN COMPLETE
===============================================
  Result: SUCCESS
  Steps: 8/30
  Time: 12.4s
  Tokens: 8,432
===============================================
```

Full JSON logs are saved to `logs/` after each run with complete observation-reasoning-action traces.

---

## Web Visualizer

Launch with `python main.py --web` and open `http://localhost:8000`.

The visualizer shows:
- **World View**: Real-time canvas rendering of the grid world
- **Agent Thinking**: The LLM's chain-of-thought reasoning each step
- **Action Log**: History of all actions and results
- **Task Progress**: Live objective tracking
- **Inventory**: Current items

### Replay Demo (Dungeon Escape)

[demo.webm](https://github.com/user-attachments/assets/224be054-3897-4c64-afe0-0cfe6d178d6c)


> **A Note on Video Playback Speed & Rate Limits:**
> If you watch the full replay video, you might notice the agent's actions slow down significantly towards the end of the run. Because this environment uses the free Cerebras API, I am subject to strict requests-per-minute (RPM) and token rate limits. During the long `dungeon_escape` task, the agent exhausted its initial token burst. However, my harness gracefully catches `429 RateLimitError` exceptions and implements an exponential backoff, allowing the agent to slowly regenerate tokens and successfully finish the run without crashing.

---

## Configuration

| Environment Variable | Description |
|---------------------|-------------|
| `CEREBRAS_API_KEY` | Cerebras API key (default provider) |

| CLI Flag | Description | Default |
|----------|-------------|---------|
| `--task` | Task to run | (required) |
| `--provider` | `cerebras` | `cerebras` |
| `--model` | Model name | `zai-glm-4.7` |
| `--web` | Start web visualizer | `false` |
| `--port` | Web server port | `8000` |
| `--all` | Run all tasks | `false` |



## License

MIT
