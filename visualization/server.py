"""Visualization Server — FastAPI server with WebSocket for real-time agent viewing.

Serves the web frontend and pushes live agent state updates via WebSocket
so you can watch the agent think and act in real-time.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.environment import Environment
from harness.llm_agent import LLMAgent, AnthropicProvider, OpenAIProvider, GeminiProvider, CerebrasProvider, save_run_log


app = FastAPI(title="LLM Agent Visualizer")

# Serve static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global state for active connections and run
active_connections: list[WebSocket] = []
current_run: Optional[dict] = None


@app.get("/")
async def root():
    """Serve the main visualization page."""
    return FileResponse(str(static_dir / "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Receive messages from client (e.g., start commands)
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "start_run":
                task_id = message.get("task_id", "navigate")
                provider_name = message.get("provider", "gemini")
                model = message.get("model", "")
                api_key = message.get("api_key", "")
                
                if provider_name == "replay":
                    asyncio.create_task(replay_agent_async(websocket, task_id))
                else:
                    # Run agent in background
                    asyncio.create_task(
                        run_agent_async(websocket, task_id, provider_name, model, api_key)
                    )

    except WebSocketDisconnect:
        active_connections.remove(websocket)


async def run_agent_async(
    websocket: WebSocket,
    task_id: str,
    provider_name: str,
    model: str = "",
    api_key: str = "",
):
    """Run the agent asynchronously and push updates via WebSocket."""
    try:
        # Create provider
        if provider_name == "openai":
            provider = OpenAIProvider(
                model=model or "gpt-4o-mini",
                api_key=api_key,
            )
        elif provider_name == "anthropic":
            provider = AnthropicProvider(
                model=model or "claude-4-8-opus-latest",
                api_key=api_key,
            )
        elif provider_name == "kimi":
            provider = OpenAIProvider(
                model=model or "kimi-k2.6",
                api_key=api_key,
                base_url="https://api.moonshot.ai/v1",
                max_tokens=2048,
            )
        elif provider_name == "cerebras":
            provider = CerebrasProvider(
                model=model or "llama3.1-70b",
                api_key=api_key
            )
        else:
            provider = GeminiProvider(
                model=model or "gemini-3.5-flash",
                api_key=api_key,
            )

        env = Environment()

        main_loop = asyncio.get_running_loop()

        # Callback to send updates via WebSocket
        async def send_update(step_data, full_state):
            try:
                await websocket.send_json({
                    "type": "step_update",
                    "step_data": step_data,
                    "full_state": full_state,
                })
            except Exception:
                pass

        def on_step_sync(step_data, full_state):
            """Synchronous wrapper for the async send."""
            asyncio.run_coroutine_threadsafe(send_update(step_data, full_state), main_loop)

        agent = LLMAgent(provider=provider, verbose=True, on_step=on_step_sync)

        # Send initial state
        obs = env.reset(task_id)
        initial_state = env.get_full_state()
        await websocket.send_json({
            "type": "run_started",
            "task_id": task_id,
            "task_description": env.task.description,
            "initial_state": initial_state,
        })

        # Run agent in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(None, agent.run, env, task_id)

        # Save log
        log_path = save_run_log(summary)

        # Send completion
        await websocket.send_json({
            "type": "run_complete",
            "summary": {
                "completed": summary["completed"],
                "steps_taken": summary["steps_taken"],
                "max_steps": summary["max_steps"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "tokens_used": summary["tokens_used"],
                "completion_message": summary["completion_message"],
                "log_path": log_path,
            },
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("RUN_AGENT ERROR:", tb)
        await websocket.send_json({
            "type": "error",
            "message": f"Error: {e}\n\nTraceback:\n{tb}",
        })

async def replay_agent_async(websocket: WebSocket, task_id: str):
    import glob, copy
    logs = sorted(glob.glob(f"logs/run_{task_id}_*.json"))
    if not logs:
        await websocket.send_json({"type": "error", "message": "No logs found for playback."})
        return
        
    try:
        with open(logs[-1], "r", encoding="utf-8") as f:
            log_data = json.load(f)
            
        env = Environment()
        seed = log_data.get("seed")
        env.reset(task_id, seed=seed)
        
        # Sync randomly spawned items to their true positions from the log (fallback for old logs without seed)
        if seed is None and log_data.get("log"):
            import re
            for step_record in log_data["log"]:
                obs = step_record.get("observation", "")
                for line in obs.split('\n'):
                    # Extract locations like: "  • sword at (2, 6)."
                    m = re.search(r"^\s*•\s+(.*?)\s+at\s+\((\d+),\s*(\d+)\)", line)
                    if m:
                        item_name = m.group(1).lower().strip()
                        x, y = int(m.group(2)), int(m.group(3))
                        
                        # Check if we already synced this specific entity to this exact location
                        already_synced_here = any(
                            e.name.lower() == item_name and getattr(e, "_synced", False) and e.x == x and e.y == y
                            for e in env.world.entities
                        )
                        
                        if not already_synced_here:
                            # Find an unsynced entity of this name and sync it to this newly discovered location
                            for e in env.world.entities:
                                if e.name.lower() == item_name and not getattr(e, "_synced", False):
                                    e.x = x
                                    e.y = y
                                    e._synced = True
                                    break

        initial_state = env.get_full_state()
        
        await websocket.send_json({
            "type": "run_started",
            "task_id": task_id,
            "task_description": log_data.get("task_description", env.task.description),
            "initial_state": initial_state,
        })
        
        # Track exact counts and locations of collected items to hide them permanently
        from collections import Counter
        permanently_hidden = set()
        previous_inv_counts = Counter()
        
        for step in log_data["log"]:
            await asyncio.sleep(0.5)
            
            # Make monsters move randomly for visual effect during replay
            env._tick_monsters()
            current_state = env.get_full_state()
            
            # Update agent position from log
            agent_pos = step.get("agent_position", {"x": 0, "y": 0})
            if isinstance(agent_pos, dict):
                ax, ay = agent_pos["x"], agent_pos["y"]
            else:
                ax, ay = agent_pos
            current_state["agent"]["position"]["x"] = ax
            current_state["agent"]["position"]["y"] = ay
            current_state["step"] = step.get("step", 0)
            
            # Detect newly collected items this step
            current_inv_counts = Counter(i.lower() for i in step.get("inventory", []))
            
            for item_name, count in current_inv_counts.items():
                newly_collected = count - previous_inv_counts.get(item_name, 0)
                if newly_collected > 0:
                    # Find the closest visible entities of this name that are not yet hidden
                    available_entities = [
                        e for e in current_state["world"]["entities"]
                        if e.get("name", "").lower() == item_name and (item_name, e.get("x", 0), e.get("y", 0)) not in permanently_hidden
                    ]
                    available_entities.sort(key=lambda e: abs(e.get("x", 0) - ax) + abs(e.get("y", 0) - ay))
                    
                    # Mark the closest ones as permanently hidden
                    for i in range(min(newly_collected, len(available_entities))):
                        e = available_entities[i]
                        permanently_hidden.add((item_name, e.get("x", 0), e.get("y", 0)))
                        
            previous_inv_counts = current_inv_counts
            
            # Filter out permanently hidden entities from the view
            current_state["world"]["entities"] = [
                e for e in current_state["world"]["entities"]
                if (e.get("name", "").lower(), e.get("x", 0), e.get("y", 0)) not in permanently_hidden
            ]
                    
            await websocket.send_json({
                "type": "step_update",
                "step_data": {
                    "action": step.get("action", ""),
                    "result": step.get("result", ""),
                    "success": step.get("success", True),
                    "step": step.get("step", 0),
                    "thinking": step.get("thinking", ""),
                    "task_progress": step.get("task_progress", {}),
                    "inventory": step.get("inventory", []),
                },
                "full_state": current_state,
            })
            
        # Send completion with properly shaped summary
        await websocket.send_json({
            "type": "run_complete",
            "summary": {
                "completed": log_data.get("completed", False),
                "steps_taken": log_data.get("steps_taken", 0),
                "max_steps": log_data.get("max_steps", 300),
                "elapsed_seconds": log_data.get("elapsed_seconds", 0),
                "tokens_used": log_data.get("tokens_used", 0),
                "completion_message": log_data.get("completion_message", "Replay complete."),
            }
        })
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("RUN ERROR:", tb)
        await websocket.send_json({"type": "error", "message": f"Error: {e}\n\nTraceback:\n{tb}"})

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the visualization server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
