"""LLM Agent — the main reasoning loop that connects the LLM to the environment.

This is the core harness: it observes state, formats observations, sends them
to the LLM, parses the response, and executes actions in a loop.

Supports multiple LLM providers (Google Gemini, Anthropic Claude, OpenAI GPT)
through a unified interface.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Optional, Callable

from engine.environment import Environment
from harness.observations import format_observation
from harness.prompts import SYSTEM_PROMPT, build_turn_prompt, parse_llm_response
from harness.actions import parse_action
from engine.entities import Sign


# ─── LLM Provider Abstraction ───────────────────────────────────────────────────

class LLMProvider:
    """Abstract base for LLM API providers."""
    
    def __init__(self, model: str = ""):
        self.model = model
        self.total_tokens_used = 0
        self.conversation_history: list[dict] = []

    def chat(self, system_prompt: str, user_prompt: str, use_history: bool = False) -> str:
        """Send a message and get a response. Override in subclasses."""
        raise NotImplementedError

    def reset_conversation(self) -> None:
        """Reset conversation history for stateful multi-turn runs."""
        self.conversation_history = []


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API."""

    def __init__(self, model: str = "claude-4-8-opus-latest", api_key: str = ""):
        super().__init__(model)
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        final_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not final_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
            
        self.client = anthropic.Anthropic(api_key=final_key)

    def chat(self, system_prompt: str, user_prompt: str, use_history: bool = False) -> str:
        if not use_history:
            messages = [{"role": "user", "content": user_prompt}]
        else:
            self.conversation_history.append({"role": "user", "content": user_prompt})
            messages = self.conversation_history

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )
        self.total_tokens_used += response.usage.input_tokens + response.usage.output_tokens
        
        content = response.content[0].text
        if use_history:
            self.conversation_history.append({"role": "assistant", "content": content})
        return content


class OpenAIProvider(LLMProvider):
    """GPT API via the OpenAI SDK."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = "", base_url: str = None, max_tokens: int = 1024):
        super().__init__(model)
        self.max_tokens = max_tokens
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
            
        final_key = api_key or os.getenv("OPENAI_API_KEY")
        if not final_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        
        self.client = OpenAI(api_key=final_key, base_url=base_url)

    def chat(self, system_prompt: str, user_prompt: str, use_history: bool = False) -> str:
        if not use_history:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            if not self.conversation_history:
                self.conversation_history.append({"role": "system", "content": system_prompt})
            self.conversation_history.append({"role": "user", "content": user_prompt})
            messages = self.conversation_history

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        if response.usage:
            self.total_tokens_used += response.usage.total_tokens
            
        content = response.choices[0].message.content
        if use_history:
            self.conversation_history.append({"role": "assistant", "content": content})
        return content


class GeminiProvider(LLMProvider):
    """Google Gemini API via google-genai."""

    def __init__(self, model: str = "gemini-3.5-flash", api_key: str = ""):
        super().__init__(model)
        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")

        final_key = api_key or os.getenv("GEMINI_API_KEY")
        if not final_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        self.client = genai.Client(api_key=final_key)
        self._chat_session = None

    def chat(self, system_prompt: str, user_prompt: str, use_history: bool = False) -> str:
        from google.genai import types
        import asyncio

        # Ensure there is an active event loop in this thread for google-genai SDK
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            import sys
            if sys.platform == 'win32':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)

        if use_history:
            if self._chat_session is None:
                self._chat_session = self.client.chats.create(
                    model=self.model,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=1024,
                        safety_settings=[
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        ]
                    )
                )
            response = self._chat_session.send_message(user_prompt)
            # Fetch usage metadata
            if response.usage_metadata:
                self.total_tokens_used += (
                    response.usage_metadata.prompt_token_count
                    + response.usage_metadata.candidates_token_count
                )
            return response.text
        else:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=1024,
                    safety_settings=[
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]
                ),
            )
            if response.usage_metadata:
                self.total_tokens_used += (
                    response.usage_metadata.prompt_token_count
                    + response.usage_metadata.candidates_token_count
                )
            return response.text

    def reset_conversation(self) -> None:
        super().reset_conversation()
        self._chat_session = None




class CerebrasProvider(LLMProvider):
    """Cerebras API via the OpenAI SDK."""

    def __init__(self, model: str = "gpt-oss-120b", api_key: str = ""):
        super().__init__(model)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        final_key = api_key or os.getenv("CEREBRAS_API_KEY")
        if not final_key:
            raise ValueError("CEREBRAS_API_KEY is not set.")
        
        self.client = OpenAI(
            base_url="https://api.cerebras.ai/v1",
            api_key=final_key,
        )

    def chat(self, system_prompt: str, user_prompt: str, use_history: bool = False) -> str:
        if not use_history:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            if not self.conversation_history:
                self.conversation_history.append({"role": "system", "content": system_prompt})
            self.conversation_history.append({"role": "user", "content": user_prompt})
            
            # Prevent infinite token growth by truncating history
            if len(self.conversation_history) > 11:
                # Keep system prompt [0], and last 10 messages
                self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-10:]
                
            messages = self.conversation_history

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=messages,
            temperature=0.2,
        )
        
        if response.usage:
            self.total_tokens_used += response.usage.total_tokens
            
        content = response.choices[0].message.content
        if use_history:
            self.conversation_history.append({"role": "assistant", "content": content})
        return content



# ─── Agent ───────────────────────────────────────────────────────────────────────

class LLMAgent:
    """The main agent that connects an LLM to the virtual environment.
    
    The agent loop:
      1. Observe the environment
      2. Format observation into structured text
      3. Send to LLM with task context and history
      4. Parse LLM response into thinking + action
      5. Execute action in environment
      6. Repeat until task complete or step limit reached
    """

    def __init__(
        self,
        provider: LLMProvider,
        verbose: bool = True,
        on_step: Optional[Callable] = None,
        use_history: bool = False,
    ):
        """
        Args:
            provider: The LLM provider to use.
            verbose: Whether to print step-by-step output to console.
            on_step: Optional callback called after each step with step data.
            use_history: If True, uses stateful state history on conversational API models.
        """
        self.provider = provider
        self.verbose = verbose
        self.on_step = on_step
        self.use_history = use_history
        self.history: list[dict] = []
        self.full_log: list[dict] = []

    def run(self, env: Environment, task_id: str) -> dict:
        """Run the agent on a specific task.
        
        Args:
            env: The environment instance.
            task_id: Which task to run ('navigate', 'key_door', 'dungeon_escape').
            
        Returns:
            A summary dict with results and full log.
        """
        import asyncio
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        # Reset environment
        observation = env.reset(task_id)
        self.history = []
        self.full_log = []
        self.provider.reset_conversation()

        # Spatial Memory
        recent_coords = []
        known_signs = {}
        seen_items = {}
        blocked_doors = {}  # Persistent memory: {"red door at (x,y)": {"color": "red", "pos": (x,y), "needs": "red key"}}

        task_description = env.task.description
        start_time = time.time()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Task: {task_id}")
            print(f"  Goal: {task_description}")
            print(f"  Max Steps: {env.task.max_steps}")
            print(f"  World: {env.world.name}")
            print(f"{'='*60}\n")

        done = False
        while not done:
            # Calculate loop warnings
            curr_pos = (env.agent.x, env.agent.y)
            recent_coords.append(curr_pos)
            if len(recent_coords) > 15:
                recent_coords.pop(0)

            loop_warning = ""
            visit_count = recent_coords.count(curr_pos)
            if visit_count >= 3:
                current_room = env.world.get_room_at(env.agent.x, env.agent.y)
                room_name = current_room.name if current_room else "Unknown area"
                loop_warning = (
                    f"Warning: You have visited position {curr_pos} in \"{room_name}\" "
                    f"{visit_count} times in your recent history. You may be stuck in a "
                    f"movement loop. Consider exploring different rooms or choosing actions "
                    f"to explore uncharted areas on your 'Global GPS Map' to make progress."
                )


            from engine.entities import Gem, Key, Weapon, Door
            for entity, dx, dy in observation.visible_entities:
                if isinstance(entity, (Gem, Key, Weapon)):
                    global_x = env.agent.x + dx
                    global_y = env.agent.y + dy
                    seen_items[entity.name] = (global_x, global_y)
                elif isinstance(entity, Door) and entity.locked:
                    global_x = env.agent.x + dx
                    global_y = env.agent.y + dy
                    seen_items[f"{entity.name} (locked)"] = (global_x, global_y)
                    # Persistently remember locked doors
                    door_key = f"{entity.color.value} door at ({global_x}, {global_y})"
                    if door_key not in blocked_doors:
                        blocked_doors[door_key] = {
                            "color": entity.color.value,
                            "pos": (global_x, global_y),
                            "needs": f"{entity.color.value} key",
                        }
            
            # Also detect locked door failures from last action result
            if observation.last_action_result and not observation.last_action_result.success:
                result_msg = observation.last_action_result.message.lower()
                if "locked" in result_msg or "need" in result_msg and "key" in result_msg:
                    # Try to find the door the agent bumped into
                    for entity, direction in env.world.get_adjacent_entities(env.agent.x, env.agent.y):
                        if isinstance(entity, Door) and entity.locked:
                            door_key = f"{entity.color.value} door at ({entity.x}, {entity.y})"
                            blocked_doors[door_key] = {
                                "color": entity.color.value,
                                "pos": (entity.x, entity.y),
                                "needs": f"{entity.color.value} key",
                            }

            # Clean up collected items and unlocked doors from memory
            for item, coord in list(seen_items.items()):
                entities_at_coord = env.world.get_entities_at(*coord)
                if "(locked)" in item:
                    door_color = item.split()[0]
                    if not any(isinstance(e, Door) and e.color.value == door_color and e.locked for e in entities_at_coord):
                        del seen_items[item]
                else:
                    if not any(getattr(e, "name", "") == item for e in entities_at_coord):
                        del seen_items[item]

            # Clean up unlocked doors from blocked_doors memory
            for door_key, door_info in list(blocked_doors.items()):
                entities_at_door = env.world.get_entities_at(*door_info["pos"])
                if not any(isinstance(e, Door) and e.locked for e in entities_at_door):
                    del blocked_doors[door_key]

            from harness.pathfinder import find_shortest_path
            navigation_hints = {}
            for item_name, coord in seen_items.items():
                path = find_shortest_path(env.world, curr_pos, coord, ignore_locked_doors=True)
                if path:
                    hint = ", ".join(path[:3])
                    if len(path) > 3:
                        hint += ", ..."
                    navigation_hints[item_name] = f"at {coord}. Path: {hint}"
                else:
                    navigation_hints[item_name] = f"at {coord}. Path: (No direct path found)"

            import re
            for sign_msg in known_signs.values():
                matches = re.findall(r'(\w+)\((\d+),\s*(\d+)\)', sign_msg)
                for name, x, y in matches:
                    coord = (int(x), int(y))
                    if coord == curr_pos:
                        continue
                    path = find_shortest_path(env.world, curr_pos, coord, ignore_locked_doors=True)
                    if path:
                        hint = ", ".join(path[:3])
                        if len(path) > 3:
                            hint += ", ..."
                        navigation_hints[f"{name.lower()} (from sign)"] = f"at {coord}. Path: {hint}"

            spatial_memory = {
                "loop_warning": loop_warning,
                "known_signs": known_signs,
                "seen_items": seen_items,
                "navigation_hints": navigation_hints,
                "blocked_doors": blocked_doors,
            }

            # 1. Format observation
            obs_text = format_observation(observation, spatial_memory=spatial_memory)

            # 2. Build prompt with task and history
            user_prompt = build_turn_prompt(
                task_description=task_description,
                observation_text=obs_text,
                history=[] if self.use_history else self.history,
            )

            # 3. Query LLM with automatic retries for rate limits
            raw_response = None
            max_retries = 5
            retry_count = 0
            while retry_count < max_retries:
                try:
                    raw_response = self.provider.chat(SYSTEM_PROMPT, user_prompt, use_history=self.use_history)
                    if not raw_response or not raw_response.strip():
                        raise ValueError("Empty response received from LLM provider (possibly blocked by safety filters).")
                    break
                except Exception as e:
                    err_msg = str(e)
                    is_empty = "Empty response received from LLM provider" in err_msg
                    is_rate_limit = any(
                        keyword in err_msg.lower() 
                        for keyword in ["429", "resource_exhausted", "quota", "rate limit", "too many requests"]
                    )
                    if (is_rate_limit or is_empty) and retry_count < max_retries - 1:
                        retry_count += 1
                        sleep_time = 30 if is_rate_limit else 2
                        if is_rate_limit and "retry in" in err_msg.lower():
                            try:
                                import re
                                match = re.search(r"retry in\s*([\d\.]+)", err_msg.lower())
                                if match:
                                    sleep_time = int(float(match.group(1).rstrip('.'))) + 2
                            except:
                                pass
                        
                        if self.verbose:
                            if is_empty:
                                print(f"\n[WARNING]  Received empty response. Retrying in {sleep_time} seconds (Attempt {retry_count}/{max_retries})...", flush=True)
                            else:
                                print(f"\n[WARNING]  API Rate Limit hit. Sleeping for {sleep_time} seconds before retrying (Attempt {retry_count}/{max_retries})...\n  [Raw Error]: {e}", flush=True)
                        time.sleep(sleep_time)
                        continue
                    else:
                        if self.verbose:
                            print(f"\n  [FAIL] LLM API error: {e}", flush=True)
                        raw_response = f"THINK: API error: {err_msg}\nACTION: look"
                        break

            # 4. Parse response
            thinking, action_str = parse_llm_response(raw_response)

            # 5. Execute action
            observation, done, info = env.step(action_str)

            # Check if sign was read
            if info.get("success", False) and action_str.strip().lower().startswith("interact"):
                sign = None
                for entity in env.world.get_entities_at(env.agent.x, env.agent.y):
                    if isinstance(entity, Sign):
                        sign = entity
                        break
                if not sign:
                    for entity, direction in env.world.get_adjacent_entities(env.agent.x, env.agent.y):
                        if isinstance(entity, Sign):
                            sign = entity
                            break
                if sign:
                    room = env.world.get_room_at(sign.x, sign.y)
                    loc_desc = f"Sign in {room.name}" if room else f"Sign at ({sign.x}, {sign.y})"
                    known_signs[loc_desc] = sign.message

            # 6. Record history
            step_data = {
                "step": env.step_count,
                "observation": obs_text,
                "thinking": thinking,
                "action": action_str,
                "result": info.get("result", ""),
                "success": info.get("success", False),
                "agent_position": {"x": env.agent.x, "y": env.agent.y},
                "inventory": [e.name for e in env.agent.inventory],
                "task_progress": env.task.check_completion(env.world, env.agent).to_dict(),
            }

            self.history.append({
                "action": action_str,
                "result": info.get("result", ""),
                "thinking": thinking,
            })
            self.full_log.append(step_data)

            # Console output
            if self.verbose:
                self._print_step(step_data)

            # Callback
            if self.on_step:
                self.on_step(step_data, env.get_full_state())

            # Small delay for visualization
            if self.on_step:
                time.sleep(0.1)

        # Final summary
        elapsed = time.time() - start_time
        progress = env.task.check_completion(env.world, env.agent)
        
        summary = {
            "task_id": task_id,
            "seed": getattr(env, "seed", None),
            "task_description": task_description,
            "completed": progress.completed,
            "completion_message": progress.message,
            "steps_taken": env.step_count,
            "max_steps": env.task.max_steps,
            "elapsed_seconds": round(elapsed, 2),
            "tokens_used": self.provider.total_tokens_used,
            "final_inventory": [e.name for e in env.agent.inventory],
            "final_position": {"x": env.agent.x, "y": env.agent.y},
            "objectives": progress.objectives,
            "log": self.full_log,
        }

        if self.verbose:
            self._print_summary(summary)

        return summary

    def _safe_print(self, text: str) -> None:
        try:
            print(text)
        except Exception:
            # Fallback for Windows console encoding issues (e.g. Errno 22 with emojis)
            try:
                print(text.encode('ascii', 'replace').decode('ascii'))
            except Exception:
                pass

    def _print_step(self, step_data: dict) -> None:
        """Pretty-print a single step to console."""
        step = step_data["step"]
        action = step_data["action"]
        thinking = step_data["thinking"].replace('\n', ' ')
        result = step_data["result"].replace('\n', ' ')
        status = "[PASS]" if step_data.get("info", {}).get("success", False) else "[FAIL]"
        
        self._safe_print(f"  ┌─ Step {step}")
        self._safe_print(f"  │ Think: Think: {thinking[:120]}{'...' if len(thinking) > 120 else ''}")
        self._safe_print(f"  │ Action: Action: {action}")
        self._safe_print(f"  │ {status} Result: {result[:120]}{'...' if len(result) > 120 else ''}")
        
        info = step_data.get("info", {})
        if "inventory" in info:
            inv = info["inventory"]
            self._safe_print(f"  │ Inventory: Inventory: {', '.join(inv)}")
        if "task_progress" in info:
            completed_objs = [k for k, v in info["task_progress"].get("objectives", {}).items() if v]
            if completed_objs:
                self._safe_print(f"  │ [PASS] Completed: {', '.join(completed_objs)}")
                
        self._safe_print(f"  └{'─'*40}")

    def _print_summary(self, summary: dict) -> None:
        """Print run summary."""
        result = "SUCCESS" if summary["completed"] else "FAILED"
        self._safe_print(f"\n{'='*60}")
        self._safe_print(f"  RUN COMPLETE")
        self._safe_print(f"{'='*60}")
        
        self._safe_print(f"  Result: {result}")
        self._safe_print(f"  Steps: {summary['steps_taken']}/{summary['max_steps']}")
        self._safe_print(f"  Time: {summary['elapsed_seconds']}s")
        self._safe_print(f"  Tokens: {summary['tokens_used']}")
        
        if summary["completion_message"]:
            self._safe_print(f"  Message: {summary['completion_message']}")
        self._safe_print(f"{'='*60}\n")


def save_run_log(summary: dict, output_dir: str = "logs") -> str:
    """Save the full run log to a JSON file.
    
    Args:
        summary: The summary dict from LLMAgent.run().
        output_dir: Directory to save logs to.
        
    Returns:
        Path to the saved log file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"run_{summary['task_id']}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return filepath
