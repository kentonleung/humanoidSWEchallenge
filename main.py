"""LLM Agent in a Virtual World — Main Entry Point.

Run the agent from the command line:
    python main.py --task navigate            # Easy: find the red gem
    python main.py --task key_door            # Medium: find key, unlock door
    python main.py --task dungeon_escape      # Hard: collect gems, escape

    python main.py --web                      # Launch web visualizer
    python main.py --task navigate --headless # Run without visualization

Options:
    --task          Task to run: navigate, key_door, dungeon_escape
    --provider      LLM provider: cerebras (default), gemini, anthropic, openai
    --model         Model name (default: zai-glm-4.7)
    --web           Start the web visualization server
    --headless      Run without web server, console output only
    --port          Port for web server (default: 8000)
"""

import argparse
import json
import os
import sys
import socket

from engine.environment import Environment
from engine.world import LEVELS
from harness.llm_agent import (
    LLMAgent, AnthropicProvider, OpenAIProvider, GeminiProvider, CerebrasProvider, save_run_log,
)


def create_provider(provider_name: str, model: str = "") -> object:
    """Create an LLM provider instance."""
    try:
        if provider_name == "openai":
            return OpenAIProvider(model=model or "gpt-4o-mini")
        elif provider_name == "anthropic":
            return AnthropicProvider(model=model or "claude-3-5-sonnet-latest")
        elif provider_name == "cerebras":
            return CerebrasProvider(model=model or "zai-glm-4.7")
        elif provider_name == "gemini":
            return GeminiProvider(model=model or "gemini-1.5-flash")
        else:
            return CerebrasProvider(model=model or "zai-glm-4.7")
    except Exception as e:
        print(f"\n[FAIL] Failed to initialize {provider_name.capitalize()} provider!")
        print(f"Error: {e}")
        print("\nTo fix this, please ensure you have installed the required package and set your API key:")
        if provider_name == "openai":
            print("  pip install openai")
            print("  Windows: set OPENAI_API_KEY=sk-...")
            print("  Mac/Linux: export OPENAI_API_KEY='sk-...'")
        elif provider_name == "anthropic":
            print("  pip install anthropic")
            print("  Windows: set ANTHROPIC_API_KEY=sk-ant-...")
            print("  Mac/Linux: export ANTHROPIC_API_KEY='sk-ant-...'")
        elif provider_name == "gemini":
            print("  pip install google-genai")
            print("  Windows: set GEMINI_API_KEY=AIza...")
            print("  Mac/Linux: export GEMINI_API_KEY='AIza...'")
        elif provider_name == "cerebras":
            print("  pip install openai")
            print("  Windows: set CEREBRAS_API_KEY=...")
            print("  Mac/Linux: export CEREBRAS_API_KEY='...'")
        sys.exit(1)


def run_headless(task_id: str, provider_name: str, model: str = ""):
    """Run the agent in headless mode (console only)."""
    print(f"\n[AGENT] LLM Agent — Headless Mode")
    print(f"   Provider: {provider_name}")
    print(f"   Task: {task_id}")

    provider = create_provider(provider_name, model)
    env = Environment()
    agent = LLMAgent(provider=provider, verbose=True)

    summary = run_with_summary(agent, env, task_id)
    return summary


def run_with_summary(agent: LLMAgent, env: Environment, task_id: str) -> dict:
    """Run agent and save results."""
    summary = agent.run(env, task_id)

    # Save log
    log_path = save_run_log(summary)
    print(f"\n  [LOG] Full log saved to: {log_path}")

    return summary


def find_free_port(start_port: int, max_port: int = 8100) -> int:
    """Find a free port starting from start_port."""
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                pass
    return start_port

def run_web_server(port: int = 8000):
    """Start the web visualization server."""
    port = find_free_port(port)
    print(f"\n[WEB] Starting web visualizer on http://localhost:{port}")
    print(f"   Open this URL in your browser to watch the agent.\n")

    from visualization.server import start_server
    start_server(port=port)


def run_all_tasks(provider_name: str, model: str = ""):
    """Run all tasks and save example outputs."""
    results = {}
    for task_id in LEVELS:
        print(f"\n{'='*60}")
        print(f"  Running task: {task_id}")
        print(f"{'='*60}")

        provider = create_provider(provider_name, model)
        env = Environment()
        agent = LLMAgent(provider=provider, verbose=True)

        summary = run_with_summary(agent, env, task_id)
        results[task_id] = summary

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  ALL TASKS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Task':<20} {'Result':<10} {'Steps':<10} {'Time':<8}")
    print(f"  {'-'*48}")
    for task_id, s in results.items():
        result = "[PASS] PASS" if s["completed"] else "[FAIL] FAIL"
        steps = f"{s['steps_taken']}/{s['max_steps']}"
        time_s = f"{s['elapsed_seconds']}s"
        print(f"  {task_id:<20} {result:<10} {steps:<10} {time_s:<8}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="LLM Agent in a Virtual World",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --task navigate                    Run the navigation task (Cerebras)
  python main.py --task key_door --provider openai  Use OpenAI GPT
  python main.py --task navigate --provider anthropic  Use Anthropic Claude
  python main.py --web                              Launch web visualizer
  python main.py --all                              Run all tasks
        """,
    )

    parser.add_argument(
        "--task",
        choices=list(LEVELS.keys()),
        help="Task to run",
    )
    parser.add_argument(
        "--provider",
        choices=["cerebras", "gemini", "anthropic", "openai"],
        default="cerebras",
        help="LLM provider (default: cerebras)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name (default depends on provider)",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start the web visualization server",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (console only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for web server (default: 8000)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tasks sequentially",
    )

    args = parser.parse_args()

    # Validate
    if not args.web and not args.task and not args.all:
        parser.print_help()
        print("\n[WARNING]  Please specify --task, --web, or --all")
        sys.exit(1)

    # Run
    if args.web:
        run_web_server(port=args.port)
    elif args.all:
        run_all_tasks(args.provider, args.model)
    elif args.task:
        run_headless(args.task, args.provider, args.model)


if __name__ == "__main__":
    main()
