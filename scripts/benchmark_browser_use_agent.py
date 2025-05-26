#!/usr/bin/env python3
"""
main.py  ·  WebArena benchmark runner (fixed)

Key fixes & features
────────────────────
1. **Correct answer propagation** – the agent’s final answer is inserted into
   the last Action so `StringEvaluator` can grade it.

2. **Auth cookies support** – each task launches its own `BrowserSession`
   using the Playwright storage-state file produced by
   `browser_env/auto_login.py`.  The file is selected automatically from the
   task’s `start_url`.

3. **Cleaner structure** – each task is fully isolated; failures in one do not
   corrupt the next.

───────────────────────────────────────────────────────────────────────────────
"""
# Python std-lib
import argparse
import asyncio
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

# Third-party / project
from langchain_openai import ChatOpenAI
from playwright.async_api import Error as PWError

from browser_use import Agent, Controller
from browser_use.browser.profile import BrowserChannel
from browser_use.browser.session import BrowserSession
from browser_env.actions import create_stop_action
from browser_env.env_config import (
    GITLAB,
    SHOPPING,
    SHOPPING_ADMIN,
    REDDIT,
)
from evaluation_harness import evaluator_router

# ────────────────────────── Auth helpers ──────────────────────────
AUTH_FOLDER = Path(".auth")  # same default as auto_login.py


def _pick_storage_state(start_url: str) -> str | None:
    """Return the path to the storage-state JSON matching `start_url`."""
    netloc = urlparse(start_url).netloc

    mapping = {
        urlparse(GITLAB).netloc: "gitlab",
        urlparse(SHOPPING).netloc: "shopping",
        urlparse(SHOPPING_ADMIN).netloc: "shopping_admin",
        urlparse(REDDIT).netloc: "reddit",
    }

    site = mapping.get(netloc)
    if site:
        path = AUTH_FOLDER / f"{site}_state.json"
        return str(path) if path.exists() else None
    return None


# ────────────────────────── Task runner ──────────────────────────
async def run_task(
    config_file: str,
    model: str,
    result_dir: Path,
    max_steps: int,
    headless: bool,
) -> float | None:
    """Run one WebArena task → return its score (or None on failure)."""
    with open(config_file) as f:
        cfg = json.load(f)

    intent     = cfg["intent"]
    start_url  = cfg["start_url"]
    task_id    = cfg["task_id"]

    storage_state = _pick_storage_state(start_url)

    # Launch a fresh browser context for this task
    session = BrowserSession(
        headless=headless,
        storage_state=storage_state,
        channel=BrowserChannel.CHROMIUM,
        keep_alive=False,           # isolated per task
    )
    await session.start()
    await session.create_new_tab(url=start_url)

    # Agent setup
    llm   = ChatOpenAI(model=model, temperature=0.0)
    agent = Agent(
        task=intent,
        llm=llm,
        browser_session=session,
        controller=Controller(),
    )

    # ── Run the agent ──
    history = await agent.run(max_steps=max_steps)

    # Persist raw trace
    result_dir.mkdir(parents=True, exist_ok=True)
    history.save_to_file(result_dir / f"{task_id}.json")

    # ── Extract final answer ──
    final_answer: str | None = history.final_result()
    if final_answer is None:
        try:
            final_answer = (
                history.steps[-1]["model_output"]
                .get("answer", "")
                .strip()
            )
        except Exception:
            final_answer = ""

    if not final_answer:
        print(f"[WARN] Task {task_id}: no answer produced → un-graded.")
        await session.stop()
        return None

    # Build trajectory for evaluator
    stop_action = create_stop_action(final_answer)
    trajectory  = [{"observation": {}, "info": {}}] + [stop_action]

    # ── Evaluate ──
    try:
        page   = await session.get_current_page()
        client = await page.context.new_cdp_session(page)
        await client.send("Accessibility.enable")
        evaluator = evaluator_router(config_file)
        score     = evaluator(trajectory, config_file, page, client)
    except PWError as e:
        print(f"[WARN] CDP session error: {e}")
        score = None
    except Exception as e:
        print(f"[WARN] Evaluation failed: {e}")
        score = None

    await session.stop()
    return score


# ────────────────────────── CLI driver ──────────────────────────
async def main(args: argparse.Namespace) -> None:
    if args.clear_results and Path(args.result_dir).exists():
        shutil.rmtree(args.result_dir)

    scores: list[float | None] = []

    for idx in range(args.test_start_idx, args.test_end_idx):
        cfg_path = Path(args.config_dir) / f"{idx}.json"
        if not cfg_path.exists():
            print(f"[INFO] Missing {cfg_path}, skipping.")
            continue

        print(f"[INFO] Running task {idx} …")
        score = await run_task(
            config_file=str(cfg_path),
            model=args.model,
            result_dir=Path(args.result_dir),
            max_steps=args.max_steps,
            headless=args.headless,
        )
        scores.append(score)
        print(f"[INFO] Task {idx} score: {score}")

    # Aggregate
    scored = [s for s in scores if s is not None]
    if scored:
        avg = sum(scored) / len(scored)
        print(f"[RESULT] Average score: {avg:.2f}")


# ────────────────────────── Entry point ──────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run browser_use agent on WebArena with auth cookies."
    )
    p.add_argument("--test_start_idx", type=int, default=0)
    p.add_argument("--test_end_idx",   type=int, default=1)
    p.add_argument("--model",          type=str,  default="gpt-4o")
    p.add_argument("--max_steps",      type=int,  default=50)
    p.add_argument("--result_dir",     type=str,  default="browser_use_results")
    p.add_argument("--config_dir",     type=str,  default="config_files")
    p.add_argument(
        "--show-browser",
        dest="headless",
        action="store_false",
        help="Show browser window during runs.",
    )
    p.add_argument("--clear_results", action="store_true")
    p.set_defaults(headless=True)

    asyncio.run(main(p.parse_args()))
