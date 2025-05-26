#!/usr/bin/env python3
import argparse
import asyncio
import json
from pathlib import Path

from browser_use import Agent, Controller
from browser_use.browser import BrowserSession
from langchain_openai import ChatOpenAI

from browser_env.actions import create_stop_action
from evaluation_harness import evaluator_router


async def run_task(
    config_file: str, model: str, result_dir: str, max_steps: int, headless: bool
) -> float:
    with open(config_file) as f:
        cfg = json.load(f)

    intent = cfg["intent"]
    start_url = cfg["start_url"]
    storage_state = cfg.get("storage_state")
    task_id = cfg["task_id"]

    session = BrowserSession(headless=headless, storage_state=storage_state)
    await session.start()
    await session.create_new_tab(url=start_url)

    llm = ChatOpenAI(model=model, temperature=0.0)
    agent = Agent(
        task=intent, llm=llm, browser_session=session, controller=Controller()
    )

    history = await agent.run(max_steps=max_steps)

    Path(result_dir).mkdir(parents=True, exist_ok=True)
    history_file = Path(result_dir) / f"{task_id}.json"
    history.save_to_file(history_file)

    page = await session.get_current_page()
    client = await page.context.new_cdp_session(page)
    await client.send("Accessibility.enable")

    evaluator = evaluator_router(config_file)
    action = create_stop_action(history.final_result() or "")
    trajectory = [{"observation": {}, "info": {}}, action]
    score = evaluator(trajectory, config_file, page, client)

    await session.stop()
    return score


async def main(args: argparse.Namespace) -> None:
    scores = []
    for idx in range(args.test_start_idx, args.test_end_idx):
        config_file = f"config_files/{idx}.json"
        if not Path(config_file).exists():
            print(f"Config file {config_file} does not exist, skipping.")
            continue

        print(f"Running task {idx}...")
        score = await run_task(
            config_file, args.model, args.result_dir, args.max_steps, args.headless
        )
        scores.append(score)
        print(f"Task {idx} score: {score}")

    if scores:
        avg = sum(scores) / len(scores)
        print(f"Average score: {avg:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark browser-use agent on WebArena benchmark"
    )
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=1)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--result_dir", type=str, default="browser_use_results")
    parser.add_argument(
        "--headless", action="store_true", help="Run browser in headless mode"
    )

    args = parser.parse_args()
    asyncio.run(main(args))
