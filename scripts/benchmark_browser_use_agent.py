import argparse
import asyncio
import json
import shutil
from pathlib import Path

from browser_use import Agent, Controller
from browser_use.browser import BrowserSession
from langchain_openai import ChatOpenAI

from browser_env.actions import create_stop_action

try:  # optional dependency
    from evaluation_harness import evaluator_router
except Exception as e:  # pragma: no cover - optional
    print(f"Could not import evaluation harness: {e}. Skipping scoring.")
    evaluator_router = None


async def run_task(
    config_file: str,
    model: str,
    result_dir: str,
    max_steps: int,
    headless: bool,
) -> float | None:
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

    score: float | None = None
    if evaluator_router is not None:
        evaluator = evaluator_router(config_file)
        action = create_stop_action(history.final_result() or "")
        trajectory = [{"observation": {}, "info": {}}, action]
        score = evaluator(trajectory, config_file, page, client)

    await session.stop()
    return score


async def main(args: argparse.Namespace) -> None:
    if args.clear_results:
        if Path(args.result_dir).exists():
            shutil.rmtree(args.result_dir)
    scores = []
    for idx in range(args.test_start_idx, args.test_end_idx):
        config_file = Path(args.config_dir) / f"{idx}.json"
        if not config_file.exists():
            print(f"Config file {config_file} does not exist, skipping.")
            continue

        print(f"Running task {idx}...")
        score = await run_task(
            str(config_file),
            args.model,
            args.result_dir,
            args.max_steps,
            args.headless,
        )
        scores.append(score)
        print(f"Task {idx} score: {score}")

    scored = [s for s in scores if s is not None]
    if scored:
        avg = sum(scored) / len(scored)
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
    parser.add_argument("--config_dir", type=str, default="config_files")
    parser.add_argument(
        "--show-browser",
        dest="headless",
        action="store_false",
        help="Show the browser window during runs",
    )
    parser.add_argument("--clear_results", action="store_true")
    parser.set_defaults(headless=True)

    args = parser.parse_args()
    asyncio.run(main(args))