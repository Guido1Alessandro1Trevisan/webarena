#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot WebArena runner   - 2025-05-26 (debug + progressive score log)
───────────────────────────────────────────────────────────────────────
* Same evaluation + rich debugging (HTML, snippets, per-evaluator scores)
* After **every task** its numeric score is appended / updated in
  `trajectories/scores_progress.json`, so you can watch the file grow.
"""

from __future__ import annotations
import argparse, asyncio, contextlib, gzip, importlib, json, subprocess, sys, tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import numpy as np, httpx
from dotenv import load_dotenv

# ─── hush httpx “event loop is closed” ──────────────────────
_orig_aclose = httpx.AsyncClient.aclose
async def _safe_aclose(self,*a,**kw):              # type: ignore[override]
    with contextlib.suppress(RuntimeError):
        await _orig_aclose(self,*a,**kw)
httpx.AsyncClient.aclose = _safe_aclose            # type: ignore[assignment]

# ─── folders & constants ───────────────────────────────────
load_dotenv()
ROOT_TRAJ_DIR = Path("trajectories"); ROOT_TRAJ_DIR.mkdir(exist_ok=True)
MASTER_INDEX  = ROOT_TRAJ_DIR / "trajectories.json"
SCORES_JSON   = ROOT_TRAJ_DIR / "scores_progress.json"   # <── progressive log

# ─── helpers ───────────────────────────────────────────────
def _jsonable(o:Any)->Any:
    if hasattr(o,"model_dump"): return o.model_dump()
    if isinstance(o,Path): return str(o)
    if isinstance(o,np.ndarray): return o.tolist()
    if isinstance(o,(np.integer,np.floating)): return o.item()
    return str(o)

def _write_jsonl(fp,item)->None:
    fp.write(json.dumps(item,default=_jsonable,ensure_ascii=False)+"\n")

_HTML_KEYS={"html","dom","dom_tree","dom_nodes","page_html",
            "inner_html","outer_html","full_html","content","document"}
def strip_html(obs:Any)->Any:
    if not isinstance(obs,dict): return obs
    return {k:v for k,v in obs.items()
            if k not in _HTML_KEYS and (not isinstance(v,str) or len(v)<=20_000)}

def first_time_setup()->None:
    if not any(Path("config_files").glob("*.json")):
        subprocess.run([sys.executable,"scripts/generate_test_data.py"],check=True)
    if not Path("storage_states").exists():
        subprocess.run(["bash","prepare.sh"],check=True)

# ─── progressive score writer ─────────────────────────────
def _append_score(task_id:int, score:float)->None:
    """
    Persist (or overwrite) the score for a task to SCORES_JSON.
    Structure: { "704": 1.0, "705": 0.0, ... }
    """
    data: Dict[str,float] = {}
    if SCORES_JSON.exists():
        try: data = json.loads(SCORES_JSON.read_text())
        except Exception: pass   # corrupt file → start fresh
    data[str(task_id)] = score
    SCORES_JSON.write_text(json.dumps(data, indent=2))

# ───────────────────────── run_task (debug edition) ───────
#   This is identical to the long “debug” function from earlier answers,
#   including HTML/URL snippet printing and evaluator tracing.
#   ↓↓↓ BEGIN run_task ↓↓↓
def run_task(cfg_path:Path, steps:int, visible:bool, inspect:bool)->float:
    async def _async()->float:
        from browser_env.async_envs import AsyncScriptBrowserEnv
        from browser_env import (
            create_none_action, create_stop_action, create_goto_url_action,
            create_scroll_action, create_id_based_action, create_page_focus_action,
            create_new_tab_action, create_page_close_action, create_go_back_action,
            Trajectory, Action
        )
        from browser_use import Agent
        from browser_use.browser.session import BrowserSession
        from langchain_openai import ChatOpenAI

        answer_stop_seen = False; answer_text = ""

        def _extract_text(p:Any)->str:
            if isinstance(p,str): return p
            if not isinstance(p,dict): return ""
            for k in ("text","answer","content","result","output","response","value"):
                if isinstance(p.get(k),str) and p[k].strip():
                    return p[k]
            for v in p.values():
                if isinstance(v,str) and v.strip(): return v
            return ""

        def translate(ma:Dict[str,Any])->List[Action]:
            nonlocal answer_stop_seen, answer_text
            name,p = next(iter(ma.items()))
            match name:
                case "click_element_by_index": return [create_id_based_action(f"click [{p['index']}]")]
                case "input_text":  return [create_id_based_action(f"type [{p['index']}] [{p.get('text',p.get('content',''))}] [0]")]
                case "search_google": return [create_goto_url_action(f"https://www.google.com/search?q={p['query']}&udm=14")]
                case "go_to_url":    return [create_goto_url_action(p["url"])]
                case "scroll_down":  return [create_scroll_action("down")]
                case "scroll_up":    return [create_scroll_action("up")]
                case "switch_tab":   return [create_page_focus_action(p["page_id"])]
                case "open_tab":
                    acts=[create_new_tab_action()]
                    if (u:=p.get("url")): acts.append(create_goto_url_action(u))
                    return acts
                case "close_tab":    return [create_page_close_action()]
                case "go_back":      return [create_go_back_action()]
                case "done":
                    txt=_extract_text(p)
                    if txt.strip(): answer_stop_seen=True; answer_text=txt
                    print(f"[translate] captured answer for STOP → «{txt}»",flush=True)
                    return [create_stop_action(txt)]
                case _: print(f"[translate] unhandled action: {ma}",flush=True); return []

        env = AsyncScriptBrowserEnv(headless=not visible)
        obs,info = await env.areset(options={"config_file":str(cfg_path)})
        session  = BrowserSession(playwright=env.playwright,browser=env.browser,
                                  browser_context=env.context,page=env.page)
        agent    = Agent(task=json.loads(cfg_path.read_text())["intent"],
                         llm=ChatOpenAI(model="gpt-4.1",temperature=0),
                         browser_session=session)

        traj:Trajectory=[{"observation":obs,"info":info}]
        stem=f"{cfg_path.stem}_{datetime.now():%Y%m%d_%H%M%S}"
        gz=gzip.open(ROOT_TRAJ_DIR/f"{stem}.jsonl.gz","wt",encoding="utf-8")
        _write_jsonl(gz,traj[-1])

        async def on_step_end(a:Agent)->None:
            hist=a.state.history.history[-1]
            if not hist or not hist.model_output: return
            for act in hist.model_output.action:
                for wa in translate(act.model_dump(exclude_none=True)):
                    traj.append(wa); _write_jsonl(gz,wa)
                    if env.page.is_closed(): continue
                    o,*_,i=await env.astep(create_none_action())
                    traj.append({"observation":o,"info":i}); _write_jsonl(gz,traj[-1])

        await agent.run(max_steps=steps,on_step_end=on_step_end)
        if hasattr(agent.llm,"aclose"): await agent.llm.aclose()

        # ensure trailing STOP
        if traj[-1].get("action_type")==8:
            if answer_stop_seen and not traj[-1].get("answer","").strip():
                traj[-1]["answer"]=answer_text
        else:
            traj.append(create_stop_action(answer_text if answer_stop_seen else ""))
            _write_jsonl(gz,traj[-1])
        gz.close()

        # ─── synchronous evaluation  (same as earlier) ────
        async def _evaluate()->float:
            hf=importlib.import_module("evaluation_harness.helper_functions")
            if not hasattr(hf,"PseudoPage"):
                class PseudoPage:
                    def __init__(self,page,url): self.original_page,self.url=page,url
                    def __getattr__(self,a): return getattr(self.original_page,a)
                hf.PseudoPage=PseudoPage
                sys.modules["evaluation_harness.helper_functions"].PseudoPage=PseudoPage

            from playwright.sync_api import sync_playwright
            from evaluation_harness.evaluators import evaluator_router

            def _work()->float:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    # IMPORTANT — use the Browser object, not the Playwright singleton
                    context = browser.new_context(storage_state=storage_state)

                    page = context.new_page()
                    page.goto(last_url, wait_until="load")

                    if inspect:
                        html_path=Path(tempfile.gettempdir())/f"webarena_{cfg_path.stem}.html"
                        html_path.write_text(page.content(),encoding="utf-8")
                        print("── DEBUG (evaluate) ───────────────────────────")
                        print("URL :",page.url)
                        print(f"Full HTML → {html_path}")
                        print("Snippet:",page.content()[:1000].replace("\n"," "),"…")
                        print("────────────────────────────────────────────────")

                    comb=evaluator_router(cfg_path)
                    score=1.0
                    for ev in comb.evaluators:
                        if inspect: print(f"\n[Eval:{ev.__class__.__name__}] starting …")
                        if inspect and ev.__class__.__name__=="HTMLContentEvaluator":
                            partial=ev.__call__(traj,cfg_path,page,None); print("[HTMLContentEvaluator] returned",partial)
                        else:
                            partial=ev(traj,cfg_path,page,None)
                        score*=partial
                        if inspect: print(f"[Eval:{ev.__class__.__name__}] partial score → {score}")
                    browser.close(); return score
            return await asyncio.to_thread(_work)

        storage_state=await env.context.storage_state(); last_url=env.page.url
        score_val=await _evaluate()

        # stripped trajectory + master index
        nohtml_file=ROOT_TRAJ_DIR/f"{stem}_nohtml.json"
        nohtml_file.write_text(json.dumps(
            [ {"observation":strip_html(t['observation']),"info":t.get('info')}
              if isinstance(t,dict) and 'observation'in t else t for t in traj],
            default=_jsonable,indent=2))
        master=json.loads(MASTER_INDEX.read_text()) if MASTER_INDEX.exists() else []
        master.append({"task":cfg_path.stem,"file":nohtml_file.name,"sanitized":True})
        MASTER_INDEX.write_text(json.dumps(master,indent=2))
        await env.aclose()
        print(f"▼ FINAL SCORE for {cfg_path.name}: {score_val:.4f}")
        return score_val
    return asyncio.run(_async())
# ↑↑↑ END run_task ↑↑↑

# ───────────────────────── range runner (with JSON log) ────
def run_range(start:int,end:int,steps:int,visible:bool,inspect:bool)->None:
    if end < start:
        sys.exit("--to must be ≥ --from")

    scores: List[float] = []
    perfect = 0

    for tid in range(start, end + 1):
        cfg = Path("config_files") / f"{tid}.json"
        if not cfg.exists():
            sys.exit(f"{cfg} not found.")

        print(f"\n=== Running task {tid} ===")
        task_score = run_task(cfg, steps, visible, inspect)

        # progressive JSON logging
        _append_score(tid, task_score)

        scores.append(task_score)
        if task_score == 1.0:
            perfect += 1

        print(f"► Score for task {tid}: {task_score:.4f} "
              f"(logged in {SCORES_JSON.name})")

    avg = sum(scores) / len(scores) if scores else 0.0
    print("\n=== Summary ===")
    print(f"Perfect (1.0): {perfect}/{len(scores)}")
    print(f"Average score: {avg:.4f}")
    print(f"Progress log : {SCORES_JSON.resolve()}")

# ─────────────────────────── CLI ────────────────────────────
if __name__ == "__main__":
    first_time_setup()

    ap = argparse.ArgumentParser(description="Run WebArena tasks")
    ap.add_argument("--from", dest="start", type=int, required=True)
    ap.add_argument("--to",   dest="end",   type=int, required=True)
    ap.add_argument("--steps",   type=int, default=20)
    ap.add_argument("--visible", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    run_range(args.start, args.end, args.steps, args.visible, args.inspect)
