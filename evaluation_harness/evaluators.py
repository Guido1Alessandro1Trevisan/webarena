"""Base classes and helpers for evaluation on WebArena benchmarks."""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import collections
import html
import importlib
import json
import time
import urllib
from pathlib import Path
from typing import Union

from beartype import beartype
from nltk.tokenize import word_tokenize        # type: ignore

# Play-wright ─ we need BOTH sync *and* async flavours
from playwright.sync_api import CDPSession as SyncCDP, Page as SyncPage
from playwright.async_api import CDPSession as AsyncCDP, Page as AsyncPage

from browser_env.actions import Action
from browser_env.utils import StateInfo
from evaluation_harness.helper_functions import (
    PseudoPage,
    gitlab_get_project_memeber_role,
    llm_fuzzy_match,
    llm_ua_match,
    reddit_get_post_url,
    shopping_get_latest_order_url,
    shopping_get_sku_latest_review_author,
    shopping_get_sku_latest_review_rating,
)

# ─────────────────────────────────────────────────────────────────────────────
# Type aliases that satisfy beartype for BOTH sync and async Playwright APIs
# ─────────────────────────────────────────────────────────────────────────────
PageLike = Union[SyncPage, AsyncPage, PseudoPage]
CDPSessionLike = Union[SyncCDP, AsyncCDP]
Trajectory = list[Union[Action, StateInfo]]

# ─────────────────────────────────────────────────────────────────────────────
# Evaluator base-class
# ─────────────────────────────────────────────────────────────────────────────
class Evaluator:
    def __init__(self, eval_tag: str = "") -> None:
        self.eval_tag = eval_tag

    # beartype enforces runtime types; we accept both sync/async flavours now
    @beartype
    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: PageLike,
        client: CDPSessionLike,
    ) -> float:
        """Concrete evaluators override this."""
        raise NotImplementedError

    # helper: last action in trajectory
    @staticmethod
    def get_last_action(trajectory: Trajectory) -> Action:
        try:
            last_action = trajectory[-1]
        except Exception:
            raise ValueError(
                "The last element of trajectory should be an Action; "
                "add a fake stop-action if needed."
            )
        return last_action  # type: ignore[return-value]

    # helper: last state in trajectory (second-to-last element)
    @staticmethod
    def get_last_state(trajectory: Trajectory) -> StateInfo:
        try:
            last_state = trajectory[-2]
        except Exception:
            raise ValueError(
                "The penultimate element of trajectory should be a StateInfo; "
                "add a fake state if needed."
            )
        return last_state  # type: ignore[return-value]

# ─────────────────────────────────────────────────────────────────────────────
# 1. String-based evaluator
# ─────────────────────────────────────────────────────────────────────────────
class StringEvaluator(Evaluator):
    """Exact / must-include / fuzzy / UA string matching."""

    # utilities ---------------------------------------------------------------
    @staticmethod
    @beartype
    def clean_answer(answer: str) -> str:
        answer = answer.strip()
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        elif answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        return answer.lower()

    @staticmethod
    @beartype
    def exact_match(ref: str, pred: str) -> float:
        return float(
            StringEvaluator.clean_answer(pred)
            == StringEvaluator.clean_answer(ref)
        )

    @staticmethod
    @beartype
    def must_include(ref: str, pred: str, tokenize: bool = False) -> float:
        clean_ref = StringEvaluator.clean_answer(ref)
        clean_pred = StringEvaluator.clean_answer(pred)

        if tokenize and len(clean_ref) == 1 and len(word_tokenize(clean_ref)) == 1:
            tok_pred = word_tokenize(clean_pred)
            return float(clean_ref in tok_pred)
        return float(clean_ref in clean_pred)

    @staticmethod
    @beartype
    def fuzzy_match(ref: str, pred: str, intent: str) -> float:
        return llm_fuzzy_match(pred, ref, intent)

    @staticmethod
    @beartype
    def ua_match(ref: str, pred: str, intent: str) -> float:
        return llm_ua_match(pred, ref, intent)

    # main call ---------------------------------------------------------------
    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: PageLike | None = None,
        client: CDPSessionLike | None = None,
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        pred = self.clean_answer(self.get_last_action(trajectory)["answer"])
        score = 1.0

        for approach, value in configs["eval"]["reference_answers"].items():
            match approach:
                case "exact_match":
                    score *= self.exact_match(ref=value, pred=pred)

                case "must_include":
                    assert isinstance(value, list)
                    for must_value in value:
                        score *= self.must_include(
                            ref=must_value,
                            pred=pred,
                            tokenize=(len(value) == 1),
                        )

                case "fuzzy_match":
                    intent = configs["intent"]
                    if value == "N/A":
                        score *= self.exact_match(ref=value, pred=pred)
                        if score != 1:
                            score = self.ua_match(
                                intent=intent,
                                ref=configs["eval"]["string_note"],
                                pred=pred,
                            )
                    else:
                        assert isinstance(value, list)
                        for reference in value:
                            score *= self.fuzzy_match(
                                ref=reference, pred=pred, intent=intent
                            )
        return score

# ─────────────────────────────────────────────────────────────────────────────
# 2. URL evaluator
# ─────────────────────────────────────────────────────────────────────────────
class URLEvaluator(Evaluator):
    """Check if the final URL matches the references."""

    @beartype
    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: PageLike,
        client: CDPSessionLike | None = None,
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        def clean_url(url: str) -> str:
            return str(url).rstrip("/")

        def parse_url(url: str) -> tuple[str, dict[str, list[str]]]:
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc + parsed.path, urllib.parse.parse_qs(parsed.query)

        # prediction
        pred = clean_url(page.url)  # type: ignore[attr-defined]

        # references
        ref_urls = configs["eval"]["reference_url"].split(" |OR| ")
        ref_urls = [clean_url(u) for u in ref_urls]
        rule = configs["eval"].get("url_note", "GOLD in PRED")

        # matching logic
        if rule == "GOLD in PRED":
            ref_paths, ref_queries = self._parse_urls(ref_urls, parse_url)
            pred_path, pred_query = parse_url(pred)

            base_ok = float(any(ref in pred_path for ref in ref_paths))
            query_ok = 1.0
            for k, vals in ref_queries.items():
                query_ok *= float(any(v in pred_query.get(k, []) for v in vals))
            return base_ok * query_ok

        raise ValueError(f"Unknown URL matching rule: {rule}")

    # helper for URL parsing across a list
    @staticmethod
    def _parse_urls(urls, parse_fn):
        paths, queries = [], collections.defaultdict(set)
        for u in urls:
            p, q = parse_fn(u)
            paths.append(p)
            for k, v in q.items():
                queries[k].update(v)
        return paths, queries

# ─────────────────────────────────────────────────────────────────────────────
# 3. HTML-content evaluator (uses JS locators / helper funcs)
# ─────────────────────────────────────────────────────────────────────────────
class HTMLContentEvaluator(Evaluator):
    """Verify that certain HTML contents appear on page(s)."""

    @beartype
    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: PageLike,
        client: CDPSessionLike | None = None,
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        score = 1.0
        for target in configs["eval"]["program_html"]:
            url = self._resolve_url(target["url"], page)
            locator = target["locator"]

            # navigate if required
            if url != "last":
                page.goto(url)                    # type: ignore[attr-defined]
                time.sleep(3)  # TODO: remove sleep

            selected = self._select_element(page, locator, target)
            selected = html.unescape(selected)

            # scoring
            score *= self._score_required_contents(
                required=target["required_contents"], selected=selected
            )
        return score

    # helper--resolve URL
    @staticmethod
    def _resolve_url(url: str, page: PageLike) -> str:
        if url.startswith("func"):
            func = url.split("func:")[1].replace("__last_url__", page.url)  # type: ignore[attr-defined]
            return eval(func)
        return url

    # helper--select an element / HTML
    @staticmethod
    def _select_element(page: PageLike, locator: str, target: dict) -> str:
        if not locator.strip():                       # whole page
            return page.content()                     # type: ignore[attr-defined]

        if locator.startswith(("document.", "[...document.")):
            if "prep_actions" in target:
                for act in target["prep_actions"]:
                    try:
                        page.evaluate(f"() => {act}") # type: ignore[attr-defined]
                    except Exception:
                        pass
            try:
                return str(
                    page.evaluate(f"() => {locator}") # type: ignore[attr-defined]
                    or ""
                )
            except Exception:
                return ""

        if locator.startswith("func:"):
            func = locator.split("func:")[1].replace("__page__", "page")
            return eval(func)

        raise ValueError(f"Unknown locator: {locator}")

    # helper--score required contents
    @staticmethod
    def _score_required_contents(required: dict, selected: str) -> float:
        score = 1.0
        if "exact_match" in required:
            score *= StringEvaluator.exact_match(
                ref=required["exact_match"], pred=selected
            )
        elif "must_include" in required:
            for content in required["must_include"]:
                opts = content.split(" |OR| ")
                ok = any(
                    StringEvaluator.must_include(ref=o, pred=selected, tokenize=False)
                    for o in opts
                )
                score *= float(ok)
        else:
            raise ValueError(f"Unknown required_contents key(s): {required.keys()}")
        return score

# ─────────────────────────────────────────────────────────────────────────────
# Evaluator combiner
# ─────────────────────────────────────────────────────────────────────────────
class EvaluatorComb:
    def __init__(self, evaluators: list[Evaluator]) -> None:
        self.evaluators = evaluators

    @beartype
    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: PageLike,
        client: CDPSessionLike,
    ) -> float:
        score = 1.0
        for ev in self.evaluators:
            score *= ev(trajectory, config_file, page, client)
        return score

# ─────────────────────────────────────────────────────────────────────────────
# Router - create composite evaluator from task config
# ─────────────────────────────────────────────────────────────────────────────
@beartype
def evaluator_router(config_file: Path | str) -> EvaluatorComb:
    with open(config_file, "r") as f:
        configs = json.load(f)

    evaluators: list[Evaluator] = []
    for etype in configs["eval"]["eval_types"]:
        match etype:
            case "string_match":
                evaluators.append(StringEvaluator())
            case "url_match":
                evaluators.append(URLEvaluator())
            case "program_html":
                evaluators.append(HTMLContentEvaluator())
            case _:
                raise ValueError(f"Unsupported eval_type '{etype}'")
    return EvaluatorComb(evaluators)
