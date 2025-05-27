# Tools to generate from OpenAI prompts.
# Adopted from https://github.com/zeno-ml/zeno-build/
"""Utility functions for generating completions and chat completions
using the **OpenAI Python SDK ≥ 1.0.0**.

The legacy `openai.Completion.create` / `openai.ChatCompletion.create` calls
were removed in v1 of the SDK.  This module upgrades the original helpers
to the new interface that hangs off an `OpenAI`/`AsyncOpenAI` client
instance (e.g. `client.chat.completions.create`).

Usage remains identical to the original helpers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any, Tuple

import aiolimiter
from tqdm.asyncio import tqdm_asyncio

# —— OpenAI v1 SDK ————————————————————————————————————————————
import openai  # needed for type hints / error subclasses
from openai import APIError, OpenAI, RateLimitError
from openai import AsyncOpenAI  # async client

###############################################################################
# Generic retry decorator — unchanged except for updated error classes        #
###############################################################################

def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: Tuple[Any, ...] = (RateLimitError,),
):
    """Retry *func* with exponential back‑off when *errors* are raised."""

    def wrapper(*args, **kwargs):  # type: ignore
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return func(*args, **kwargs)
            except errors:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                delay *= exponential_base * (1 + jitter * random.random())
                print(f"Retrying in {delay:.2f} seconds …")
                time.sleep(delay)
            except Exception:  # pragma: no cover — re‑raise unexpected errors
                raise

    return wrapper

###############################################################################
# Helper constructors for (a)synchronous OpenAI clients                      #
###############################################################################

def _get_openai_client() -> OpenAI:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable must be set.")
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        organization=os.environ.get("OPENAI_ORGANIZATION"),
    )


def _get_async_openai_client() -> AsyncOpenAI:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable must be set.")
    return AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        organization=os.environ.get("OPENAI_ORGANIZATION"),
    )

###############################################################################
# Completion helpers (sync + async)                                          #
###############################################################################

async def _throttled_openai_completion_acreate(
    engine: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
):
    async with limiter:
        client = _get_async_openai_client()
        for _ in range(3):
            try:
                return await client.completions.create(
                    model=engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except RateLimitError:
                logging.warning("Rate limit hit — sleeping 10 s …")
                await asyncio.sleep(10)
            except APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        # Fallback empty response (shape‑compatible)
        class _EmptyResp:
            choices = [type("Choice", (), {"text": ""})()]
        return _EmptyResp()


async def agenerate_from_openai_completion(
    prompts: list[str],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Async bulk generation via the *Completion* endpoint."""

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    tasks = [
        _throttled_openai_completion_acreate(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for prompt in prompts
    ]
    responses = await tqdm_asyncio.gather(*tasks)
    return [r.choices[0].text for r in responses]


@retry_with_exponential_backoff
def generate_from_openai_completion(
    prompt: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    """Synchronous call to the *Completion* endpoint."""

    client = _get_openai_client()
    resp = client.completions.create(
        model=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=[stop_token] if stop_token else None,
    )
    return resp.choices[0].text

###############################################################################
# Chat‑completion helpers (sync + async)                                     #
###############################################################################

async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
):
    async with limiter:
        client = _get_async_openai_client()
        for _ in range(3):
            try:
                return await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except (RateLimitError, APIError) as e:
                logging.warning(f"{e.__class__.__name__}: {e} — sleeping 10 s …")
                await asyncio.sleep(10)
        # Fallback empty response
        class _EmptyResp:
            choices = [
                type(
                    "Choice",
                    (),
                    {"message": type("Msg", (), {"content": ""})()},
                )()
            ]
        return _EmptyResp()


async def agenerate_from_openai_chat_completion(
    messages_list: list[list[dict[str, str]]],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Async bulk generation via the *Chat Completion* endpoint."""

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    tasks = [
        _throttled_openai_chat_completion_acreate(
            model=engine,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for msgs in messages_list
    ]
    responses = await tqdm_asyncio.gather(*tasks)
    return [r.choices[0].message.content for r in responses]


@retry_with_exponential_backoff
def generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    """Synchronous call to the *Chat Completion* endpoint."""

    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=[stop_token] if stop_token else None,
    )
    return resp.choices[0].message.content

###############################################################################
# Fake chat‑completion for offline testing                                    #
###############################################################################

def fake_generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:  # noqa: D401 — simple sentence
    """Return a deterministic fake response (no API calls)."""

    return (
        "Let's think step‑by‑step. This page shows a list of links and buttons. "
        "There is a search box labelled ‘Search query’. I will click the search box "
        'to type the query. So the action I will perform is "click [60]".'
    )
