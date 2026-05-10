#!/usr/bin/env python3
"""
Daily trend collector for Anime/Manga/Novel sources.

Writes:
  - data/trending.json (latest normalized snapshot)
  - data/history.json   (rolling 7-day history for velocity/trend score)
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRENDING_FILE = DATA_DIR / "trending.json"
HISTORY_FILE = DATA_DIR / "history.json"

REQUEST_TIMEOUT = 20
HEADERS = {"User-Agent": "NoyakuTrendingBot/1.0 (+github-actions)"}
MAX_ITEMS = 15
HISTORY_DAYS = 7
REQUEST_PAUSE_SECONDS = 1.2


def _build_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session


SESSION = _build_session()


@dataclass
class Item:
    id: str
    title: str
    source: str
    category: str
    score: float
    url: str | None = None
    image: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None


def safe_get(url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
    # Keep outbound request pace low for public APIs.
    time.sleep(REQUEST_PAUSE_SECONDS + random.uniform(0.0, 0.35))
    response = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_anilist_trending_anime(limit: int = MAX_ITEMS) -> list[Item]:
    query = """
    query ($page: Int, $perPage: Int) {
      Page(page: $page, perPage: $perPage) {
        media(type: ANIME, sort: TRENDING_DESC, status_in: [RELEASING, FINISHED]) {
          id
          title {
            romaji
            english
          }
          description(asHtml: false)
          coverImage {
            extraLarge
            large
            medium
          }
          popularity
          averageScore
          siteUrl
        }
      }
    }
    """

    payload = {"query": query, "variables": {"page": 1, "perPage": limit}}
    time.sleep(REQUEST_PAUSE_SECONDS + random.uniform(0.0, 0.35))
    response = SESSION.post(
        "https://graphql.anilist.co",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    raw = response.json()
    media = raw["data"]["Page"]["media"]

    results: list[Item] = []
    for m in media:
        title = m["title"].get("english") or m["title"].get("romaji") or "Unknown"
        score = float(m.get("popularity") or 0)
        results.append(
            Item(
                id=f"anilist-anime-{m['id']}",
                title=title,
                source="AniList",
                category="anime",
                score=score,
                url=m.get("siteUrl"),
                image=(m.get("coverImage") or {}).get("extraLarge")
                or (m.get("coverImage") or {}).get("large")
                or (m.get("coverImage") or {}).get("medium"),
                description=m.get("description"),
                metadata={"average_score": m.get("averageScore")},
            )
        )
    return results


def fetch_jikan_season_now(limit: int = MAX_ITEMS) -> list[Item]:
    raw = safe_get("https://api.jikan.moe/v4/seasons/now", params={"limit": limit})
    data = raw.get("data", []) if isinstance(raw, dict) else []

    results: list[Item] = []
    for m in data:
        mal_id = m.get("mal_id")
        if mal_id is None:
            continue
        title = m.get("title") or "Unknown"
        popularity = float(m.get("popularity") or 0)
        scored_by = float(m.get("scored_by") or 0)
        score = max(popularity, scored_by)
        results.append(
            Item(
                id=f"jikan-anime-{mal_id}",
                title=title,
                source="Jikan",
                category="anime",
                score=score,
                url=m.get("url"),
                image=((m.get("images") or {}).get("jpg") or {}).get("large_image_url")
                or ((m.get("images") or {}).get("jpg") or {}).get("image_url")
                or ((m.get("images") or {}).get("webp") or {}).get("large_image_url")
                or ((m.get("images") or {}).get("webp") or {}).get("image_url"),
                description=m.get("synopsis"),
                metadata={"mal_score": m.get("score"), "members": m.get("members")},
            )
        )
    return results


def fetch_jikan_top_manga(limit: int = MAX_ITEMS) -> list[Item]:
    raw = safe_get("https://api.jikan.moe/v4/top/manga", params={"limit": limit})
    data = raw.get("data", []) if isinstance(raw, dict) else []

    results: list[Item] = []
    for m in data:
        mal_id = m.get("mal_id")
        if mal_id is None:
            continue
        title = m.get("title") or "Unknown"
        popularity = float(m.get("popularity") or 0)
        score = float(m.get("members") or popularity or 0)
        results.append(
            Item(
                id=f"jikan-manga-{mal_id}",
                title=title,
                source="Jikan",
                category="manga",
                score=score,
                url=m.get("url"),
                image=((m.get("images") or {}).get("jpg") or {}).get("large_image_url")
                or ((m.get("images") or {}).get("jpg") or {}).get("image_url")
                or ((m.get("images") or {}).get("webp") or {}).get("large_image_url")
                or ((m.get("images") or {}).get("webp") or {}).get("image_url"),
                description=m.get("synopsis"),
                metadata={"rank": m.get("rank"), "mal_score": m.get("score")},
            )
        )
    return results


def _compute_velocity(current_score: float, previous_score: float | None) -> float | None:
    if previous_score is None or previous_score <= 0:
        return None
    return (current_score - previous_score) / previous_score


def _load_history() -> dict[str, Any]:
    if not HISTORY_FILE.exists():
        return {"days": []}
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        logging.exception("Failed to parse history.json; starting with empty history.")
        return {"days": []}


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_snapshot() -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    items: list[Item] = []

    providers = [
        ("anilist_trending_anime", fetch_anilist_trending_anime),
        ("jikan_season_now", fetch_jikan_season_now),
        ("jikan_top_manga", fetch_jikan_top_manga),
    ]

    for provider_name, fn in providers:
        try:
            provider_items = fn(MAX_ITEMS)
            items.extend(provider_items)
            logging.info("Fetched %s items from %s", len(provider_items), provider_name)
        except Exception as exc:
            logging.exception("Provider %s failed", provider_name)
            errors.append({"provider": provider_name, "error": str(exc)})

    now = datetime.now(timezone.utc).replace(microsecond=0)
    today_key = now.date().isoformat()
    history = _load_history()
    history_days = history.get("days", [])

    yesterday_map: dict[str, float] = {}
    if history_days:
        yesterday_items = history_days[-1].get("items", [])
        for i in yesterday_items:
            item_id = i.get("id")
            score = i.get("score")
            if item_id is not None and isinstance(score, (int, float)):
                yesterday_map[item_id] = float(score)

    normalized_items: list[dict[str, Any]] = []
    for it in items:
        prev = yesterday_map.get(it.id)
        velocity = _compute_velocity(it.score, prev)
        row = asdict(it)
        row["previous_score"] = prev
        row["velocity"] = velocity
        normalized_items.append(row)

    for row in normalized_items:
        row.setdefault("image", None)
        row.setdefault("description", None)

    previous_snapshot_items: list[dict[str, Any]] = []
    if TRENDING_FILE.exists():
        try:
            previous_snapshot = json.loads(TRENDING_FILE.read_text(encoding="utf-8"))
            previous_snapshot_items = previous_snapshot.get("items", [])
        except Exception:
            logging.exception("Could not parse previous trending.json")

    normalized_items.sort(
        key=lambda x: (
            x["velocity"] if x["velocity"] is not None else float("-inf"),
            x["score"],
        ),
        reverse=True,
    )

    if not normalized_items and previous_snapshot_items:
        logging.warning(
            "All providers failed or returned empty; reusing previous non-empty snapshot items."
        )
        normalized_items = previous_snapshot_items
        for row in normalized_items:
            row.setdefault("image", None)
            row.setdefault("description", None)

    snapshot = {
        "generated_at_utc": now.isoformat().replace("+00:00", "Z"),
        "date": today_key,
        "item_count": len(normalized_items),
        "errors": errors,
        "items": normalized_items,
    }

    history_days.append(
        {
            "date": today_key,
            "items": [{"id": x["id"], "score": x["score"]} for x in normalized_items],
        }
    )
    history["days"] = history_days[-HISTORY_DAYS:]

    _save_json(TRENDING_FILE, snapshot)
    _save_json(HISTORY_FILE, history)
    return snapshot


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    snapshot = build_snapshot()
    logging.info("Done. Wrote %s items to %s", snapshot["item_count"], TRENDING_FILE)
    if snapshot["errors"]:
        logging.warning("Completed with %s provider error(s).", len(snapshot["errors"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
