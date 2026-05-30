#!/usr/bin/env python3
"""
Daily trend collector for Anime/Manga/Novel sources.

Writes:
  - data/trending.json (latest normalized snapshot)
  - data/trending_<category>.json (category-specific snapshots)
  - data/history.json   (rolling 7-day history for velocity/trend score)
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
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
REDDIT_SUBREDDITS = ("manhwa", "manga", "noveltranslations", "LightNovels", "webnovel")


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


def _classify_anime_category(country_of_origin: str | None) -> str:
    if country_of_origin == "CN":
        return "donghua"
    return "anime"


def _classify_manga_category(country_of_origin: str | None) -> str:
    if country_of_origin == "KR":
        return "manhwa"
    if country_of_origin == "CN":
        return "manhua"
    return "manga"


def _classify_text_novel_category(
    *,
    title: str | None,
    description: str | None,
    genres: list[str] | None,
) -> str:
    haystack = " ".join(
        [
            title or "",
            description or "",
            " ".join(genres or []),
        ]
    ).lower()
    if "web novel" in haystack or "webnovel" in haystack:
        return "web_novel"
    return "light_novel"


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
    genres: list[str] | None = None
    metadata: dict[str, Any] | None = None


def safe_get(url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
    # Keep outbound request pace low for public APIs.
    time.sleep(REQUEST_PAUSE_SECONDS + random.uniform(0.0, 0.35))
    response = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def safe_get_text(url: str, *, params: dict[str, Any] | None = None) -> str:
    # Keep outbound request pace low for public websites.
    time.sleep(REQUEST_PAUSE_SECONDS + random.uniform(0.0, 0.35))
    response = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def safe_get_text_with_headers(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> str:
    # Keep outbound request pace low for public websites.
    time.sleep(REQUEST_PAUSE_SECONDS + random.uniform(0.0, 0.35))
    headers = dict(SESSION.headers)
    if extra_headers:
        headers.update(extra_headers)
    response = SESSION.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def _absolute_url(base_url: str, raw_url: str | None) -> str | None:
    if not raw_url:
        return None
    return urljoin(base_url, raw_url)


def _pick_best_image_url(base_url: str, img_node: Any) -> str | None:
    if img_node is None:
        return None
    candidates = [
        img_node.get("data-src"),
        img_node.get("data-original"),
        img_node.get("data-lazy-src"),
        img_node.get("src"),
    ]
    srcset = img_node.get("data-srcset") or img_node.get("srcset")
    if srcset:
        first_src = srcset.split(",")[0].strip().split(" ")[0].strip()
        if first_src:
            candidates.insert(0, first_src)

    for raw in candidates:
        if not raw:
            continue
        raw = str(raw).strip()
        if not raw:
            continue
        if raw.startswith("data:image/gif;base64"):
            continue
        return _absolute_url(base_url, raw)
    return None


def _clean_novel_title(raw_title: str) -> str:
    title = re.sub(r"\s+", " ", raw_title).strip()
    title = re.sub(r"^\d+(?:\.\d+)?\s+", "", title)
    return title


def _clean_reddit_topic(raw_title: str) -> str:
    title = raw_title.strip()
    title = re.sub(r"^\[[^\]]+\]\s*", "", title)
    title = re.sub(r"^\([^)]+\)\s*", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title[:180]


def _hydrate_from_detail_page(url: str) -> tuple[str | None, str | None]:
    try:
        html = safe_get_text_with_headers(
            url,
            extra_headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
    except Exception:
        return None, None

    soup = BeautifulSoup(html, "html.parser")
    image_url: str | None = None
    description: str | None = None

    og_image = soup.select_one("meta[property='og:image']")
    if og_image:
        image_url = _absolute_url(url, og_image.get("content"))
    if not image_url:
        tw_image = soup.select_one("meta[name='twitter:image']")
        if tw_image:
            image_url = _absolute_url(url, tw_image.get("content"))
    if not image_url:
        img = soup.select_one("img")
        image_url = _pick_best_image_url(url, img)

    og_desc = soup.select_one("meta[property='og:description']")
    if og_desc and og_desc.get("content"):
        description = og_desc.get("content", "").strip()
    if not description:
        desc = soup.select_one("meta[name='description']")
        if desc and desc.get("content"):
            description = desc.get("content", "").strip()
    if not description:
        node = soup.select_one(".summary, .description, .book-description, .novel-desc, p")
        if node:
            description = node.get_text(" ", strip=True)

    return image_url, description


def _slug_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    return path.replace("/", "-") or "unknown"


def _extract_site_novel_items(
    *,
    source_name: str,
    ranking_url: str,
    allowed_path_prefixes: tuple[str, ...],
    limit: int = MAX_ITEMS,
) -> list[Item]:
    html = safe_get_text(ranking_url)
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.select("a[href]")

    seen: set[str] = set()
    results: list[Item] = []
    rank_score = float(limit + 1)

    for a in anchors:
        href = _absolute_url(ranking_url, a.get("href"))
        if not href:
            continue
        parsed = urlparse(href)
        if not parsed.netloc:
            continue
        path = parsed.path.strip("/")
        if not path:
            continue
        if not any(path.startswith(prefix) for prefix in allowed_path_prefixes):
            continue
        if parsed.fragment:
            continue

        title = _clean_novel_title(a.get_text(" ", strip=True))
        if not title or len(title) < 2:
            continue
        if title.lower() == "read" or title.isdigit():
            continue
        if href in seen:
            continue
        seen.add(href)

        card = a.find_parent(["li", "article", "div"])
        image_url: str | None = None
        description: str | None = None
        if card:
            img = card.select_one("img")
            image_url = _pick_best_image_url(ranking_url, img)
            desc_node = card.select_one("p, .summary, .description, .intro")
            if desc_node:
                description = desc_node.get_text(" ", strip=True)

        score = rank_score
        rank_score -= 1.0
        results.append(
            Item(
                id=f"{source_name.lower()}-{_slug_from_url(href)}",
                title=title,
                source=source_name,
                category="web_novel",
                score=score,
                url=href,
                image=image_url,
                description=description,
                genres=[],
                metadata={"ranking_page": ranking_url},
            )
        )
        if len(results) >= limit:
            break

    return results


def fetch_anilist_trending_anime(limit: int = MAX_ITEMS) -> list[Item]:
    query = """
    query ($page: Int, $perPage: Int) {
      Page(page: $page, perPage: $perPage) {
        media(type: ANIME, sort: TRENDING_DESC, status_in: [RELEASING, FINISHED]) {
          id
          countryOfOrigin
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
          genres
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
                category=_classify_anime_category(m.get("countryOfOrigin")),
                score=score,
                url=m.get("siteUrl"),
                image=(m.get("coverImage") or {}).get("extraLarge")
                or (m.get("coverImage") or {}).get("large")
                or (m.get("coverImage") or {}).get("medium"),
                description=m.get("description"),
                genres=[g for g in (m.get("genres") or []) if isinstance(g, str)],
                metadata={"average_score": m.get("averageScore")},
            )
        )
    return results


def fetch_anilist_trending_manga(limit: int = MAX_ITEMS) -> list[Item]:
    query = """
    query ($page: Int, $perPage: Int) {
      Page(page: $page, perPage: $perPage) {
        media(type: MANGA, sort: TRENDING_DESC, status_in: [RELEASING, FINISHED]) {
          id
          countryOfOrigin
          format
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
          genres
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
        format_value = m.get("format")
        description = m.get("description")
        genres = [g for g in (m.get("genres") or []) if isinstance(g, str)]
        category = (
            _classify_text_novel_category(
                title=title,
                description=description,
                genres=genres,
            )
            if format_value == "NOVEL"
            else _classify_manga_category(m.get("countryOfOrigin"))
        )
        results.append(
            Item(
                id=f"anilist-manga-{m['id']}",
                title=title,
                source="AniList",
                category=category,
                score=score,
                url=m.get("siteUrl"),
                image=(m.get("coverImage") or {}).get("extraLarge")
                or (m.get("coverImage") or {}).get("large")
                or (m.get("coverImage") or {}).get("medium"),
                description=description,
                genres=genres,
                metadata={
                    "average_score": m.get("averageScore"),
                    "country_of_origin": m.get("countryOfOrigin"),
                    "format": format_value,
                },
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
                genres=[
                    g.get("name")
                    for g in (m.get("genres") or [])
                    if isinstance(g, dict) and isinstance(g.get("name"), str)
                ],
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
                genres=[
                    g.get("name")
                    for g in (m.get("genres") or [])
                    if isinstance(g, dict) and isinstance(g.get("name"), str)
                ],
                metadata={"rank": m.get("rank"), "mal_score": m.get("score")},
            )
        )
    return results


def fetch_jikan_top_novels(limit: int = MAX_ITEMS) -> list[Item]:
    last_error: Exception | None = None
    raw: dict[str, Any] | list[Any] | None = None
    for type_value in ("novel", "novels", "lightnovels"):
        try:
            raw = safe_get(
                "https://api.jikan.moe/v4/top/manga",
                params={"limit": limit, "type": type_value},
            )
            break
        except requests.HTTPError as exc:
            last_error = exc
            continue

    if raw is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Could not fetch Jikan top novels with known type filters.")

    data = raw.get("data", []) if isinstance(raw, dict) else []

    results: list[Item] = []
    for m in data:
        mal_id = m.get("mal_id")
        if mal_id is None:
            continue
        title = m.get("title") or "Unknown"
        popularity = float(m.get("popularity") or 0)
        score = float(m.get("members") or popularity or 0)
        description = m.get("synopsis")
        genres = [
            g.get("name")
            for g in (m.get("genres") or [])
            if isinstance(g, dict) and isinstance(g.get("name"), str)
        ]
        category = _classify_text_novel_category(
            title=title,
            description=description,
            genres=genres,
        )
        results.append(
            Item(
                id=f"jikan-novel-{mal_id}",
                title=title,
                source="Jikan",
                category=category,
                score=score,
                url=m.get("url"),
                image=((m.get("images") or {}).get("jpg") or {}).get("large_image_url")
                or ((m.get("images") or {}).get("jpg") or {}).get("image_url")
                or ((m.get("images") or {}).get("webp") or {}).get("large_image_url")
                or ((m.get("images") or {}).get("webp") or {}).get("image_url"),
                description=description,
                genres=genres,
                metadata={"rank": m.get("rank"), "mal_score": m.get("score"), "type": "novels"},
            )
        )
    return results


def fetch_webnovel_best_sellers(limit: int = MAX_ITEMS) -> list[Item]:
    candidate_urls = [
        "https://www.webnovel.com/ranking/novel/season/best_sellers",
        "https://www.webnovel.com/ranking/novel/week/best_sellers",
        "https://www.webnovel.com/ranking/novel/all-time/best_sellers",
        "https://www.webnovel.com/",
    ]
    browser_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.webnovel.com/",
    }

    last_error: Exception | None = None
    for ranking_url in candidate_urls:
        try:
            html = safe_get_text_with_headers(ranking_url, extra_headers=browser_headers)
            soup = BeautifulSoup(html, "html.parser")
            anchors = soup.select("a[href]")
            seen: set[str] = set()
            results: list[Item] = []
            rank_score = float(limit + 1)

            for a in anchors:
                href = _absolute_url(ranking_url, a.get("href"))
                if not href:
                    continue
                parsed = urlparse(href)
                path = parsed.path.strip("/")
                if not path or not path.startswith("book/") or parsed.fragment:
                    continue
                title = _clean_novel_title(a.get_text(" ", strip=True))
                if (
                    not title
                    or len(title) < 2
                    or title.lower() == "read"
                    or title.isdigit()
                    or href in seen
                ):
                    continue
                seen.add(href)

                card = a.find_parent(["li", "article", "div"])
                image_url: str | None = None
                description: str | None = None
                if card:
                    img = card.select_one("img")
                    image_url = _pick_best_image_url(ranking_url, img)
                    desc_node = card.select_one("p, .summary, .description, .intro")
                    if desc_node:
                        description = desc_node.get_text(" ", strip=True)
                if not image_url or not description:
                    detail_image, detail_desc = _hydrate_from_detail_page(href)
                    image_url = image_url or detail_image
                    description = description or detail_desc

                results.append(
                    Item(
                        id=f"webnovel-{_slug_from_url(href)}",
                        title=title,
                        source="WebNovel",
                        category="web_novel",
                        score=rank_score,
                        url=href,
                        image=image_url,
                        description=description,
                        genres=[],
                        metadata={"ranking_page": ranking_url},
                    )
                )
                rank_score -= 1.0
                if len(results) >= limit:
                    break

            if results:
                return results
        except Exception as exc:
            last_error = exc
            continue

    logging.warning("WebNovel appears blocked (403/anti-bot). Returning no items for now.")
    if last_error:
        logging.info("WebNovel last error: %s", last_error)
    return []


def fetch_novelfire_home(limit: int = MAX_ITEMS) -> list[Item]:
    candidate_urls = [
        "https://novelfire.net/home",
        "https://novelfire.net/",
        "https://novelfire.net/ranking",
        "https://novelfire.net/ranking/weekly-ranking",
    ]
    browser_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://novelfire.net/",
    }

    last_error: Exception | None = None
    for ranking_url in candidate_urls:
        try:
            html = safe_get_text_with_headers(ranking_url, extra_headers=browser_headers)
            soup = BeautifulSoup(html, "html.parser")
            anchors = soup.select("a[href]")
            seen: set[str] = set()
            results: list[Item] = []
            rank_score = float(limit + 1)

            for a in anchors:
                href = _absolute_url(ranking_url, a.get("href"))
                if not href:
                    continue
                parsed = urlparse(href)
                path = parsed.path.strip("/")
                if not path:
                    continue
                if not (path.startswith("book/") or path.startswith("novel/")):
                    continue
                if parsed.fragment:
                    continue
                title = _clean_novel_title(a.get_text(" ", strip=True))
                if (
                    not title
                    or len(title) < 2
                    or title.lower() == "read"
                    or title.isdigit()
                    or href in seen
                ):
                    continue
                seen.add(href)

                card = a.find_parent(["li", "article", "div"])
                image_url: str | None = None
                description: str | None = None
                if card:
                    img = card.select_one("img")
                    image_url = _pick_best_image_url(ranking_url, img)
                    desc_node = card.select_one("p, .summary, .description, .intro")
                    if desc_node:
                        description = desc_node.get_text(" ", strip=True)
                if not image_url or not description:
                    detail_image, detail_desc = _hydrate_from_detail_page(href)
                    image_url = image_url or detail_image
                    description = description or detail_desc

                results.append(
                    Item(
                        id=f"novelfire-{_slug_from_url(href)}",
                        title=title,
                        source="NovelFire",
                        category="web_novel",
                        score=rank_score,
                        url=href,
                        image=image_url,
                        description=description,
                        genres=[],
                        metadata={"ranking_page": ranking_url},
                    )
                )
                rank_score -= 1.0
                if len(results) >= limit:
                    break

            if results:
                return results
        except Exception as exc:
            last_error = exc
            continue

    logging.warning("NovelFire appears blocked (403/anti-bot). Returning no items for now.")
    if last_error:
        logging.info("NovelFire last error: %s", last_error)
    return []


def fetch_ranobes_home(limit: int = MAX_ITEMS) -> list[Item]:
    return _extract_site_novel_items(
        source_name="Ranobes",
        ranking_url="https://ranobes.top/",
        allowed_path_prefixes=("book/", "novels/", "ranobe/"),
        limit=limit,
    )


def _extract_romance_site_items(
    *,
    source_name: str,
    candidate_urls: list[str],
    allowed_path_prefixes: tuple[str, ...],
    limit: int = MAX_ITEMS,
) -> list[Item]:
    browser_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    last_error: Exception | None = None

    for ranking_url in candidate_urls:
        try:
            html = safe_get_text_with_headers(ranking_url, extra_headers=browser_headers)
            soup = BeautifulSoup(html, "html.parser")
            anchors = soup.select("a[href]")
            seen: set[str] = set()
            results: list[Item] = []
            rank_score = float(limit + 1)

            for a in anchors:
                href = _absolute_url(ranking_url, a.get("href"))
                if not href:
                    continue
                parsed = urlparse(href)
                path = parsed.path.strip("/")
                if not path or parsed.fragment:
                    continue
                if not any(path.startswith(prefix) for prefix in allowed_path_prefixes):
                    continue

                title = _clean_novel_title(a.get_text(" ", strip=True))
                if (
                    not title
                    or len(title) < 2
                    or title.lower() == "read"
                    or title.isdigit()
                    or href in seen
                ):
                    continue
                seen.add(href)

                card = a.find_parent(["li", "article", "div"])
                image_url: str | None = None
                description: str | None = None
                if card:
                    img = card.select_one("img")
                    image_url = _pick_best_image_url(ranking_url, img)
                    desc_node = card.select_one("p, .summary, .description, .intro")
                    if desc_node:
                        description = desc_node.get_text(" ", strip=True)
                if not image_url or not description:
                    detail_image, detail_desc = _hydrate_from_detail_page(href)
                    image_url = image_url or detail_image
                    description = description or detail_desc

                results.append(
                    Item(
                        id=f"{source_name.lower()}-{_slug_from_url(href)}",
                        title=title,
                        source=source_name,
                        category="romance",
                        score=rank_score,
                        url=href,
                        image=image_url,
                        description=description,
                        genres=["Romance"],
                        metadata={"ranking_page": ranking_url},
                    )
                )
                rank_score -= 1.0
                if len(results) >= limit:
                    break

            if results:
                return results
        except Exception as exc:
            last_error = exc
            continue

    logging.warning("%s appears blocked/unavailable. Returning no items for now.", source_name)
    if last_error:
        logging.info("%s last error: %s", source_name, last_error)
    return []


def fetch_dreame_romance(limit: int = MAX_ITEMS) -> list[Item]:
    return _extract_romance_site_items(
        source_name="Dreame",
        candidate_urls=[
            "https://www.dreame.com/",
            "https://www.dreame.com/discover",
            "https://www.dreame.com/discover/romance",
        ],
        allowed_path_prefixes=("novel/", "story/", "book/"),
        limit=limit,
    )


def fetch_joyread_romance(limit: int = MAX_ITEMS) -> list[Item]:
    return _extract_romance_site_items(
        source_name="Joyread",
        candidate_urls=[
            "https://www.joyread.com/",
            "https://www.joyread.com/category/romance",
            "https://www.joyread.com/ranking",
        ],
        allowed_path_prefixes=("book/", "novel/", "story/"),
        limit=limit,
    )


def fetch_goodnovel_romance(limit: int = MAX_ITEMS) -> list[Item]:
    return _extract_romance_site_items(
        source_name="GoodNovel",
        candidate_urls=[
            "https://www.goodnovel.com/",
            "https://www.goodnovel.com/romance",
            "https://www.goodnovel.com/ranking",
        ],
        allowed_path_prefixes=("book/", "novel/", "story/"),
        limit=limit,
    )


def fetch_manhuaplus_manhua(limit: int = MAX_ITEMS) -> list[Item]:
    candidate_urls = [
        "https://manhuaplus.com/genre/manhua/",
        "https://manhuaplus.com/",
    ]
    browser_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://manhuaplus.com/",
    }
    last_error: Exception | None = None

    for ranking_url in candidate_urls:
        try:
            html = safe_get_text_with_headers(ranking_url, extra_headers=browser_headers)
            soup = BeautifulSoup(html, "html.parser")
            anchors = soup.select("a[href]")
            seen: set[str] = set()
            results: list[Item] = []
            rank_score = float(limit + 1)

            for a in anchors:
                href = _absolute_url(ranking_url, a.get("href"))
                if not href:
                    continue
                parsed = urlparse(href)
                path = parsed.path.strip("/")
                if not path or parsed.fragment:
                    continue
                if not path.startswith("manga/"):
                    continue

                title = _clean_novel_title(a.get_text(" ", strip=True))
                if (
                    not title
                    or len(title) < 2
                    or title.lower() == "read"
                    or title.isdigit()
                    or href in seen
                ):
                    continue
                seen.add(href)

                card = a.find_parent(["li", "article", "div"])
                image_url: str | None = None
                description: str | None = None
                if card:
                    img = card.select_one("img")
                    image_url = _pick_best_image_url(ranking_url, img)
                    desc_node = card.select_one("p, .summary, .description, .intro")
                    if desc_node:
                        description = desc_node.get_text(" ", strip=True)
                if not image_url or not description:
                    detail_image, detail_desc = _hydrate_from_detail_page(href)
                    image_url = image_url or detail_image
                    description = description or detail_desc

                results.append(
                    Item(
                        id=f"manhuaplus-{_slug_from_url(href)}",
                        title=title,
                        source="ManhuaPlus",
                        category="manhua",
                        score=rank_score,
                        url=href,
                        image=image_url,
                        description=description,
                        genres=[],
                        metadata={"ranking_page": ranking_url},
                    )
                )
                rank_score -= 1.0
                if len(results) >= limit:
                    break

            if results:
                return results
        except Exception as exc:
            last_error = exc
            continue

    logging.warning("ManhuaPlus appears blocked/unavailable. Returning no items for now.")
    if last_error:
        logging.info("ManhuaPlus last error: %s", last_error)
    return []


def fetch_reddit_discussed(limit: int = MAX_ITEMS) -> list[Item]:
    rows: list[Item] = []
    for subreddit in REDDIT_SUBREDDITS:
        raw = safe_get(
            f"https://www.reddit.com/r/{subreddit}/hot.json",
            params={"limit": max(10, limit)},
        )
        listing = raw.get("data", {}).get("children", []) if isinstance(raw, dict) else []
        for child in listing:
            data = child.get("data", {}) if isinstance(child, dict) else {}
            post_id = data.get("id")
            title = _clean_reddit_topic(str(data.get("title") or ""))
            if not post_id or not title:
                continue
            comments = int(data.get("num_comments") or 0)
            ups = int(data.get("ups") or 0)
            discussion_score = float((comments * 3) + ups)
            permalink = data.get("permalink")
            url = _absolute_url("https://www.reddit.com", permalink) if permalink else None
            rows.append(
                Item(
                    id=f"reddit-{subreddit}-{post_id}",
                    title=title,
                    source="Reddit",
                    category="discussed",
                    score=discussion_score,
                    url=url,
                    image=None,
                    description=data.get("selftext") or None,
                    genres=[],
                    metadata={
                        "subreddit": subreddit,
                        "ups": ups,
                        "num_comments": comments,
                        "discussion_score": discussion_score,
                    },
                )
            )

    rows.sort(key=lambda x: x.score, reverse=True)
    deduped: list[Item] = []
    seen_titles: set[str] = set()
    for item in rows:
        key = item.title.lower().strip()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


_CATEGORY_GENRES: dict[str, list[str]] = {
    "anime": ["Anime"],
    "donghua": ["Anime", "Chinese"],
    "manga": ["Manga"],
    "manhwa": ["Manhwa"],
    "manhua": ["Manhua"],
    "web_novel": ["Web Novel"],
    "light_novel": ["Light Novel"],
    "romance": ["Romance"],
    "discussed": [],
}


def _backfill_genres(row: dict[str, Any]) -> None:
    """Fill genres from category when the scraper couldn't extract them."""
    if row.get("genres"):
        return
    category = row.get("category") or ""
    row["genres"] = _CATEGORY_GENRES.get(category, [])


def _compute_velocity(current_score: float, previous_score: float | None) -> float | None:
    if previous_score is None or previous_score <= 0:
        return None
    return (current_score - previous_score) / previous_score


def _load_history() -> dict[str, Any]:
    if not HISTORY_FILE.exists():
        return {"days": []}
    raw = HISTORY_FILE.read_text(encoding="utf-8")
    if not raw.strip():
        logging.warning("history.json is empty; starting with empty history.")
        return {"days": []}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        backup_file = HISTORY_FILE.with_suffix(".invalid.json")
        backup_file.write_text(raw, encoding="utf-8")
        logging.warning(
            "history.json is invalid JSON; backed up to %s and starting with empty history.",
            backup_file,
        )
        return {"days": []}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("days", []), list):
        logging.warning("history.json has unexpected shape; starting with empty history.")
        return {"days": []}
    return parsed


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_category_snapshots(
    *,
    generated_at_utc: str,
    date: str,
    errors: list[dict[str, str]],
    items: list[dict[str, Any]],
) -> None:
    category_map: dict[str, list[dict[str, Any]]] = {}
    for row in items:
        category = row.get("category")
        if not isinstance(category, str) or not category:
            continue
        category_map.setdefault(category, []).append(row)

    for category, category_items in category_map.items():
        category_snapshot = {
            "generated_at_utc": generated_at_utc,
            "date": date,
            "category": category,
            "item_count": len(category_items),
            "errors": errors,
            "items": category_items,
        }
        _save_json(DATA_DIR / f"trending_{category}.json", category_snapshot)


def build_snapshot() -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    items: list[Item] = []

    providers = [
        ("anilist_trending_anime", fetch_anilist_trending_anime),
        ("anilist_trending_manga", fetch_anilist_trending_manga),
        ("jikan_season_now", fetch_jikan_season_now),
        ("jikan_top_manga", fetch_jikan_top_manga),
        ("jikan_top_novels", fetch_jikan_top_novels),
        ("webnovel_best_sellers", fetch_webnovel_best_sellers),
        ("novelfire_home", fetch_novelfire_home),
        # ranobes_home: blocked by bot protection — returns no image or description
        # ("ranobes_home", fetch_ranobes_home),
        ("manhuaplus_manhua", fetch_manhuaplus_manhua),
        ("reddit_discussed", fetch_reddit_discussed),
        # dreame_romance: JS SPA with bot protection — title bleeds into description,
        #                 og:image is a site-wide SVG placeholder for all stories
        # ("dreame_romance", fetch_dreame_romance),
        ("joyread_romance", fetch_joyread_romance),
        ("goodnovel_romance", fetch_goodnovel_romance),
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
        row.setdefault("genres", [])
        _backfill_genres(row)

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
            row.setdefault("genres", [])
            _backfill_genres(row)

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
    _write_category_snapshots(
        generated_at_utc=snapshot["generated_at_utc"],
        date=today_key,
        errors=errors,
        items=normalized_items,
    )
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
