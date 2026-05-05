"""
Tải văn bản Wikipedia (tiếng Anh) về các công ty AI qua API MediaWiki.
Với prop=extracts, khi gộp nhiều titles trong một request, Wikipedia thường chỉ
điền extract đầy đủ cho một trang — nên gọi lần lượt từng title.

https://www.mediawiki.org/wiki/API:Etiquette — gửi User-Agent hợp lệ.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "wikipedia_ai_companies"

ARTICLE_TITLES = [
    "OpenAI",
    "Anthropic",
    "Google DeepMind",
    "Cohere",
    "Stability AI",
    "Mistral AI",
    "Hugging Face",
    "Scale AI",
    "Nvidia",
    "IBM Watson",
]

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "GraphRAGCourseBot/1.0 (educational corpus; Python urllib)"
REQUEST_PAUSE_SEC = 0.35


def api_get(params: dict) -> dict:
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def slugify(title: str) -> str:
    s = title.replace(" ", "_")
    s = re.sub(r"[^\w\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120] or "article"


def fetch_one_extract(title: str) -> tuple[str, str, str, str]:
    """Returns (canonical_title, extract, fullurl, pageid)."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "inprop": "url",
        "explaintext": "1",
        "exsectionformat": "plain",
        "redirects": "1",
        "titles": title,
        "exchars": "120000",
    }
    data = api_get(params)
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return title, "", "", ""
    page = next(iter(pages.values()))
    if "missing" in page:
        return title, "", "", ""
    ctitle = page.get("title", title)
    extract = page.get("extract") or ""
    url = page.get("fullurl") or (
        f"https://en.wikipedia.org/wiki/{urllib.parse.quote(ctitle.replace(' ', '_'))}"
    )
    pid = str(page.get("pageid", ""))
    return ctitle, extract, url, pid


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for requested in ARTICLE_TITLES:
        try:
            resolved, extract, url, pageid = fetch_one_extract(requested)
        except Exception as e:
            print(f"[error] {requested}: {e}")
            continue

        if not extract.strip():
            print(f"[warn] empty extract: {requested} -> {resolved}")
        slug = slugify(resolved)
        path = OUT_DIR / f"{slug}.txt"
        header = (
            f"Title: {resolved}\n"
            f"Requested: {requested}\n"
            f"Page ID: {pageid}\n"
            f"URL: {url}\n"
            f"Source: Wikipedia (en), plain-text extract via MediaWiki API\n"
            f"---\n\n"
        )
        path.write_text(header + extract, encoding="utf-8")
        manifest.append(
            {
                "requested_title": requested,
                "resolved_title": resolved,
                "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "url": url,
                "page_id": pageid,
                "chars": len(extract),
            }
        )
        print(f"OK {resolved} ({len(extract)} chars) -> {path.name}")
        time.sleep(REQUEST_PAUSE_SEC)

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nmanifest -> {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()