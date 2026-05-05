"""
Trích xuất thực thể–quan hệ (indexing): gọi LLM đọc từng tài liệu corpus và trả về bộ ba (subject, predicate, object).

Biến môi trường: OPENAI_API_KEY (có thể đặt trong file .env ở thư mục gốc dự án).
Chạy: python scripts/extract_triples.py [--model gpt-4o-mini] [--corpus-dir ...] [--out-dir ...]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)
DEFAULT_CORPUS = ROOT / "data" / "wikipedia_ai_companies"
DEFAULT_OUT = ROOT / "data" / "indexed"

SYSTEM = """You extract a knowledge graph from the given text. Return strict JSON only."""

USER_TEMPLATE = """Read the passage and extract factual triples for a knowledge graph.

Output JSON with this shape:
{{"triples": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}

Rules:
- Only facts clearly supported by the passage (no guessing).
- "subject" and "object": short noun phrases in English, no pronouns; keep names as in the text.
- "predicate": UPPER_SNAKE_CASE English, e.g. FOUNDED_IN, FOUNDED_BY, HEADQUARTERED_IN, CEO_OF, DEVELOPED_PRODUCT, OWNED_BY, INVESTMENT_FROM, SUBSIDIARY_OF, LOCATED_IN, VALUED_AT, RELEASED_IN, LEGAL_ACTION_AGAINST, RELATED_TO.
- Years and amounts as literal strings in "object" (e.g. "2015", "$500 billion").
- Merge duplicate or trivially redundant triples.
- If the passage is very short, still extract every explicit relation you can.

Passage:
---
{text}
---
"""


def rel_to_root(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def strip_corpus_header(raw: str) -> str:
    if "---" in raw:
        parts = raw.split("---", 1)
        if len(parts) == 2:
            return parts[1].lstrip("\n")
    return raw


def list_corpus_files(corpus_dir: Path) -> list[Path]:
    files = sorted(corpus_dir.glob("*.txt"))
    return [p for p in files if p.name.lower() != "readme.txt"]


def call_llm(client: OpenAI, model: str, text: str) -> list[dict]:
    user = USER_TEMPLATE.format(text=text.strip())
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    content = resp.choices[0].message.content or "{}"
    data = json.loads(content)
    triples = data.get("triples")
    if not isinstance(triples, list):
        return []
    out: list[dict] = []
    for t in triples:
        if not isinstance(t, dict):
            continue
        s, p, o = t.get("subject"), t.get("predicate"), t.get("object")
        if not all(isinstance(x, str) and x.strip() for x in (s, p, o)):
            continue
        out.append(
            {
                "subject": s.strip(),
                "predicate": p.strip().upper().replace(" ", "_"),
                "object": o.strip(),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM triple extraction for Wikipedia AI corpus")
    ap.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", default=os.environ.get("OPENAI_TRIPLE_MODEL", "gpt-4o-mini"))
    ap.add_argument("--sleep", type=float, default=0.4, help="Seconds between API calls")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY in environment.", file=sys.stderr)
        return 1

    corpus_dir: Path = args.corpus_dir
    if not corpus_dir.is_dir():
        print(f"Corpus dir not found: {corpus_dir}", file=sys.stderr)
        return 1

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "triples.jsonl"
    summary_path = out_dir / "extraction_summary.json"

    client = OpenAI()
    files = list_corpus_files(corpus_dir)
    if not files:
        print(f"No .txt files in {corpus_dir}", file=sys.stderr)
        return 1

    all_rows: list[dict] = []
    summary_docs: list[dict] = []

    # Ghi JSONL mới mỗi lần chạy (indexing lại từ đầu)
    jsonl_path.write_text("", encoding="utf-8")

    for path in files:
        raw = path.read_text(encoding="utf-8")
        body = strip_corpus_header(raw)
        title_match = re.search(r"^Title:\s*(.+)$", raw, re.MULTILINE)
        doc_title = title_match.group(1).strip() if title_match else path.stem

        try:
            triples = call_llm(client, args.model, body)
        except Exception as e:
            print(f"[error] {path.name}: {e}", file=sys.stderr)
            summary_docs.append(
                {"source_file": path.name, "doc_title": doc_title, "error": str(e), "triple_count": 0}
            )
            time.sleep(args.sleep)
            continue

        for tr in triples:
            row = {
                "source_file": path.name,
                "doc_title": doc_title,
                "subject": tr["subject"],
                "predicate": tr["predicate"],
                "object": tr["object"],
            }
            all_rows.append(row)
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary_docs.append(
            {
                "source_file": path.name,
                "doc_title": doc_title,
                "triple_count": len(triples),
                "model": args.model,
            }
        )
        print(f"OK {path.name}: {len(triples)} triples")
        time.sleep(args.sleep)

    summary = {
        "model": args.model,
        "corpus_dir": rel_to_root(corpus_dir),
        "documents": len(files),
        "total_triples": len(all_rows),
        "per_document": summary_docs,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {jsonl_path} ({len(all_rows)} lines)")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())