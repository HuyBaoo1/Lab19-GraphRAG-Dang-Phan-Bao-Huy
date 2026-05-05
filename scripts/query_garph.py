"""
Bước 3 — Querying:
- nhận câu hỏi người dùng
- dùng LLM trích xuất thực thể chính (entity focus)
- tìm node tương ứng trong Neo4j và duyệt lân cận trong phạm vi 2-hop
- textualization bối cảnh đồ thị rồi gửi lại cho LLM để trả lời

Chạy:
  python scripts/query_graph.py --question "Who founded OpenAI?"
  python scripts/query_graph.py  # interactive mode
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)

EXTRACT_SYSTEM = "Extract the primary target entity from a user question. Return strict JSON."
EXTRACT_USER = """Question: {question}

Return JSON only:
{{"entity": "...", "keywords": ["...", "..."]}}

Rules:
- "entity": most important concrete entity in question (company/person/product), short phrase.
- "keywords": 2-5 useful terms for fallback graph lookup.
- If unsure, still provide best guess."""

ANSWER_SYSTEM = """You are a graph-grounded QA assistant.
Answer only from provided graph context. If context is insufficient, say so briefly."""

ANSWER_USER = """User question:
{question}

Graph context:
{context}

Please answer in Vietnamese, concise but clear."""


def extract_focus_entity(client: OpenAI, model: str, question: str) -> tuple[str, list[str]]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": EXTRACT_USER.format(question=question.strip())},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    content = resp.choices[0].message.content or "{}"
    data = json.loads(content)
    entity = str(data.get("entity", "")).strip()
    keywords = data.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []
    clean_keywords = [str(k).strip() for k in keywords if str(k).strip()]
    return entity, clean_keywords


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def resolve_entity_node(session, entity: str, keywords: list[str]) -> str | None:
    if entity:
        row = session.run(
            """
            MATCH (e:Entity)
            WHERE toLower(e.name) = toLower($name)
            RETURN e.name AS name
            LIMIT 1
            """,
            name=entity,
        ).single()
        if row:
            return row["name"]

        row = session.run(
            """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($name)
            RETURN e.name AS name, size(e.name) AS ln
            ORDER BY ln ASC
            LIMIT 1
            """,
            name=entity,
        ).single()
        if row:
            return row["name"]

    terms = [t for t in keywords if t]
    for term in terms:
        row = session.run(
            """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($term)
            RETURN e.name AS name, size(e.name) AS ln
            ORDER BY ln ASC
            LIMIT 1
            """,
            term=term,
        ).single()
        if row:
            return row["name"]
    return None


def fetch_two_hop_subgraph(session, center_name: str, rel_limit: int = 200) -> list[dict]:
    result = session.run(
        """
        MATCH p = (c:Entity {name: $center})-[r:RELATES_TO*1..2]-(n:Entity)
        UNWIND relationships(p) AS rel
        WITH DISTINCT rel
        RETURN
          startNode(rel).name AS s,
          rel.predicate AS p,
          endNode(rel).name AS o,
          rel.doc_title AS doc
        LIMIT $lim
        """,
        center=center_name,
        lim=rel_limit,
    )
    rows = []
    for rec in result:
        rows.append(
            {
                "subject": rec["s"],
                "predicate": rec["p"],
                "object": rec["o"],
                "doc_title": rec["doc"] or "",
            }
        )
    return rows


def textualize(center: str, triples: list[dict]) -> str:
    if not triples:
        return f"Center node: {center}\nNo neighboring relation found within 2 hops."
    lines = [f"Center node: {center}", f"Neighbor triples within 2 hops: {len(triples)}"]
    for t in triples:
        doc = f" [doc={t['doc_title']}]" if t["doc_title"] else ""
        lines.append(f"- ({t['subject']}, {t['predicate']}, {t['object']}){doc}")
    return "\n".join(lines)


def answer_with_context(client: OpenAI, model: str, question: str, context: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": ANSWER_USER.format(question=question.strip(), context=context)},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def run_query(question: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str, llm_model: str) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY.", file=sys.stderr)
        return 1
    if not neo4j_password:
        print("Missing NEO4J_PASSWORD.", file=sys.stderr)
        return 1

    client = OpenAI()
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        entity, keywords = extract_focus_entity(client, llm_model, question)
        with driver.session() as session:
            center = resolve_entity_node(session, entity, keywords)
            if not center:
                print("Khong tim thay node phu hop trong do thi.")
                print(f"Entity extract: {entity!r}; keywords: {keywords}")
                return 2
            triples = fetch_two_hop_subgraph(session, center)

        context = textualize(center, triples)
        answer = answer_with_context(client, llm_model, question, context)

        print("\n=== Entity Extraction ===")
        print(f"entity={entity!r}")
        print(f"keywords={keywords}")
        print(f"resolved_node={center!r}")

        print("\n=== Graph Context (2-hop) ===")
        print(context)

        print("\n=== LLM Answer ===")
        print(answer)
        return 0
    finally:
        driver.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Query graph with entity extraction + 2-hop traversal + LLM answer")
    ap.add_argument("--question", help="Question from user")
    ap.add_argument("--model", default=os.environ.get("OPENAI_TRIPLE_MODEL", "gpt-4o-mini"))
    ap.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    ap.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", ""))
    args = ap.parse_args()

    if args.question:
        return run_query(args.question, args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.model)

    print("Interactive mode. Nhap cau hoi (go 'exit' de thoat).")
    while True:
        q = input("\nQuestion> ").strip()
        if not q:
            continue
        if normalize(q) in {"exit", "quit"}:
            return 0
        code = run_query(q, args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.model)
        if code not in (0, 2):
            return code


if __name__ == "__main__":
    raise SystemExit(main())