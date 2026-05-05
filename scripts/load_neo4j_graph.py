"""
Bước 2 — Build graph: đọc triples.jsonl, nạp vào Neo4j, gán embedding (OpenAI) cho các node.

Biến môi trường:
  NEO4J_URI      (mặc định bolt://localhost:7687)
  NEO4J_USER     (mặc định neo4j)
  NEO4J_PASSWORD (bắt buộc)
  OPENAI_API_KEY (bắt buộc cho embedding)
  OPENAI_EMBEDDING_MODEL (mặc định text-embedding-3-small)

Có thể đặt biến trong file .env ở thư mục gốc dự án (xem env.sample).

Chạy sau khi có data/indexed/triples.jsonl (scripts/extract_triples.py):
  python scripts/load_neo4j_graph.py
  python scripts/load_neo4j_graph.py --wipe   # xóa toàn bộ graph trước khi nạp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)
DEFAULT_TRIPLES = ROOT / "data" / "indexed" / "triples.jsonl"


def load_triples(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def unique_entities(triples: list[dict]) -> list[str]:
    s: set[str] = set()
    for t in triples:
        s.add(t["subject"])
        s.add(t["object"])
    return sorted(s)


def embed_batch(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    r = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in r.data]


def ensure_constraint(session) -> None:
    session.run(
        """
        CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
        FOR (e:Entity) REQUIRE e.name IS UNIQUE
        """
    )


def wipe_graph(session) -> None:
    session.run("MATCH (n) DETACH DELETE n")


def upsert_triples(session, triples: list[dict], chunk: int = 250) -> None:
    q = """
    UNWIND $rows AS row
    MERGE (s:Entity {name: row.subject})
    MERGE (o:Entity {name: row.object})
    MERGE (s)-[r:RELATES_TO {predicate: row.predicate}]->(o)
    SET r.source_file = row.source_file,
        r.doc_title = row.doc_title
    """
    for i in range(0, len(triples), chunk):
        batch = triples[i : i + chunk]
        rows = [
            {
                "subject": t["subject"],
                "object": t["object"],
                "predicate": t["predicate"],
                "source_file": t.get("source_file", ""),
                "doc_title": t.get("doc_title", ""),
            }
            for t in batch
        ]
        session.run(q, rows=rows)


def set_embeddings(
    session, names: list[str], vectors: list[list[float]], model: str, chunk: int = 100
) -> None:
    q = """
    UNWIND $rows AS row
    MATCH (e:Entity {name: row.name})
    SET e.embedding = row.embedding,
        e.embedding_model = $model,
        e.embedding_dim = size(row.embedding)
    """
    for i in range(0, len(names), chunk):
        rows = [{"name": n, "embedding": v} for n, v in zip(names[i : i + chunk], vectors[i : i + chunk])]
        session.run(q, rows=rows, model=model)


def main() -> int:
    ap = argparse.ArgumentParser(description="Load triples into Neo4j and attach node embeddings")
    ap.add_argument("--triples", type=Path, default=DEFAULT_TRIPLES)
    ap.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    ap.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", ""))
    ap.add_argument(
        "--embedding-model",
        default=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--wipe", action="store_true", help="Delete all nodes/relationships first")
    args = ap.parse_args()

    if not args.neo4j_password:
        print("Set NEO4J_PASSWORD (or pass --neo4j-password).", file=sys.stderr)
        return 1
    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY for embeddings.", file=sys.stderr)
        return 1
    if not args.triples.is_file():
        print(f"Triples file not found: {args.triples}", file=sys.stderr)
        print("Run: python scripts/extract_triples.py", file=sys.stderr)
        return 1

    triples = load_triples(args.triples)
    if not triples:
        print("triples.jsonl is empty.", file=sys.stderr)
        return 1

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    oai = OpenAI()

    try:
        with driver.session() as session:
            if args.wipe:
                wipe_graph(session)
                print("Wiped graph.")
            ensure_constraint(session)
            print(f"Upserting {len(triples)} triples...")
            upsert_triples(session, triples)

        entities = unique_entities(triples)
        print(f"Embedding {len(entities)} entities ({args.embedding_model})...")

        all_vecs: list[list[float]] = []
        bs = max(1, args.batch_size)
        for i in range(0, len(entities), bs):
            batch = entities[i : i + bs]
            all_vecs.extend(embed_batch(oai, args.embedding_model, batch))
            time.sleep(0.05)

        with driver.session() as session:
            set_embeddings(session, entities, all_vecs, args.embedding_model)

        print(f"Done. Nodes: {len(entities)}, relationships: {len(triples)} (typed RELATES_TO + predicate).")
    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())