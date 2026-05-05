"""
Evaluation: so sánh FlatRAG vs GraphRAG trên 5 câu hỏi phức tạp.

FlatRAG:
- retrieval theo embedding trên corpus text thuần
- answer từ các đoạn top-k

GraphRAG:
- extract entity trọng tâm từ câu hỏi
- resolve node trong Neo4j + traverse 2-hop
- textualization triples rồi answer

Output:
- data/eval/eval_results.json
- data/eval/eval_report.md
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)

CORPUS_DIR = ROOT / "data" / "wikipedia_ai_companies"
OUT_DIR = ROOT / "data" / "eval"

LLM_MODEL = os.environ.get("OPENAI_TRIPLE_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")


QUESTIONS = [
    "To chuc nao trong corpus duoc neu ro la da sat nhap voi Google Brain, va sat nhap vao thoi diem nao?",
    "Co phai Anthropic duoc thanh lap boi Sam Altman khong? Neu sai thi ai la nguoi sang lap theo du lieu?",
    "Trong corpus, cong ty nao bi noi la da tung bi board sa thai lanh dao vao nam 2023? Mo ta su kien gon.",
    "So sanh OpenAI va IBM Watson: ben nao co su kien co lien quan den vu kien ban quyen, ben nao co spin-off bo phan suc khoe?",
    "Neu xet OpenAI, Anthropic, Mistral AI: hay chi ra moc thanh lap va valuation/so tien duoc neu ro trong du lieu, khong suy doan.",
]


EXTRACT_SYSTEM = "Extract target entities from a user question. Return strict JSON."
EXTRACT_USER = """Question: {question}

Return JSON only:
{{"entities": ["...", "..."], "keywords": ["...", "..."]}}
"""

ANSWER_SYSTEM = "You are a helpful QA assistant. Use only provided context. If insufficient evidence, say clearly."
ANSWER_USER = """Question:
{question}

Context:
{context}

Answer in Vietnamese."""

JUDGE_SYSTEM = """You are an evaluator for RAG systems.
Judge factual support carefully from given evidence. Return strict JSON."""
JUDGE_USER = """Question:
{question}

FlatRAG answer:
{flat_answer}

FlatRAG evidence:
{flat_context}

GraphRAG answer:
{graph_answer}

GraphRAG evidence:
{graph_context}

Evaluation policy:
- Mark flat_has_hallucination=true if FlatRAG contains ANY key claim not supported by its own evidence.
- Be strict on names, founders, years, ownership and causal claims.
- graph_is_better=true when GraphRAG is more faithful to evidence or clearly avoids unsupported claims.

Return JSON:
{{
  "flat_has_hallucination": true/false,
  "graph_is_better": true/false,
  "winner": "flat|graph|tie",
  "reason": "short reason in Vietnamese"
}}
"""


@dataclass
class RagResult:
    answer: str
    context: str


def strip_header(raw: str) -> str:
    if "---" in raw:
        return raw.split("---", 1)[1].strip()
    return raw.strip()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def chunk_text(text: str, max_chars: int = 600, overlap: int = 100) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks


def load_corpus_chunks() -> list[dict]:
    rows = []
    for p in sorted(CORPUS_DIR.glob("*.txt")):
        raw = p.read_text(encoding="utf-8")
        body = strip_header(raw)
        for idx, ch in enumerate(chunk_text(body)):
            rows.append({"source": p.name, "chunk_id": idx, "text": ch})
    return rows


def llm_json(client: OpenAI, model: str, system: str, user: str) -> dict:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(r.choices[0].message.content or "{}")


def llm_answer(client: OpenAI, model: str, question: str, context: str) -> str:
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": ANSWER_USER.format(question=question, context=context)},
        ],
        temperature=0.2,
    )
    return (r.choices[0].message.content or "").strip()


def run_flatrag(client: OpenAI, question: str, corpus_chunks: list[dict], k: int = 5) -> RagResult:
    q_vec = client.embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding

    # Embed all chunks in one call for consistency (small corpus).
    texts = [r["text"] for r in corpus_chunks]
    vecs = client.embeddings.create(model=EMBED_MODEL, input=texts).data

    scored = []
    for row, v in zip(corpus_chunks, vecs):
        scored.append((cosine(q_vec, v.embedding), row))
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:k]

    context_lines = []
    for score, row in top:
        context_lines.append(f"[{row['source']}#{row['chunk_id']} score={score:.3f}] {row['text']}")
    context = "\n".join(context_lines)
    answer = llm_answer(client, LLM_MODEL, question, context)
    return RagResult(answer=answer, context=context)


def extract_entities(client: OpenAI, question: str) -> tuple[list[str], list[str]]:
    d = llm_json(client, LLM_MODEL, EXTRACT_SYSTEM, EXTRACT_USER.format(question=question))
    entities = d.get("entities", [])
    if not isinstance(entities, list):
        entities = []
    clean_entities = [str(e).strip() for e in entities if str(e).strip()]
    kws = d.get("keywords", [])
    if not isinstance(kws, list):
        kws = []
    clean_keywords = [str(k).strip() for k in kws if str(k).strip()]
    return clean_entities[:5], clean_keywords[:8]


def resolve_one_entity(session, entity: str, keywords: list[str]) -> str | None:
    if entity.strip():
        row = session.run(
            "MATCH (e:Entity) WHERE toLower(e.name)=toLower($n) RETURN e.name AS n LIMIT 1", n=entity
        ).single()
        if row:
            return row["n"]
        row = session.run(
            """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($n)
            RETURN e.name AS n, size(e.name) AS l
            ORDER BY l ASC LIMIT 1
            """,
            n=entity,
        ).single()
        if row:
            return row["n"]
    for kw in keywords:
        row = session.run(
            """
            MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($kw)
            RETURN e.name AS n, size(e.name) AS l
            ORDER BY l ASC LIMIT 1
            """,
            kw=kw,
        ).single()
        if row:
            return row["n"]
    return None


def run_graphrag(client: OpenAI, driver, question: str) -> RagResult:
    entities, keywords = extract_entities(client, question)
    if not entities:
        entities = keywords[:2]
    with driver.session() as s:
        centers = []
        for ent in entities:
            c = resolve_one_entity(s, ent, keywords)
            if c and c not in centers:
                centers.append(c)
        if not centers:
            context = f"Entity extracts={entities!r}, keywords={keywords}. Khong tim thay node."
            answer = "Khong tim thay thuc the phu hop trong do thi de tra loi cau hoi nay."
            return RagResult(answer=answer, context=context)

        triples = []
        for center in centers:
            records = s.run(
                """
                MATCH p=(c:Entity{name:$center})-[r:RELATES_TO*1..2]-(n:Entity)
                UNWIND relationships(p) AS rel
                WITH DISTINCT rel
                RETURN startNode(rel).name AS s, rel.predicate AS p, endNode(rel).name AS o, rel.doc_title AS d
                LIMIT 160
                """,
                center=center,
            )
            triples.extend([{"s": r["s"], "p": r["p"], "o": r["o"], "d": r["d"] or ""} for r in records])

    dedup = {}
    for t in triples:
        key = (t["s"], t["p"], t["o"], t["d"])
        dedup[key] = t
    triples = list(dedup.values())[:300]

    lines = [f"centers={centers}", f"entities={entities}", f"keywords={keywords}", f"triples={len(triples)}"]
    for t in triples:
        doc = f" [doc={t['d']}]" if t["d"] else ""
        lines.append(f"- ({t['s']}, {t['p']}, {t['o']}){doc}")
    context = "\n".join(lines)
    answer = llm_answer(client, LLM_MODEL, question, context)
    return RagResult(answer=answer, context=context)


def judge_case(client: OpenAI, question: str, flat: RagResult, graph: RagResult) -> dict:
    d = llm_json(
        client,
        LLM_MODEL,
        JUDGE_SYSTEM,
        JUDGE_USER.format(
            question=question,
            flat_answer=flat.answer,
            flat_context=flat.context,
            graph_answer=graph.answer,
            graph_context=graph.context,
        ),
    )
    return {
        "flat_has_hallucination": bool(d.get("flat_has_hallucination", False)),
        "graph_is_better": bool(d.get("graph_is_better", False)),
        "winner": str(d.get("winner", "tie")),
        "reason": str(d.get("reason", "")),
    }


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY")
        return 1
    if not NEO4J_PASSWORD:
        print("Missing NEO4J_PASSWORD")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    corpus_chunks = load_corpus_chunks()

    results = []
    try:
        for q in QUESTIONS:
            print(f"\nQ: {q}")
            flat = run_flatrag(client, q, corpus_chunks)
            graph = run_graphrag(client, driver, q)
            judge = judge_case(client, q, flat, graph)
            item = {
                "question": q,
                "flatrag": {"answer": flat.answer, "context": flat.context},
                "graphrag": {"answer": graph.answer, "context": graph.context},
                "evaluation": judge,
            }
            results.append(item)
            print(f" -> winner={judge['winner']} | flat_hallu={judge['flat_has_hallucination']}")
    finally:
        driver.close()

    out_json = OUT_DIR / "eval_results.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown report focused on requested cases
    lines = [
        "# Evaluation Report: FlatRAG vs GraphRAG",
        "",
        f"- Questions: {len(QUESTIONS)}",
        f"- LLM model: `{LLM_MODEL}`",
        f"- Embedding model: `{EMBED_MODEL}`",
        "",
        "## Cases where FlatRAG hallucinated but GraphRAG was better",
        "",
    ]
    count = 0
    for i, r in enumerate(results, start=1):
        ev = r["evaluation"]
        if ev["flat_has_hallucination"] and ev["graph_is_better"]:
            count += 1
            lines.append(f"### Case {count} (Q{i})")
            lines.append(f"**Question:** {r['question']}")
            lines.append(f"**FlatRAG answer:** {r['flatrag']['answer']}")
            lines.append(f"**GraphRAG answer:** {r['graphrag']['answer']}")
            lines.append(f"**Reason:** {ev['reason']}")
            lines.append("")
    if count == 0:
        lines.append("Khong ghi nhan case nao thoa dieu kien tren trong 5 cau hoi nay.")
        lines.append("")

    lines.append("## Summary table")
    lines.append("")
    lines.append("| # | Winner | Flat hallucination | Graph better |")
    lines.append("|---|---|---|---|")
    for i, r in enumerate(results, start=1):
        ev = r["evaluation"]
        lines.append(f"| {i} | {ev['winner']} | {ev['flat_has_hallucination']} | {ev['graph_is_better']} |")

    out_md = OUT_DIR / "eval_report.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())