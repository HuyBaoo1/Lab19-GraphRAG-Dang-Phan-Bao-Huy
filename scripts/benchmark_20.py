"""
Benchmark 20 questions (simple -> complex) for FlatRAG vs GraphRAG.

Outputs:
- data/eval/benchmark20_results.json
- data/eval/benchmark20_table.csv
- data/eval/benchmark20_report.md
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
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
    "OpenAI founded in year nào?",
    "Ai là người sáng lập Nvidia?",
    "Anthropic tập trung vào định hướng nào trong AI?",
    "Mistral AI đặt trụ sở ở đâu?",
    "IBM Watson ban đầu nổi tiếng với chương trình gì?",
    "Google DeepMind là công ty con của tập đoàn nào?",
    "OpenAI đã phát triển những dòng sản phẩm chính nào?",
    "Scale AI cung cấp các dịch vụ gì liên quan LLM?",
    "Hugging Face có hợp tác với AWS không?",
    "Nvidia có thị phần GPU rời khoảng bao nhiêu theo dữ liệu?",
    "Google DeepMind hình thành từ sự kiện nào và khi nào?",
    "OpenAI chuyển từ nonprofit sang cấu trúc hiện tại như thế nào?",
    "So sánh Anthropic và OpenAI về mốc thành lập và định vị sản phẩm.",
    "Công ty nào trong corpus có valuation được nêu rõ và con số là bao nhiêu?",
    "IBM Watson vì sao bị xem là không hiệu quả về tài chính theo dữ liệu?",
    "Nvidia liên quan gì đến làn sóng generative AI xét theo bối cảnh dữ liệu?",
    "Nếu hỏi về vụ kiện bản quyền liên quan dữ liệu huấn luyện thì công ty nào nổi bật?",
    "Từ góc nhìn dữ liệu hiện có, công ty nào vừa có sự kiện legal issue vừa có mốc gọi vốn/định giá lớn?",
    "Hãy đối chiếu OpenAI, Anthropic, Mistral AI theo founding year, valuation và định vị model.",
    "Với câu hỏi đa thực thể: mối liên hệ giữa OpenAI, Microsoft, Google DeepMind và Anthropic là gì?",
]

EXTRACT_SYSTEM = "Extract target entities from question. Return JSON."
EXTRACT_USER = """Question: {question}
Return JSON only:
{{"entities":["..."],"keywords":["..."]}}"""

ANSWER_SYSTEM = "Answer using only supplied context. If insufficient, say so."
ANSWER_USER = """Question:
{question}

Context:
{context}

Answer in Vietnamese."""

JUDGE_SYSTEM = "Evaluate FlatRAG vs GraphRAG based on provided evidence. Return JSON only."
JUDGE_USER = """Question:
{question}

Flat answer:
{flat_answer}
Flat context:
{flat_context}

Graph answer:
{graph_answer}
Graph context:
{graph_context}

Return JSON:
{{
  "winner": "flat|graph|tie",
  "flat_has_hallucination": true/false,
  "graph_has_hallucination": true/false,
  "reason": "ngan gon tieng Viet"
}}"""


def strip_header(raw: str) -> str:
    return raw.split("---", 1)[1].strip() if "---" in raw else raw.strip()


def chunk_text(text: str, size: int = 650, overlap: int = 120) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out = []
    i = 0
    while i < len(text):
        j = min(len(text), i + size)
        out.append(text[i:j])
        if j >= len(text):
            break
        i = max(0, j - overlap)
    return out


def load_corpus_chunks() -> list[dict]:
    rows = []
    for p in sorted(CORPUS_DIR.glob("*.txt")):
        body = strip_header(p.read_text(encoding="utf-8"))
        for idx, ch in enumerate(chunk_text(body)):
            rows.append({"source": p.name, "chunk_id": idx, "text": ch})
    return rows


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def llm_json(client: OpenAI, system: str, user: str) -> dict:
    r = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(r.choices[0].message.content or "{}")


def llm_answer(client: OpenAI, question: str, context: str) -> str:
    r = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": ANSWER_USER.format(question=question, context=context)},
        ],
        temperature=0.2,
    )
    return (r.choices[0].message.content or "").strip()


def run_flatrag(client: OpenAI, question: str, chunks: list[dict], k: int = 6) -> tuple[str, str]:
    qv = client.embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding
    ev = client.embeddings.create(model=EMBED_MODEL, input=[c["text"] for c in chunks]).data
    scored = []
    for c, d in zip(chunks, ev):
        scored.append((cosine(qv, d.embedding), c))
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
    context = "\n".join([f"[{c['source']}#{c['chunk_id']}:{s:.3f}] {c['text']}" for s, c in top])
    return llm_answer(client, question, context), context


def resolve_one(session, ent: str, kws: list[str]) -> str | None:
    if ent:
        row = session.run(
            "MATCH (e:Entity) WHERE toLower(e.name)=toLower($n) RETURN e.name AS n LIMIT 1", n=ent
        ).single()
        if row:
            return row["n"]
        row = session.run(
            """
            MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($n)
            RETURN e.name AS n, size(e.name) AS l ORDER BY l ASC LIMIT 1
            """,
            n=ent,
        ).single()
        if row:
            return row["n"]
    for kw in kws:
        row = session.run(
            """
            MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($kw)
            RETURN e.name AS n, size(e.name) AS l ORDER BY l ASC LIMIT 1
            """,
            kw=kw,
        ).single()
        if row:
            return row["n"]
    return None


def run_graphrag(client: OpenAI, driver, question: str) -> tuple[str, str]:
    d = llm_json(client, EXTRACT_SYSTEM, EXTRACT_USER.format(question=question))
    ents = d.get("entities", [])
    kws = d.get("keywords", [])
    if not isinstance(ents, list):
        ents = []
    if not isinstance(kws, list):
        kws = []
    ents = [str(e).strip() for e in ents if str(e).strip()][:5]
    kws = [str(k).strip() for k in kws if str(k).strip()][:8]
    if not ents:
        ents = kws[:2]

    with driver.session() as s:
        centers = []
        for e in ents:
            c = resolve_one(s, e, kws)
            if c and c not in centers:
                centers.append(c)
        if not centers:
            context = f"entities={ents}; keywords={kws}; not found."
            return "Khong tim thay thuc the phu hop trong do thi.", context

        triples = []
        for c in centers:
            rec = s.run(
                """
                MATCH p=(c:Entity{name:$c})-[r:RELATES_TO*1..2]-(n:Entity)
                UNWIND relationships(p) AS rel
                WITH DISTINCT rel
                RETURN startNode(rel).name AS s, rel.predicate AS p, endNode(rel).name AS o, rel.doc_title AS d
                LIMIT 160
                """,
                c=c,
            )
            triples.extend([{"s": r["s"], "p": r["p"], "o": r["o"], "d": r["d"] or ""} for r in rec])
    dedup = {(t["s"], t["p"], t["o"], t["d"]): t for t in triples}
    triples = list(dedup.values())[:300]
    lines = [f"centers={centers}", f"entities={ents}", f"keywords={kws}", f"triples={len(triples)}"]
    lines.extend([f"- ({t['s']}, {t['p']}, {t['o']}) [doc={t['d']}]" for t in triples])
    context = "\n".join(lines)
    return llm_answer(client, question, context), context


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
    chunks = load_corpus_chunks()
    results = []
    try:
        for i, q in enumerate(QUESTIONS, start=1):
            t0 = time.perf_counter()
            flat_ans, flat_ctx = run_flatrag(client, q, chunks)
            t1 = time.perf_counter()
            graph_ans, graph_ctx = run_graphrag(client, driver, q)
            t2 = time.perf_counter()
            judge = llm_json(
                client,
                JUDGE_SYSTEM,
                JUDGE_USER.format(
                    question=q,
                    flat_answer=flat_ans,
                    flat_context=flat_ctx,
                    graph_answer=graph_ans,
                    graph_context=graph_ctx,
                ),
            )
            item = {
                "id": i,
                "question": q,
                "flatrag": {"answer": flat_ans, "context": flat_ctx, "latency_sec": round(t1 - t0, 2)},
                "graphrag": {"answer": graph_ans, "context": graph_ctx, "latency_sec": round(t2 - t1, 2)},
                "evaluation": {
                    "winner": str(judge.get("winner", "tie")),
                    "flat_has_hallucination": bool(judge.get("flat_has_hallucination", False)),
                    "graph_has_hallucination": bool(judge.get("graph_has_hallucination", False)),
                    "reason": str(judge.get("reason", "")),
                },
            }
            results.append(item)
            print(
                f"Q{i:02d}: winner={item['evaluation']['winner']} | "
                f"flat_hallu={item['evaluation']['flat_has_hallucination']}"
            )
    finally:
        driver.close()

    json_path = OUT_DIR / "benchmark20_results.json"
    csv_path = OUT_DIR / "benchmark20_table.csv"
    md_path = OUT_DIR / "benchmark20_report.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "question",
                "winner",
                "flat_has_hallucination",
                "graph_has_hallucination",
                "flat_latency_sec",
                "graph_latency_sec",
                "reason",
            ]
        )
        for r in results:
            ev = r["evaluation"]
            w.writerow(
                [
                    r["id"],
                    r["question"],
                    ev["winner"],
                    ev["flat_has_hallucination"],
                    ev["graph_has_hallucination"],
                    r["flatrag"]["latency_sec"],
                    r["graphrag"]["latency_sec"],
                    ev["reason"],
                ]
            )

    flat_h = sum(1 for r in results if r["evaluation"]["flat_has_hallucination"])
    graph_h = sum(1 for r in results if r["evaluation"]["graph_has_hallucination"])
    graph_w = sum(1 for r in results if r["evaluation"]["winner"] == "graph")
    flat_w = sum(1 for r in results if r["evaluation"]["winner"] == "flat")
    tie_w = len(results) - graph_w - flat_w
    md = [
        "# Benchmark 20 Questions: FlatRAG vs GraphRAG",
        "",
        f"- Total questions: **{len(results)}**",
        f"- Winner: Graph={graph_w}, Flat={flat_w}, Tie={tie_w}",
        f"- Hallucination count: Flat={flat_h}, Graph={graph_h}",
        "",
        "## Comparison Table",
        "",
        "| # | Winner | Flat Hallucination | Graph Hallucination | Flat Latency(s) | Graph Latency(s) |",
        "|---|---|---|---|---:|---:|",
    ]
    for r in results:
        ev = r["evaluation"]
        md.append(
            f"| {r['id']} | {ev['winner']} | {ev['flat_has_hallucination']} | "
            f"{ev['graph_has_hallucination']} | {r['flatrag']['latency_sec']} | {r['graphrag']['latency_sec']} |"
        )
    md.append("")
    md.append("## Hallucination Cases (Flat true, Graph false)")
    md.append("")
    idx = 0
    for r in results:
        ev = r["evaluation"]
        if ev["flat_has_hallucination"] and not ev["graph_has_hallucination"]:
            idx += 1
            md.append(f"### Case {idx} - Q{r['id']}")
            md.append(f"**Question:** {r['question']}")
            md.append(f"**Reason:** {ev['reason']}")
            md.append("")
    if idx == 0:
        md.append("No strict case found in this run.")
        md.append("")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())