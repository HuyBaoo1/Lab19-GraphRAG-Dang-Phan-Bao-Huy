"""
Export knowledge graph from Neo4j for visualization.

Outputs:
- data/graph_export/graph.json      (nodes/edges for reuse)
- data/graph_export/graph.graphml   (open with Gephi / yEd)
- data/graph_export/graph.html      (interactive browser visualization)

Run:
  python scripts/export_neo4j_graph.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx
from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)

OUT_DIR = ROOT / "data" / "graph_export"


def fetch_graph(driver, limit_edges: int = 1500) -> tuple[list[dict], list[dict]]:
    with driver.session() as session:
        recs = session.run(
            """
            MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
            RETURN s.name AS s, r.predicate AS p, o.name AS o, r.doc_title AS doc
            LIMIT $lim
            """,
            lim=limit_edges,
        )
        edges = []
        node_names = set()
        for r in recs:
            s = r["s"]
            p = r["p"] or "RELATED_TO"
            o = r["o"]
            doc = r["doc"] or ""
            node_names.add(s)
            node_names.add(o)
            edges.append({"source": s, "target": o, "predicate": p, "doc_title": doc})
        nodes = [{"id": n, "label": n} for n in sorted(node_names)]
        return nodes, edges


def save_graphml(nodes: list[dict], edges: list[dict], path: Path) -> None:
    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n["id"], label=n["label"])
    for e in edges:
        g.add_edge(
            e["source"],
            e["target"],
            predicate=e["predicate"],
            doc_title=e["doc_title"],
            label=e["predicate"],
        )
    nx.write_graphml(g, path)


def html_template(graph_json: str) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>GraphRAG Knowledge Graph</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; }}
    #top {{ padding: 10px 14px; border-bottom: 1px solid #ddd; }}
    #network {{ width: 100vw; height: calc(100vh - 70px); }}
    .hint {{ color: #666; font-size: 13px; }}
  </style>
</head>
<body>
  <div id="top">
    <b>GraphRAG Knowledge Graph</b>
    <div class="hint">Drag nodes, zoom wheel, click edge to view predicate.</div>
  </div>
  <div id="network"></div>

  <script>
    const graph = {graph_json};
    const nodes = new vis.DataSet(graph.nodes.map(n => ({{
      id: n.id,
      label: n.label,
      shape: "dot",
      size: 12
    }})));
    const edges = new vis.DataSet(graph.edges.map((e, i) => ({{
      id: i + 1,
      from: e.source,
      to: e.target,
      label: e.predicate,
      arrows: "to",
      font: {{align: "middle", size: 10}},
      title: (e.doc_title ? ("doc: " + e.doc_title + "<br>") : "") + "predicate: " + e.predicate
    }})));

    const container = document.getElementById("network");
    const data = {{ nodes, edges }};
    const options = {{
      physics: {{ stabilization: false, solver: "forceAtlas2Based" }},
      interaction: {{ hover: true, navigationButtons: true }},
      edges: {{ smooth: true }},
      nodes: {{ color: {{ background: "#86b7ff", border: "#2f6fdb" }} }}
    }};
    new vis.Network(container, data, options);
  </script>
</body>
</html>
"""


def main() -> int:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD", "")
    if not pwd:
        print("Missing NEO4J_PASSWORD.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    try:
        nodes, edges = fetch_graph(driver)
    finally:
        driver.close()

    if not edges:
        print("No edges found in Neo4j graph.")
        return 1

    graph_json_obj = {"nodes": nodes, "edges": edges}
    graph_json = OUT_DIR / "graph.json"
    graph_graphml = OUT_DIR / "graph.graphml"
    graph_html = OUT_DIR / "graph.html"

    graph_json.write_text(json.dumps(graph_json_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    save_graphml(nodes, edges, graph_graphml)
    graph_html.write_text(html_template(json.dumps(graph_json_obj, ensure_ascii=False)), encoding="utf-8")

    print(f"Exported nodes={len(nodes)}, edges={len(edges)}")
    print(f"- {graph_json}")
    print(f"- {graph_graphml}")
    print(f"- {graph_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())