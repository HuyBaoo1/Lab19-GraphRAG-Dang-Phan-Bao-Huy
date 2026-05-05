# GraphRAG Lab (AI Company Corpus)

README này giúp người đọc đi nhanh qua toàn bộ kết quả lab theo 5 phần chính.

## 1) Mã nguồn

Các script chính trong `scripts/`:

- `scripts/fetch_wikipedia_ai_companies.py`: lấy dữ liệu Wikipedia (10 bài).
- `scripts/extract_triples.py`: trích xuất bộ ba `(subject, predicate, object)` bằng LLM.
- `scripts/load_neo4j_graph.py`: nạp triples vào Neo4j + gán embedding cho node.
- `scripts/query_graph.py`: truy vấn GraphRAG (entity extraction -> 2-hop traverse -> answer).
- `scripts/evaluate_rag.py`: đánh giá FlatRAG vs GraphRAG trên 5 câu hỏi phức tạp.
- `scripts/benchmark_20.py`: benchmark 20 câu hỏi và xuất bảng so sánh.
- `scripts/export_neo4j_graph.py`: export đồ thị Neo4j ra HTML/GraphML/JSON để trực quan.

## 2) Đồ thị tri thức đã xây dựng

Kết quả đồ thị từ Neo4j:

- `data/indexed/triples.jsonl`: dữ liệu triples sau indexing.
- `data/graph_export/graph.html`: bản trực quan tương tác (mở bằng browser).
- `data/graph_export/graph.graphml`: dùng cho Gephi / yEd / Cytoscape.
- `data/graph_export/graph.json`: dữ liệu node-edge để tái sử dụng.

Ghi chú: bạn có thể mở trực tiếp `data/graph_export/graph.html` để xem liên kết bằng mắt thường.

## 3) Eval chạy thử 5 câu hỏi phức tạp

Kết quả lưu tại:

- `data/eval/eval_results.json`
- `data/eval/eval_report.md`

Nội dung gồm:

- câu hỏi
- câu trả lời FlatRAG
- câu trả lời GraphRAG
- đánh giá winner / hallucination / reason

## 4) Bảng so sánh kết quả 20 benchmark giữa FlatRAG và GraphRAG

Kết quả benchmark 20 câu:

- `data/eval/benchmark20_results.json` (chi tiết từng câu)
- `data/eval/benchmark20_table.csv` (bảng để mở bằng Excel)
- `data/eval/benchmark20_report.md` (báo cáo tổng hợp markdown)

Các cột chính:

- winner
- flat hallucination / graph hallucination
- latency của mỗi hệ
- reason đánh giá

## 5) Bảng phân tích ngắn gọn chi phí

Phân tích ngắn đã tổng hợp tại:

- `data/eval/cost_analysis_brief.md`

Bao gồm:

- thời gian chạy chính
- ước lượng token usage
- nhận xét thành phần tốn chi phí
- đề xuất tối ưu chi phí

---

## Chạy lại nhanh (tuỳ chọn)

```bash
python scripts/fetch_wikipedia_ai_companies.py
python scripts/extract_triples.py
python scripts/load_neo4j_graph.py --wipe
python scripts/evaluate_rag.py
python scripts/benchmark_20.py
python scripts/export_neo4j_graph.py
```
