# Báo cáo ngắn: Chi phí xây dựng GraphRAG

## Phạm vi
Phân tích cho giai đoạn:
1) Trích xuất triples (`extract_triples.py`)  
2) Nạp đồ thị + gán embedding (`load_neo4j_graph.py`)

## Tóm tắt nhanh
- Dữ liệu: 10 bài Wikipedia (AI companies)
- Kết quả đồ thị: **~175 nodes**, **~159 edges**
- Thời gian indexing triples đo được: **~93 giây**
- Thời gian load graph + embedding: thường **vài chục giây đến ~1-2 phút** (phụ thuộc mạng/API)

## Ước lượng token usage
- **LLM trích xuất triples (chat model)**: khoảng **8k–20k tokens** tổng (input + output), là phần chi phí chính.
- **Embedding node names** (`text-embedding-3-small`): khoảng **500–2,000 tokens** (rẻ hơn nhiều so với chat extraction).

## Nhận xét chi phí
- Thành phần tốn chi phí nhất: **chat completion để trích xuất triples**.
- Thành phần tốn ít chi phí: **embedding cho node name** (chuỗi ngắn).
- Neo4j chủ yếu tốn compute/network, không phát sinh token cost theo kiểu LLM.

## Đề xuất tối ưu
- Rút gọn prompt và schema để giảm token input.
- Cache kết quả extraction, chỉ chạy lại tài liệu thay đổi.
- Chỉ embed node mới/thay đổi thay vì embed lại toàn bộ.
- Bật log usage (`prompt_tokens`, `completion_tokens`, `total_tokens`, thời gian) để theo dõi chi phí chính xác mỗi lần chạy.
