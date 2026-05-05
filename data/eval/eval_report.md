# Evaluation Report: FlatRAG vs GraphRAG

- Questions: 5
- LLM model: `gpt-4o-mini`
- Embedding model: `text-embedding-3-small`

## Cases where FlatRAG hallucinated but GraphRAG was better

### Case 1 (Q1)
**Question:** To chuc nao trong corpus duoc neu ro la da sat nhap voi Google Brain, va sat nhap vao thoi diem nao?
**FlatRAG answer:** Công ty được nhắc đến trong corpus là DeepMind, và nó đã sát nhập với Google Brain vào tháng 4 năm 2023.
**GraphRAG answer:** Trong corpus, tổ chức được nêu rõ là Google Brain, nhưng không có thông tin cụ thể về thời điểm sát nhập.
**Reason:** FlatRAG khẳng định rằng Google Brain đã sát nhập với DeepMind vào tháng 4 năm 2023, nhưng không có thông tin nào trong bằng chứng hỗ trợ cho tuyên bố này. Trong khi đó, GraphRAG chỉ nêu rõ Google Brain mà không đưa ra thông tin sai lệch.

### Case 2 (Q3)
**Question:** Trong corpus, cong ty nao bi noi la da tung bi board sa thai lanh dao vao nam 2023? Mo ta su kien gon.
**FlatRAG answer:** Trong corpus, không có thông tin nào đề cập đến công ty nào bị sa thải lãnh đạo vào năm 2023.
**GraphRAG answer:** Khong tim thay thuc the phu hop trong do thi de tra loi cau hoi nay.
**Reason:** FlatRAG đưa ra thông tin không chính xác về việc công ty nào bị sa thải lãnh đạo vào năm 2023, trong khi GraphRAG không đưa ra thông tin sai lệch.

### Case 3 (Q4)
**Question:** So sanh OpenAI va IBM Watson: ben nao co su kien co lien quan den vu kien ban quyen, ben nao co spin-off bo phan suc khoe?
**FlatRAG answer:** Dựa trên thông tin có sẵn, IBM Watson đã có sự kiện liên quan đến vụ kiện bản quyền và đã spin-off bộ phận sức khỏe. Cụ thể, vào năm 2022, IBM đã tách bộ phận Watson Health thành Merative và bán cho Francisco Partners. Trong khi đó, không có thông tin nào cho thấy OpenAI có sự kiện liên quan đến vụ kiện bản quyền hoặc spin-off bộ phận sức khỏe. 

Tóm lại:
- IBM Watson: Có vụ kiện bản quyền và đã spin-off bộ phận sức khỏe.
- OpenAI: Không có thông tin về vụ kiện bản quyền hay spin-off bộ phận sức khỏe.
**GraphRAG answer:** OpenAI có sự kiện liên quan đến vụ kiện bản quyền, trong khi IBM Watson có spin-off bộ phận sức khỏe (Watson Health division).
**Reason:** GraphRAG cung cấp thông tin chính xác về vụ kiện bản quyền của OpenAI và spin-off bộ phận sức khỏe của IBM Watson, trong khi FlatRAG có thông tin sai lệch về OpenAI.

### Case 4 (Q5)
**Question:** Neu xet OpenAI, Anthropic, Mistral AI: hay chi ra moc thanh lap va valuation/so tien duoc neu ro trong du lieu, khong suy doan.
**FlatRAG answer:** 1. **OpenAI**: Không có thông tin về ngày thành lập và định giá trong dữ liệu cung cấp.

2. **Anthropic**: 
   - Ngày thành lập: 2021
   - Định giá: 380 tỷ USD (tính đến tháng 2 năm 2026)

3. **Mistral AI**: 
   - Ngày thành lập: Tháng 4 năm 2023
   - Định giá: Hơn 14 tỷ USD (tính đến năm 2025)
**GraphRAG answer:** - OpenAI: Thành lập năm 2015, được định giá 500 tỷ USD.
- Anthropic: Thành lập năm 2021, được định giá 380 tỷ USD.
- Mistral AI: Không có thông tin về năm thành lập và định giá trong dữ liệu.
**Reason:** GraphRAG cung cấp thông tin chính xác về ngày thành lập và định giá của OpenAI, trong khi FlatRAG có thông tin không chính xác về Mistral AI và không có thông tin về OpenAI.

## Summary table

| # | Winner | Flat hallucination | Graph better |
|---|---|---|---|
| 1 | graph | True | True |
| 2 | graph | False | True |
| 3 | graph | True | True |
| 4 | graph | True | True |
| 5 | graph | True | True |