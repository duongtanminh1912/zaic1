# LifeMate public52 — Docker bài nộp chính thức

ASR PhoWhisper-large FP16 giữ nguyên50,9; Qwen3.5-4B đã qua DPO63updates sau
SFT489updates. DPO dùng63approved pairs/189replay rows TRAIN-only. Deployed Qwen
text-only Q4_K_M4.205.751.296params + ASR1.543.304.960 =5.749.056.256params.
Chi tiết training, augmentation, review, seed/config và base URLs/revisions ở
`training_code/baseline_52_release/TRAINING_REPRODUCTION.md` và training/*.

Qwen inference SHA256 aee8106ebe960c26e905443fedb3e8bd2e3a6c1218e63806892f849dbb8e512e.
Parent GGUF50,9 thuộc dữ liệu TRAIN harvest, không dùng deployed52. Manifest52
khóa đúng weights/processor/retrieval và từ chối parent. Điểm52 là nhóm báo từ
portal, không phải điểm được đo lại bởi bộ đóng gói này.

Inference offline: serial ASR batch2→CPU→Qwen GPU từng mẫu, greedy/thinking off,
seed20261103, context16384/output512, guards8calls/2repairs, deadline30s gồm ASR,
transfers, Qwen startup/shutdown, tools và hash. Chưa warmup TRAIN, chưa đo GPU
Linux trong lần tích hợp. Giữ mọi user-history audio, không truncate hoặc gold
fallback. Retrieval TF-IDF1shot TRAIN-fit và temporal helper giữ nguyên.

Input `/data/<split>` và SDK/catalog của split do BTC mount. Một ToolEnv/mẫu,
không đọc state trực tiếp/replay history. predict.sh ghi submission.jsonl;
notebook Run All ghi jupyter_submission.jsonl, đều ở /result, có time(ms).
Stdout model_load_ms/predict_ms; JupyterLab9777/tokenzac2026.

training_code có full source52 + DATA_ACCESS.json chứa link/checksum hai ZIP
training03+04; không đóng thêm base weights hoặc evaluation traces vào inference.
Dependencies training5.17.0 tạo venv riêng với inference4.57.1. Reproduction
Windows/Linux adaptations/giới hạn ghi rõ trong training README.

Hướng dẫn build/test/export/load/MD5/nộp:
https://github.com/duongtanminh1912/zaic1/blob/main/ops/packaging/RTX4060_SUBMISSION_52.md

Bộ code chưa build image/chạy GPU hoặc submit portal. Kiểm trên4060 và rehearsal
archive vẫn cần thực hiện theo hướng dẫn. Không dùng MD5/image52 giả từ bản50,9.
