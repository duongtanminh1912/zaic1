# LifeMate baseline 50,9 — Docker bài nộp

Pipeline: trained PhoWhisper-large (1.543.304.960 parameters) → Qwen3.5-4B
text-only GGUF Q4_K_M (4.205.751.296 parameters) → native tool-calling agent.
Context16.384, output512tokens/turn, greedy, thinking off, seed20261103;
8calls/2repairs guards. ASR FP16, Vietnamese transcription, beams1, batch2.
Serial ASR GPU→CPU, Qwen startup/agent/shutdown trong deadline30s/mẫu.
Port chưa warmup TRAIN; chưa kiểm GPU timing/model parity trong lần tích hợp.

Input mount `/data/<split>` gồm JSONL, audio, catalog và SDK BTC. Dùng SDK mount,
một ToolEnv/mẫu, không đọc state trực tiếp hoặc replay history. Offline inference,
model/processor/retrieval nằm sẵn trong image. Retrieval TF-IDF một ví dụ TRAIN-fit
và helper ngày/giờ V1 của baseline. Không dùng API ngoài.

`predict.sh` ghi `/result/submission.jsonl`; notebook Run All ghi
`/result/jupyter_submission.jsonl`, gồm id/tool_calls/state_hash/time(ms).
Stdout ghi model_load_ms/predict_ms. JupyterLab port9777, token zac2026.

## Training và tài nguyên tái lập

`training_code/README.md` hướng dẫn bộ tái lập mới;
`training_code/DATA_ACCESS.json` chứa URL và SHA256 ZIP dữ liệu cho BTC tải.
Source nguyên vẹn ở `training_code/baseline_50_9_training_repro/`, có base
URLs/revisions, environment/requirements, recipe và CLI prepare/train/merge/export.
Không đóng base weights chỉ phục vụ training vào image inference.

TRAIN-fit1.236/VAL309 tách theo nhóm trước augmentation. Qwen3.906/963 examples,
QLoRA r16/alpha32,489updates; ASR3.997/1.039 audio, LoRA r32/alpha64,250updates,
chọn best bằng generated VAL WER. Augmentation dùng noise TRAIN-fit với semantic
guards; không ép app-mask ratio khi không đủ điều kiện. Seed training20261103.
VAL có dùng phát triển/chọn checkpoint. Chi tiết trong training README/recipe.

Đã kiểm toàn archive và dựng lại8inputs/retrieval byte-identical trên CPU.
Chưa train lại, merge/export mới hoặc xác nhận Linux training. Training dùng
môi trường riêng (Transformers5.17.0); inference dùng4.57.1. Không cài training
dependencies vào venv inference. README training ghi rõ Windows/Linux adaptation.

## Thực hiện trên RTX4060

Xem `ops/packaging/RTX4060_SUBMISSION_50_9.md` trong repo:
https://github.com/duongtanminh1912/zaic1/blob/main/ops/packaging/RTX4060_SUBMISSION_50_9.md

Model/private assets cần giải nén ngoài Git. Tạo context với prepare_context.py
và `--training-data-url` là URL ZIP training đúng bản cho BTC truy cập. Helper
kiểm model/runtime/source hashes, copy đúng training package và giữ checksum
LF/CRLF. ASR Large có index safetensors +2shards, không phải file weights đơn.

Chạy offline predict/notebook, kiểm output/logs, export image, load/chạy lại trên
máy khác, lấy MD5 chính archive gzip, rồi nộp link Drive file+MD5 theo BTC.
GPU/build/export/nộp chưa được thực hiện bởi bước chuẩn bị code này.
