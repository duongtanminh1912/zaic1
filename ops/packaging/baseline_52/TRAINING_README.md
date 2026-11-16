# Training reproduction — LifeMate public52

Full source và hướng dẫn nằm trong `baseline_52_release/`, nhất là
`TRAINING_REPRODUCTION.md`, `training/base50_9/README.md` và hai reproduce.py.
`DATA_ACCESS.json` ghi URLs, SHA256 và dung lượng hai ZIP cho BTC tải:

- 03_TRAINING_BASE50_9.zip: raw TRAIN/audio, prepared SFT/ASR, split, noise,
  token cache, parent SFT/ASR adapters/processors, converter/quantizer.
- 04_TRAINING_DPO52.zip: preference/replay/review63, TRAIN rollout/SDK/state,
  reference cache, DPO adapter/tokenizer/checkpoint63 và parent GGUF cho harvest.

Tải và kiểm SHA256 ngoài Git; dùng release_tools.py unpack đúng tên archive theo
README release. Không đưa base weights chỉ dùng training vào inference image.
Base URLs/revisions/file hashes nằm trong training/base50_9/BASE_MODELS.json.
Giữ URLs/dữ liệu bất biến sau deadline và bảo đảm BTC truy cập được.

## Thứ tự

1. Môi trường training riêng theo ENVIRONMENT.json/requirements: lịch sử Windows,
   Python3.13, torch2.8.0+cu128, transformers5.17.0; không đổi inference4.57.1.
2. Parent: verify → prepare → retrieval → profile → train-qwen/asr smoke/full.
3. DPO: verify-data → rebuild-data (hoặc setup, không cả hai trên cùng work) →
   math/reference → smoke → train. Giữ review63/189replay, không ép draft64.
4. Merge adapter cuối vào đúng base, ASR FP16, Qwen BF16→b11800 Q4_K_M text-only,
   đủ32blocks. Linux cần quantizer b11800 đã pin riêng; .exe trong ZIP là Windows.

Seed20261103; parent Qwen SFT489steps, LoRA r16/alpha32; ASR250steps r32/alpha64;
DPO63steps LR2e-6/beta0.1/dropout0. TRAIN-fit1236/VAL309 tách nhóm. VAL dùng chọn
checkpoint/đánh giá, không gradient. ZIP05 chứa evaluation, không dùng train.

Không cần train lại để dùng model52 đã merge trong ZIP01. Parent GGUF50,9 ở
ZIP04 chỉ để tái lập TRAIN harvest, không được thay cho deployed52.

## Giới hạn

Source/data parity, provenance và kiểm đã chạy được ghi trong VALIDATION.json.
Bộ Ops mới kiểm source hashes, Drive inventory/pins, API/serial adapter CPU.
Chưa tải/đọc lại toàn bộ ZIP01/04 trong lần cập nhật này; prepare_context và
release_tools sẽ kiểm đủ từng byte trên máy build. Chưa chạy GPU/full retrain,
merge/export mới hay Linux training. Không dùng kết quả Docker50,9 chứng nhận52.
