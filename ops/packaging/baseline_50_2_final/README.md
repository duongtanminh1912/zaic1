# LifeMate — PhoWhisper Medium V3 + Qwen3.5 V2

Đây là image inference của giải pháp50,2, bổ sung code/data access để tái lập
training. Điểm50,2 là kết quả public do đội báo cho bài nộp gốc. Image bổ sung
có danh tính/archive/MD5 mới; trạng thái Test Solution cũ không tự chứng nhận
image mới. Xem hồ sơ kiểm thực tế trong training_code/baseline_50_2_training_repro/VALIDATION.json.

## Model và cấu hình

| Thành phần | Cấu hình |
|---|---|
| ASR | vinai/PhoWhisper-medium, revision55a7e3eb6c906de891f8f06a107754427dd3be79,763.857.920 tham số; LoRA V3 đã merge FP16 |
| Agent | Qwen/Qwen3.5-4B, revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a; QLoRA V2 đã merge, GGUF Q4_K_M |
| Tham số | Full hai base5.303.123.456 (<6B); inference converter chỉ deploy text Qwen và ASR |
| Context/decode | Context16.384; output tối đa512tokens/lượt; greedy, temperature0, thinking off, seed20261103 |
| Ngân sách |30giây/mẫu, tối đa8tool calls và2lần sửa format/schema; ghi thời gian thực ms |

## Phương pháp

ASR phiên âm live audio hiện tại và audio user trong history bằng Medium FP16,
WAV16kHz mono PCM16, greedy batch4. Prompt giữ transcript gốc, bổ sung gợi ý
số/giờ và tối đa một ví dụ tương tự TRAIN-fit bằng word TF-IDF. Retrieval loại
trừ chính mẫu/cùng nhóm; chỉ đưa schema tools thuộc app đã cài. Không dùng
transcript chuẩn thay ASR trong inference.

Qwen sinh một next-action mỗi lượt bằng native tool calling. Agent validate rồi
thực thi ToolEnv thật, đọc observation trước khi quyết định tiếp, hỏi thiếu
thông tin hoặc dừng. History actions đã thực hiện không được chạy lại.
state_hash lấy từ SDK; không do LLM tự tạo. Một backend Qwen resident được
khởi động trước vòng lặp timed samples, đúng profile image50,2 gốc.

## Training/data

Chia theo nhóm trước mở rộng:1.236TRAIN-fit và309VAL, không trùng nhóm.
Qwen3.906TRAIN/963VAL next-action SFT, loss chỉ completion/EOS; QLoRA NF4/BF16
r16/alpha32/dropout0,05, LR1e-4, batch1/acc8,1epoch489updates.
ASR3.997TRAIN/1.039VAL audio clips; LoRA r32/alpha64/dropout0,05 vào attention
vàfc1/fc2, LR5e-4, batch1/acc32, warmup30,2epoch250updates; chọn theo generated
VAL WER. Seed20261103. VAL dùng phát triển/chọn checkpoint, không gradient.

Chuẩn bị Qwen dùng lịch sử assistant/tool thật, giữ input công cụ và nhãn gold;
noise transcript chỉ TRAIN-fit, được lọc bảo vệ số/phủ định/tên. Chọn30% nguồn
để thử noise và10% để thử bỏ tên app khi đủ điều kiện; tỷ lệ thay thực tế phụ
thuộc điều kiện. Không dùng output LLM làm gold, không đưa nhãn public/private
vào train. Retrieval chỉ từ TRAIN-fit.

Code đầy đủ, hai URL tải dữ liệu/checkpoints có checksum, base revisions và lệnh
prepare → train → merge/export/quantize ở `/code/training_code/README.md` và
`/code/training_code/DATA_ACCESS.json`. Đây là hồ sơ tái lập mới từ source lịch sử;
chưa thực hiện full retrain/merge lại trên GPU trong đợt đóng gói này.
Medium checkpoint250 hoàn tất optimizer updates; lỗi callback lưu trùng epoch
có RECOVERY riêng, giữ nguyên hồ sơ gốc và không chạy thêm bước251.

## Inference CLI và notebook

Mount split chính thức vào `/data:ro`, output vào `/result`, dùng GPU0:

```sh
docker run --rm --gpus '"device=0"' --network none \
  -v "$DATA:/data:ro" -v "$RESULT:/result" \
  zaic-team:50-2-final-training-v1 /bin/bash /code/predict.sh
```

CLI ghi `/result/submission.jsonl`, mỗi dòng có `id`, `tool_calls`, `state_hash`
và `time`ms. Dataset/SDK do BTC mount; weights và Linux llama.cpp b11800 đã có
trong image. Inference không cần mạng. Torch2.8.0+cu128, Transformers4.57.1,
CUDA base13.2, Linux/AMD64. Môi trường training được ghi riêng.

```sh
docker run --rm --gpus '"device=0"' -p 127.0.0.1:9777:9777 \
  -v "$DATA:/data:ro" -v "$RESULT:/result" \
  zaic-team:50-2-final-training-v1 /bin/bash /code/start_jupyter.sh
```

Mở localhost:9777, tokenzac2026, kernel mới, Run All predict_notebook.ipynb.
Notebook ghi jupyter_submission.jsonl; so ID/calls/hash với CLI và kiểm thời
gian thực từng mẫu. Mẫu smoke TRAIN-fit chỉ kiểm kỹ thuật; không chứng nhận
toàn bộ hidden split hoặc tốc độ trên GPU BTC.

## File nộp

Sau kiểm CLI/notebook, export đúng image đã chạy bằng docker save rồi gzip;
tính MD5 trên file `.tar.gz` cuối, docker load để kiểm archive. Upload đúng
archive lên Drive, cho phép BTC tải qua link và điền link file/MD5 vào nơi
nộp solution chính thức. Dùng MD5 mới của archive này.
