# Training code/data — bài nộp LifeMate 50,9

Source tái lập đầy đủ ở `baseline_50_9_training_repro/`. `DATA_ACCESS.json` ghi
URL archive training, dung lượng và SHA256; BTC cần quyền tải file đó. Không
thay dữ liệu/model tại URL sau deadline. Không đưa ZIP/data private vào Git.

## Nhận dữ liệu và chạy

1. Dùng môi trường training riêng; giữ nguyên môi trường inference của image.
   Lịch sử dùng Windows/Python3.13, torch2.8.0+cu128, transformers5.17.0 và các
   pin trong ENVIRONMENT.json/requirements-training.txt. Inference dùng4.57.1.
2. Tải file ở DATA_ACCESS.json ra ngoài checkout. Đối chiếu SHA256 toàn ZIP.
3. Từ thư mục `/code/training_code` (hoặc bản copy training_code ra môi trường
   training), tạo file checksums theo cấu trúc công cụ unpack mong đợi:

```python
import json
from pathlib import Path
access = json.loads(Path('DATA_ACCESS.json').read_text())
Path('checksums.local.json').write_text(json.dumps({'files': {
    access['archive']: {'bytes': access['bytes'], 'sha256': access['sha256']}
}}))
```

4. Từ `baseline_50_9_training_repro/`, chạy unpack theo README của package với
   `--checksums ../checksums.local.json`, chọn private/work mới ngoài code.
5. Lấy đúng base snapshots theo URLs/revisions/file hashes ở BASE_MODELS.json,
   sửa config.local.json theo config.example.json. Base weights training không
   đóng vào image inference.
6. Làm đúng thứ tự README package: verify → prepare → retrieval → profile →
   train-qwen smoke/full → train-asr smoke/full → merge → export.

Dữ liệu:1.236 TRAIN-fit/309 VAL không trùng nhóm; Qwen3.906/963 examples;
ASR3.997/1.039 clips. Seed20261103. VAL dùng chọn checkpoint, không blind test.
Qwen QLoRA r16/alpha32,489steps; ASR Large LoRA r32/alpha64,250steps.

Prepared config lịch sử ghi Medium cho lần chuẩn bị chung; wrapper mới khóa
PhoWhisper-large theo recipe riêng. Không chạy trực tiếp scripts trong history.
Không resume checkpoint gốc bằng wrapper mới; dùng checkpoint do wrapper tạo.

Linux: cần Triton/FLA wheels và kernel probe phù hợp, quantizer Linux b11800
cùng SHA256 riêng khi export. Windows quantizer trong ZIP không chạy trên Linux.
Không mặc định Linux retraining hay GGUF byte parity đã được xác minh.

## Bằng chứng hiện có và giới hạn

Đã đọc CRC/SHA256 toàn bộ ZIP5.237members, kiểm audio/split/adapters và chạy CPU
verify/prepare/retrieval:8input files dựng lại byte-identical. Chưa chạy token
profile từ base snapshot, GPU retraining hoặc merge/export mới trong lần tích
hợp này. Chi tiết nguồn, thay đổi wrapper và giới hạn nằm trong PROVENANCE.json,
VALIDATION.json và README package. Không gọi các kiểm CPU là đã train lại đạt.
