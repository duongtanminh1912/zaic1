# Training bàn giao cho Medium V3 + Qwen V2 (50,2)

Đây là phần bổ sung cho image `zaic-team:rtx4060-test1-fix2`. Dùng cùng thư mục
`baseline_50_9_training_repro` ở ngay bên cạnh để lấy helper V2, prepare, Qwen trainer,
converter và các pin môi trường. **Lệnh của thư mục này khóa ASR Medium; không chạy
`train-asr` từ thư mục 50,9 vì lệnh đó train Large.** Không có DPO52 ở pipeline này.

## Hai đầu vào private

1. `LIFEMATE_BASELINE_50_2_TRAINING_CHECKPOINTS.zip` (743.551.608 bytes), gồm
   Medium V3 bước250/250, Qwen V2 bước489/489, optimizer/scheduler/RNG và chứng cứ.
2. `LIFEMATE_50_9_TRAINING_PRIVATE.zip`, cũng được phát hành nguyên byte với tên
   `03_TRAINING_BASE50_9.zip` (2.775.709.761 bytes), để lấy raw TRAIN, audio,
   prepared SFT, noise, token cache, processor Qwen và converter/quantizer.

`ARCHIVES.json` khóa SHA256 hai ZIP. `verify` kiểm hash từng file và xác nhận split,
hai ASR manifests, Qwen data manifest và Qwen adapter khớp checkpoint50,2.
Đó là lý do có thể dùng lại dữ liệu chung; **adapter Large trong ZIP thứ hai không
được dùng cho Medium**. Hai ZIP phải được cung cấp riêng cho BTC bằng các URL tải
được; không đưa dữ liệu/weights vào Git. `DATA_ACCESS.json` trong image sẽ lưu URL
thực tế cùng byte count và checksum.

## Môi trường và base models

Python3.13.13/Windows là môi trường lịch sử. Dùng venv training riêng; pin package
trong `../baseline_50_9_training_repro/ENVIRONMENT.json` và requirements ở đó:

```text
python -m venv .venv-training
.venv-training/Scripts/python -m pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.venv-training/Scripts/python -m pip install -r ../baseline_50_9_training_repro/requirements-training.txt
```

Qwen cần FLA/Triton tương ứng nền tảng; xem README chung và chạy kernel probe.
GPU cần BF16 nguyên sinh. Chưa xác nhận full retrain trên Linux; môi trường
inference Transformers4.57.1 trong image không phải môi trường train5.17.0.

Tải snapshot theo tên/revision và đủ danh sách file trong `BASE_MODELS.json`.
Có thể dùng `huggingface_hub.snapshot_download(repo_id=..., revision=...,
local_dir=..., allow_patterns=[...])` ở môi trường có mạng. Sau đó `snapshot`
trong wrapper kiểm SHA256 từng file trước train/merge. Qwen revision851bf6e,
PhoWhisper-**medium** revision55a7e3eb; không dùng PhoWhisper-large.

## Các lệnh thực tế

Chạy từ thư mục chứa README này, Python>=3.11. Hai thư mục code phải ở cạnh nhau.
Giải nén vào hai nơi mới, copy `config.example.json` thành file config ngoài Git,
sửa đường dẫn absolute và chọn work mới, tách khỏi input:

```text
python -B reproduce.py unpack --role checkpoints --archive PATH/LIFEMATE_BASELINE_50_2_TRAINING_CHECKPOINTS.zip --dest PATH/checkpoints_50_2
python -B reproduce.py unpack --role shared_training --archive PATH/03_TRAINING_BASE50_9.zip --dest PATH/shared_training
python -B reproduce.py verify --config PATH/config.local.json
python -B reproduce.py prepare --config PATH/config.local.json
python -B reproduce.py retrieval --config PATH/config.local.json
python -B reproduce.py profile --config PATH/config.local.json
python -B reproduce.py train-qwen --config PATH/config.local.json --smoke
python -B reproduce.py train-qwen --config PATH/config.local.json
python -B reproduce.py train-asr --config PATH/config.local.json --smoke
python -B reproduce.py train-asr --config PATH/config.local.json
```

Prepare phải tái tạo đủ tám model input khớp byte, gồm ASR clips và retrieval
TRAIN-fit. Qwen profile phải khớp token cache; không cắt context. Seed20261103;
1.236 TRAIN-fit/309 VAL tách nhóm, 3.906/963 mẫu SFT và 3.997/1.039 audio clips.
Qwen QLoRA r16/alpha32, LR1e-4,1epoch/489updates; Medium LoRA r32/alpha64,
LR5e-4,2epoch/250updates, batch1/acc32, warmup30. Chi tiết ở training_recipe.json.
VAL dùng chọn checkpoint, không nhận gradient. Noise chỉ TRAIN-fit, không đưa
public/private labels hoặc output LLM vào train.

Smoke2updates độc lập với full run. Chỉ dùng `--resume` cho checkpoint do wrapper
mới tạo; không resume optimizer gốc bước250/489 để finalize. Giữ nguyên các ZIP
và hồ sơ lịch sử. Bộ bàn giao này không yêu cầu phải train lại trước khi inference.

## Merge/export

Dựng từ adapter lịch sử đã có:

```text
python -B reproduce.py merge-qwen --config PATH/config.local.json
python -B reproduce.py merge-asr --config PATH/config.local.json
python -B reproduce.py export --config PATH/config.local.json
```

Qwen merge CPU BF16 → converter b11800 `--outtype bf16 --no-nextn` → Q4_K_M.
Medium merge CPU FP16 từ **checkpoint Medium250 của ZIP thứ nhất**, dùng processor
Medium gốc theo ASR_PROCESSOR_COMPAT. Output ở `work/merged/{qwen,asr}` và
`work/exports/Qwen3.5-4B-LifeMate-V2-Q4_K_M.gguf`.

Merge adapter mới: truyền `--adapter-dir`. Qwen processor khai báo trong config;
Medium dùng processor base Medium đã pin. Qwen adapter mới ở
`work/qwen/outputs/llm_full/adapter_final`; Medium ở
`work/asr_medium/outputs/asr_full/best_adapter` (chọn theo VAL WER).
Quantizer Windows nằm trong ZIP chung; Linux phải cấu hình `quantizer` và
`quantizer_sha256` cho binary b11800 Linux. Không mặc định hash GGUF giống nhau
giữa nền tảng.

## Hồ sơ và giới hạn

Checkpoint Medium250 đủ 2epochs. RUN_MANIFEST lịch sử ghi lỗi callback
`Refuse overwrite epoch adapter`; RECOVERY xác nhận đã copy/merge checkpoint250
mà không thêm optimizer update. Trainer state, adapter hash và WER10,79196%
khớp hồ sơ. Wrapper ASR mới bỏ nhánh LLM V3 không dùng và chặn xử lý trùng callback
save trong cùng run; không sửa các file lịch sử, recipe hoặc weights gốc.

`PROVENANCE.json` mô tả thay đổi; `VALIDATION.json` phân biệt CPU/data verification
đã làm với full retrain/merge/export/GPU chưa làm. Bộ code có các bước tái lập,
nhưng chưa chứng nhận một lượt train mới ra weights/score giống hệt.
Không lấy trạng thái Test Solution cũ để chứng nhận image có lớp training mới.
