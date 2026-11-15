# Docker Test Solution — Large 50,9

Dùng code `llm/baseline_50_9_verified` từ nhánh feat/llm-dog, không dùng Medium
50,2. Đây là candidate tích hợp; chưa build/chạy weights trên NVIDIA GPU.

Cần Linux x86_64 hoặc WSL2 có NVIDIA GPU, Docker GPU hoạt động, Python >=3.11,
model ZIP Large 50,9 và dataset/SDK BTC. Mac ARM chỉ kiểm code/I/O, không chứng
nhận GPU. Không dùng runtime Windows ZIP trong Linux.

## Chuẩn bị ngoài Git

Chạy từ root repo, đặt TASKROOT vào ổ đủ chỗ; không để weights trong checkout.

```sh
export TASKROOT="$HOME/lifemate50-9"
mkdir -p "$TASKROOT/incoming"
# Chép MODELS_AND_PRIVATE_DATA.zip và SHA256SUMS.json từ Drive vào incoming.
python3 llm/baseline_50_9_verified/tools/unpack_assets.py \
  --archive "$TASKROOT/incoming/LIFEMATE_BASELINE_50_9_MODELS_AND_PRIVATE_DATA.zip" \
  --checksums "$TASKROOT/incoming/SHA256SUMS.json" \
  --dest "$TASKROOT/private_assets"
curl -fL --retry 2 \
  https://github.com/ggml-org/llama.cpp/releases/download/b11800/llama-b11800-bin-ubuntu-cuda-12.8-x64.tar.gz \
  -o "$TASKROOT/incoming/llama-linux.tar.gz"
python3 ops/packaging/baseline_50_9/stage_linux_runtime.py \
  "$TASKROOT/incoming/llama-linux.tar.gz" "$TASKROOT/linux_runtime"
```

Chuẩn bị `training_handoff/README.md` cùng source/tài nguyên train thực tế.
`training_reference/` chỉ là source tham khảo, không chứng nhận tái lập đầy đủ.
Để thử kỹ thuật có thể copy bộ tham khảo với README gốc ghi rõ giới hạn:

```sh
mkdir -p "$TASKROOT/training_handoff"
cp -R llm/baseline_50_9_verified/training_reference/. "$TASKROOT/training_handoff/"
cp llm/baseline_50_9_verified/training_recipe.json "$TASKROOT/training_handoff/"
python3 ops/packaging/baseline_50_9/prepare_context.py \
  --assets "$TASKROOT/private_assets" --runtime "$TASKROOT/linux_runtime" \
  --training "$TASKROOT/training_handoff" --destination "$TASKROOT/docker_context"
docker build --platform linux/amd64 -t lifemate:50-9-test "$TASKROOT/docker_context"
```

## Kiểm và chạy offline

`smoke_data/` chứa đúng một split BTC, có JSONL/catalog/audio/SDK/environment.
Ưu tiên vài mẫu TRAIN-fit để kiểm tích hợp trước. Không mount cả TRAIN rồi vô tình
chạy toàn bộ. Loader không đọc state/labels; ToolEnv của SDK quản lý session.

```sh
docker run --rm --gpus all --network none lifemate:50-9-test \
  python3 -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
mkdir -p "$TASKROOT/result"
docker run --rm --gpus all --network none \
  -v "$TASKROOT/smoke_data:/data:ro" -v "$TASKROOT/result:/result" \
  lifemate:50-9-test /bin/bash /code/predict.sh
```

Đầu ra `/result/submission.jsonl` gồm id/tool_calls/state_hash/time (ms), kèm
 diagnostics và log lỗi. Giữ output và kiểm ASR/Qwen thực chạy, đủ ID, không
asr/backend/OOM errors. Một file JSONL đủ dòng chưa chứng minh inference đạt.
Không ghi đè output hoặc tự chạy lại mẫu lỗi để chọn kết quả.

Notebook dùng cùng engine, serial placement và deadline. Lần port này không
warmup bằng TRAIN; đây là khác biệt startup được ghi `warmup=false`, cần đo và
đối chiếu trên GPU. Mỗi mẫu gồm ASR chuyển GPU/CPU, Qwen startup/shutdown, tools
và xuất session. Không đổi thành resident để cải thiện timing rồi gọi cùng profile.

```sh
docker run --rm --gpus all -p 127.0.0.1:9777:9777 \
  -v "$TASKROOT/smoke_data:/data:ro" -v "$TASKROOT/result:/result" \
  lifemate:50-9-test /bin/bash /code/start_jupyter.sh
```

Mở localhost:9777, token zac2026, notebook Run All từ kernel mới. So
jupyter_submission.jsonl với submission.jsonl theo ID/calls/hash; giữ cả lỗi và
thời gian. Kiểm pip check, lưu pip freeze/image ID/driver/VRAM và logs.

## Xuất archive sau khi chạy đạt

```sh
set -o pipefail
docker save lifemate:50-9-test | gzip > "$TASKROOT/lifemate50-9.tar.gz"
md5sum "$TASKROOT/lifemate50-9.tar.gz" > "$TASKROOT/lifemate50-9.tar.gz.md5"
docker load -i "$TASKROOT/lifemate50-9.tar.gz"
```

Upload archive lên Drive và gửi link file + MD5 của chính archive gzip vào Test
Solution. Build/test/export không tự submit portal hay dùng quota đội.
Theo tài liệu BTC đã lưu trong repo, contract là `/code/predict.sh`, mount
`/data`, ghi `/result/submission.jsonl`, inference offline, Jupyter port9777.
Nguồn: https://dl-challenge.zalo.ai/2026/lifemate_argfzdgert/Docker_Guide_ZAC2026_LifeMate.pdf

Trạng thái cần hoàn thành: build AMD64, smoke GPU thật, notebook parity,
rehearsal offline, export/load archive. Training reproduction đầy đủ là mục
riêng chưa được xác nhận bởi candidate kỹ thuật.
