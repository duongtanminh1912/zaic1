# RTX 4060 → Docker Test Solution — model 50,9, ngày 15/11/2026

Dùng lại máy Kiên đã chạy thành công ngày 14/11: Ubuntu-24.04/WSL2,
RTX4060 Laptop 8GB, Docker Desktop. Đây là bộ **code/hướng dẫn chuẩn bị**,
chưa phải báo cáo build hoặc chạy GPU thành công hôm nay.

Model hôm nay: trained PhoWhisper-large + Qwen3.5-4B-LifeMate-V2-Q4_K_M,
helper ngày/giờ V1, retrieval TRAIN-fit. Không dùng private_assets của Medium50,2.
Giữ image `zaic-team:rtx4060-test1-fix2`, archive và kết quả hôm qua.

## 1. Cài bộ code mới và đặt thư mục làm việc

Toàn bộ lệnh chạy trong **Ubuntu/WSL Bash**, không chạy trong PowerShell.
Mở Docker Desktop như hôm qua. Lấy code mới từ `main` bằng một clone riêng,
giữ nguyên repo và kết quả hôm qua:

```bash
export TODAY=/mnt/d/ZAIC/lifemate50-9-20261115
export PREVIOUS=/home/kien/lifemate-handoff
mkdir -p "$TODAY/incoming"
git clone --branch main --single-branch https://github.com/duongtanminh1912/zaic1.git "$TODAY/source"
cd "$TODAY/source"
git rev-parse HEAD
python3 --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
sudo docker version
df -h "$TODAY" /
sudo docker run --rm --gpus all nvidia/cuda:13.2.0-runtime-ubuntu24.04 nvidia-smi
```

Nếu đã nhận `lifemate50-9-source.tar.gz`, có thể giải nén vào `$TODAY/source`
thay cho `git clone`, rồi `cd` vào đó. Chỉ chọn một cách; source archive không
chứa weights/dataset và không phải image Docker để nộp.

Python host >=3.11; driver/Docker phải nhận GPU. Đo dung lượng thực tế trước
build: assets mới khoảng5,8GB, còn context/image/export và Docker VHDX trên C.
Không xóa/prune bản đã nộp hoặc containers của dự án khác để lấy chỗ.

## 2. Nhận đúng model ZIP 50,9

Từ thư mục Drive model mới đã bàn giao, tải vào `$TODAY/incoming/`:

- `LIFEMATE_BASELINE_50_9_MODELS_AND_PRIVATE_DATA.zip`
- `SHA256SUMS.json`

Không cần tải runtime Windows ZIP. Không dùng file model50,2 hoặc đổi tên ZIP50,2.
Giữ đúng tên ZIP gốc vì checksum tra theo tên. Chạy:

```bash
python3 llm/baseline_50_9_verified/tools/unpack_assets.py \
  --archive "$TODAY/incoming/LIFEMATE_BASELINE_50_9_MODELS_AND_PRIVATE_DATA.zip" \
  --checksums "$TODAY/incoming/SHA256SUMS.json" \
  --dest "$TODAY/private_assets"
PYTHONPATH=llm/baseline_50_9_verified python3 -c \
  'from baseline50.release import verify; verify(); print("SOURCE_50_9_VERIFIED")'
```

Phải có `UNPACKED_AND_VERIFIED` và `SOURCE_50_9_VERIFIED`. Bộ source bảo toàn
đúng byte LF/CRLF được checksum; không chạy dos2unix toàn bộ package Python.
Riêng shell scripts Docker được chuẩn hóa LF tự động khi tạo context.

## 3. Runtime Linux và 5 mẫu smoke mới

Có thể dùng lại runtime đã stage hôm qua; prepare_context sẽ kiểm hash từng file:

```bash
export RUNTIME="$PREVIOUS/linux_runtime"
```

Nếu thư mục đó không còn, tải/stage lại đúng runtime:

```bash
curl -fL --retry 2 \
  https://github.com/ggml-org/llama.cpp/releases/download/b11800/llama-b11800-bin-ubuntu-cuda-12.8-x64.tar.gz \
  -o "$TODAY/incoming/llama-linux.tar.gz"
python3 ops/packaging/baseline_50_9/stage_linux_runtime.py \
  "$TODAY/incoming/llama-linux.tar.gz" "$TODAY/linux_runtime"
export RUNTIME="$TODAY/linux_runtime"
```

Tái sử dụng dataset BTC hôm qua. Nếu đã di chuyển, chỉ sửa `--dataset`:

```bash
python3 ops/packaging/baseline_50_9/prepare_smoke.py \
  --dataset "$PREVIOUS/train+public_test" \
  --assets "$TODAY/private_assets" --destination "$TODAY/smoke"
```

Phải báo `samples: 5`. Audio count tùy 5 ID FIT được chọn hôm nay, không mặc định7.
Helper kiểm hash TRAIN JSONL/catalog/SDK, split/index50,9 và audio; không đọc
labels vào input. Chép state nguyên byte cho SDK, không phân tích state.
Không dùng smoke_data cũ nếu IDs không thuộc FIT của50,9.

## 4. Training handoff và build image mới

Bộ private model không chứa full training reproduction bundle. Để thử kỹ thuật,
đưa source tham khảo50,9 vào candidate, giữ README gốc ghi rõ giới hạn:

```bash
mkdir "$TODAY/training_handoff"
cp -R llm/baseline_50_9_verified/training_reference/. "$TODAY/training_handoff/"
cp llm/baseline_50_9_verified/training_recipe.json "$TODAY/training_handoff/"
python3 ops/packaging/baseline_50_9/prepare_context.py \
  --assets "$TODAY/private_assets" --runtime "$RUNTIME" \
  --training "$TODAY/training_handoff" --destination "$TODAY/docker_context"
set -o pipefail
sudo docker build --platform linux/amd64 -t zaic-team:50-9-candidate \
  "$TODAY/docker_context" 2>&1 | tee "$TODAY/build.log"
sudo docker run --rm --gpus all --network none zaic-team:50-9-candidate \
  python3 -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__); print(torch.cuda.get_device_name(0))'
sudo docker run --rm --network none zaic-team:50-9-candidate python3 -m pip check
```

Đã tích hợp hai lỗi hôm qua: scripts LF và manifest tại đúng đường dẫn50,9.
Môi trường CUDA13.2, torch2.8.0+cu128, transformers4.57.1, llama.cpp b11800.
Model Large dùng **serial**: ASR lên GPU → ASR về CPU → nạp Qwen → giải phóng
Qwen mỗi mẫu, khác profile resident của Docker50,2. Không thay weights/context/
batch/GPU layers để ép vừa rồi vẫn gọi đó là cùng cấu hình. Port này chưa warmup
TRAIN; `warmup=false` được ghi diagnostics, cần đánh giá timing thực tế.

Training reproduction chưa hoàn thiện phải theo dõi riêng. Test kỹ thuật thành
công không chứng nhận yêu cầu Training Code/Data của bài nộp cuối đã đầy đủ.

## 5. Predict offline trên 4060

```bash
mkdir "$TODAY/result"
sudo docker run --rm --gpus '"device=0"' --network none \
  -v "$TODAY/smoke/data:/data:ro" -v "$TODAY/result:/result" \
  zaic-team:50-9-candidate /bin/bash /code/predict.sh \
  2>&1 | tee "$TODAY/predict.log"
```

Kiểm đủ5 output và diagnostics. Phải thực sự chạy ASR/LLM, không có runtime_error,
backend_error, execution_error hoặc OOM. Timeout phải được báo, không xóa mẫu.
Một lượt exit0 hoặc JSONL đủ dòng chưa chứng minh model chạy được.
Deadline30s gồm audio, chuyển ASR GPU/CPU, startup/shutdown Qwen, tools và session
output. Không lấy số đo50,2 hôm qua làm số đo50,9 hoặc suy ra tốc độ RTX5090.

Có thể ghi VRAM trong terminal khác bằng:

```bash
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
  --format=csv -l 1 > "$TODAY/gpu-monitor.csv"
```

Dừng monitor bằng Ctrl+C sau lượt chạy. Đây là bộ nhớ toàn GPU, có thể gồm app
khác. Nếu lỗi, giữ logs/result và xử lý nguyên nhân trước lượt mới; dùng output
folder mới, không retry riêng mẫu để chọn kết quả.

## 6. Notebook và đối chiếu

```bash
sudo docker run --rm --gpus '"device=0"' -p 127.0.0.1:9777:9777 \
  -v "$TODAY/smoke/data:/data:ro" -v "$TODAY/result:/result" \
  zaic-team:50-9-candidate /bin/bash /code/start_jupyter.sh
```

Mở http://localhost:9777, token `zac2026`, mở `/code/predict_notebook.ipynb`,
Restart Kernel rồi Run All. Notebook phải tạo `jupyter_submission.jsonl` và
 diagnostics riêng, không ghi đè submission CLI. Sau khi chạy xong:

```bash
python3 ops/packaging/baseline_50_9/check_results.py \
  --result "$TODAY/result" --expected "$TODAY/smoke/prepared/expected_ids.jsonl"
nvidia-smi > "$TODAY/gpu.txt"
sudo docker image inspect zaic-team:50-9-candidate > "$TODAY/image.json"
sudo docker run --rm --network none zaic-team:50-9-candidate \
  python3 -m pip freeze > "$TODAY/pip-freeze.txt"
```

Checker yêu cầu đúng5 ID độc lập, không lỗi kỹ thuật/timeout và calls/hash CLI
khớp notebook. Pass chỉ là technical smoke, không accuracy/official score.
Nếu quá30s trên4060, giữ log để phân tích; không tăng budget hoặc tự nhận5090
sẽ đạt. Có thể xem xét Test Solution sau đánh giá riêng, không tự coi smoke pass.

## 7. Export archive mới + MD5

Chỉ thực hiện sau khi kiểm kết quả model mới. Dùng file/tag riêng, không ghi đè
bản hôm qua. Ví dụ export đầu tiên:

```bash
sudo docker tag zaic-team:50-9-candidate zaic-team:rtx4060-50-9-test1
mkdir "$TODAY/export-test1"
set -o pipefail
sudo docker save zaic-team:rtx4060-50-9-test1 | gzip > "$TODAY/export-test1/zaic_team_50_9.tar.gz"
md5sum "$TODAY/export-test1/zaic_team_50_9.tar.gz" \
  > "$TODAY/export-test1/zaic_team_50_9.tar.gz.md5"
sudo docker load -i "$TODAY/export-test1/zaic_team_50_9.tar.gz"
```

Sau load, chạy offline lại với output folder mới; nếu có máy khác thì load và
chạy trên máy đó theo guide. Không lấy MD5/archive/imageID hôm qua cho bản hôm nay.
Nếu sửa image và export lại, đặt tên phiên mới, tính lại MD5.

## 8. Drive và portal

Upload **zaic_team_50_9.tar.gz**, bật Anyone with the link, gửi link FILE và
MD5 của chính file gzip vào Test Solution theo thông báo BTC hiện hành. Không
đưa link model ZIP, link folder hoặc checksum source thay archive Docker.
Không thay file đang được BTC tải/chạy. Người phụ trách tự thao tác portal;
bộ hướng dẫn không tự dùng quota hoặc upload tài nguyên private.

Gửi lại nhóm: build.log, predict.log, gpu.txt, image.json, pip-freeze.txt,
VRAM monitor, notebook executed, hai submission và diagnostics/server logs;
archive/link/MD5 mới, trạng thái portal nếu đã gửi. Không push dữ liệu/log thô.

Nguồn: Docker Guide BTC, mục4–8:
https://dl-challenge.zalo.ai/2026/lifemate_argfzdgert/Docker_Guide_ZAC2026_LifeMate.pdf
Và code `llm/baseline_50_9_verified/DOCKER_HANDOFF.md`.
