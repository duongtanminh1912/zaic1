# RTX 4060 → bài nộp chính thức Docker — model 52

Dùng lại máy Kiên đã chạy thành công ngày 14/11: Ubuntu-24.04/WSL2,
RTX4060 Laptop 8GB, Docker Desktop. Đây là bộ **code/hướng dẫn chuẩn bị**,
chưa phải báo cáo build hoặc chạy GPU thành công hôm nay.

Model mới chốt: trained PhoWhisper-large + Qwen3.5-4B-LifeMate-decision-dpo-Q4_K_M,
helper ngày/giờ V1, retrieval TRAIN-fit. Không dùng private_assets của Medium50,2.
Giữ image `zaic-team:rtx4060-test1-fix2`, archive và kết quả hôm qua.

## 1. Cài bộ code mới và đặt thư mục làm việc

Toàn bộ lệnh chạy trong **Ubuntu/WSL Bash**, không chạy trong PowerShell.
Mở Docker Desktop như hôm qua. Lấy code mới từ `main` bằng một clone riêng,
giữ nguyên repo và kết quả hôm qua:

```bash
export TODAY=/mnt/d/ZAIC/lifemate52-final-20261116
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

Nếu đã nhận `lifemate52-source.tar.gz`, có thể giải nén vào `$TODAY/source`
thay cho `git clone`, rồi `cd` vào đó. Chỉ chọn một cách; source archive không
chứa weights/dataset và không phải image Docker để nộp.

Python host >=3.12; driver/Docker phải nhận GPU. Đo dung lượng thực tế trước
build: assets mới khoảng5,8GB, còn context/image/export và Docker VHDX trên C.
Không xóa/prune bản đã nộp hoặc containers của dự án khác để lấy chỗ.

## 2. Dùng đúng Drive52 và ZIP inference mới

Từ thư mục Drive52 mới được bàn giao cho nhóm, tải `01_INFERENCE_MODELS_52.zip`
vào `$TODAY/incoming/`. Có thể tải SHA256SUMS.json để đối chiếu; công cụ bên
 dưới đã ghim outer SHA và từng member qua DRIVE_ARCHIVES.json trong code:

```bash
python3 -B llm/baseline_52_release/release_tools.py unpack \
  --archive "$TODAY/incoming/01_INFERENCE_MODELS_52.zip" \
  --dest "$TODAY/private_assets"
PYTHONPATH=llm/baseline_52_release/inference python3 -B -c \
  'from baseline50.release import verify; verify(); print("SOURCE_52_VERIFIED")'
```

Không cần ZIP02 runtime Windows trên WSL/Linux. ZIP03/04 là dữ liệu tái lập;
image ghi link tải hai file này, không cần chép thêm6GB training vào image.
ZIP05 chỉ là bằng chứng đánh giá/submission lịch sử, không dùng làm inference
input hoặc labels train. Không cần tải nó để build image.

Qwen deployed phải là `models/qwen/Qwen3.5-4B-LifeMate-decision-dpo-Q4_K_M.gguf`,
SHA256 `aee8106ebe960c26e905443fedb3e8bd2e3a6c1218e63806892f849dbb8e512e`.
Parent50,9 có trong ZIP04 chỉ phục vụ TRAIN harvest, không dùng cho inference52.
ASR giữ nguyên Large hai shard. Không sửa hash guards để nhận weights cũ.
Module Python vẫn tên baseline50 theo API lịch sử; source/manifest hiện là52.

Package giữ nguyên bytes source LF/CRLF bằng Git attributes. Riêng shell script
được chuẩn hóa LF khi tạo context. Source audit đầy đủ dùng Python>=3.12 vì
release_tools dùng Path.is_junction; host Ubuntu24.04/Python3.12 như máy Kiên.

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
python3 ops/packaging/baseline_52/stage_linux_runtime.py \
  "$TODAY/incoming/llama-linux.tar.gz" "$TODAY/linux_runtime"
export RUNTIME="$TODAY/linux_runtime"
```

Tái sử dụng dataset BTC hôm qua. Nếu đã di chuyển, chỉ sửa `--dataset`:

```bash
python3 ops/packaging/baseline_52/prepare_smoke.py \
  --dataset "$PREVIOUS/train+public_test" \
  --assets "$TODAY/private_assets" --destination "$TODAY/smoke"
```

Phải báo `samples: 5`. Audio count tùy 5 ID FIT được chọn hôm nay, không mặc định7.
Helper kiểm hash TRAIN JSONL/catalog/SDK, split/index52 và audio; không đọc
labels vào input. Chép state nguyên byte cho SDK, không phân tích state.
Không dùng smoke_data cũ nếu IDs không thuộc FIT của52.

## 4. Cả hai giai đoạn training và build image mới

Package `llm/baseline_52_release` chứa inference, training/base50_9 và training/dpo.
DPO chốt63 preference đã review,189replay rows,63updates; không phải draft64.
Training README có base pins/URLs, seed/environment, prepare/profile/train,
reference, merge/export. Đặt hai URL của đúng ZIP03 và ZIP04 (người BTC tải được)
vào image. Không dùng link folder, ZIP01 hoặc link file cũ sai checksum:

```bash
read -r -p 'Link HTTPS 03_TRAINING_BASE50_9.zip: ' BASE_TRAINING_URL
read -r -p 'Link HTTPS 04_TRAINING_DPO52.zip: ' DPO_TRAINING_URL
python3 ops/packaging/baseline_52/prepare_context.py \
  --assets "$TODAY/private_assets" --runtime "$RUNTIME" \
  --base-training-url "$BASE_TRAINING_URL" --dpo-training-url "$DPO_TRAINING_URL" \
  --destination "$TODAY/docker_context"
set -o pipefail
sudo docker build --platform linux/amd64 -t zaic-team:52-candidate \
  "$TODAY/docker_context" 2>&1 | tee "$TODAY/build.log"
sudo docker run --rm --gpus all --network none zaic-team:52-candidate \
  python3 -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__); print(torch.cuda.get_device_name(0))'
sudo docker run --rm --network none zaic-team:52-candidate python3 -m pip check
sudo docker run --rm --network none zaic-team:52-candidate \
  cat /code/training_code/DATA_ACCESS.json
```

Context chứa `/code/training_code/README.md`, DATA_ACCESS.json và toàn bộ package
52 nguyên vẹn; DATA_ACCESS ghi cả hai URL, SHA256 và dung lượng từ pins repo.
Giữ tài nguyên tại URL nguyên trạng sau deadline. Không commit URL private vào Git.

Dùng runtime Linux llama.cpp b11800 đã kiểm hash hôm qua. CUDA13.2,
torch2.8.0+cu128, transformers4.57.1 cho inference; training5.17.0 dùng môi trường
riêng. Không chạy .exe Windows hoặc cài training libraries vào venv inference.

Serial Large ASR→CPU→Qwen52 từng mẫu, batch2/context16384/output512 và deadline30s.
Không ngầm dùng resident hoặc thay model/context/GPU layers để ép vừa. Port này
chưa warmup TRAIN, diagnostics ghi warmup=false. Bản52 chưa được chạy Docker GPU
trong lần chuẩn bị code này; Windows public run/điểm báo cáo không chứng nhận
Linux image. Full retrain/reference/merge-export mới chưa được thực hiện.

## 5. Predict offline trên 4060

Nếu gặp startup timeouts như lượt WSL10GB/8CPU, giữ output cũ và dùng
[hướng dẫn đo timeout](RTX4060_TIMEOUT_52.md) trước khi điều chỉnh cấu hình.
CLI mới trả exit2 khi có technical failure/timeout; đủ JSONL không phải đạt smoke.

```bash
mkdir "$TODAY/result"
sudo docker run --rm --gpus '"device=0"' --network none \
  -v "$TODAY/smoke/data:/data:ro" -v "$TODAY/result:/result" \
  zaic-team:52-candidate /bin/bash /code/predict.sh \
  2>&1 | tee "$TODAY/predict.log"
```

Kiểm đủ5 output và diagnostics. Phải thực sự chạy ASR/LLM, không có runtime_error,
backend_error, execution_error hoặc OOM. Timeout phải được báo, không xóa mẫu.
Một lượt exit0 hoặc JSONL đủ dòng chưa chứng minh model chạy được.
Deadline30s gồm audio, chuyển ASR GPU/CPU, startup/shutdown Qwen, tools và session
output. Không lấy số đo50,2 hôm qua làm số đo52 hoặc suy ra tốc độ RTX5090.

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
  zaic-team:52-candidate /bin/bash /code/start_jupyter.sh
```

Mở http://localhost:9777, token `zac2026`, mở `/code/predict_notebook.ipynb`,
Restart Kernel rồi Run All. Notebook phải tạo `jupyter_submission.jsonl` và
 diagnostics riêng, không ghi đè submission CLI. Sau khi chạy xong:

```bash
python3 ops/packaging/baseline_52/check_results.py \
  --result "$TODAY/result" --expected "$TODAY/smoke/prepared/expected_ids.jsonl"
nvidia-smi > "$TODAY/gpu.txt"
sudo docker image inspect zaic-team:52-candidate > "$TODAY/image.json"
sudo docker run --rm --network none zaic-team:52-candidate \
  python3 -m pip freeze > "$TODAY/pip-freeze.txt"
```

Checker yêu cầu đúng5 ID độc lập, không lỗi kỹ thuật/timeout và calls/hash CLI
khớp notebook. Pass chỉ là technical smoke, không accuracy/official score.
Nếu quá30s trên4060, giữ log để phân tích; không tăng budget hoặc tự nhận5090
sẽ đạt. Cần đánh giá lỗi/timing trước khi chốt bài nộp; không tự coi smoke pass.

## 7. Export archive mới + MD5

Chỉ thực hiện sau khi kiểm kết quả model mới. Dùng file/tag riêng, không ghi đè
bản hôm qua. Ví dụ export đầu tiên:

```bash
sudo docker tag zaic-team:52-candidate zaic-team:rtx4060-52-final1
mkdir "$TODAY/export-final1"
set -o pipefail
sudo docker save zaic-team:rtx4060-52-final1 | gzip > "$TODAY/export-final1/zaic_team_52.tar.gz"
md5sum "$TODAY/export-final1/zaic_team_52.tar.gz" \
  > "$TODAY/export-final1/zaic_team_52.tar.gz.md5"
sudo docker load -i "$TODAY/export-final1/zaic_team_52.tar.gz"
```

Sau load, chạy offline lại với output folder mới; nếu có máy khác thì load và
chạy trên máy đó theo guide. Không lấy MD5/archive/imageID hôm qua cho bản hôm nay.
Nếu sửa image và export lại, đặt tên phiên mới, tính lại MD5.

## 8. Kiểm tra bài nộp cuối, Drive và portal

Sau khi load archive, dùng output mới để chạy lại predict (và trên máy khác theo
Docker guide). Giữ logs nếu lỗi. Kiểm README phương pháp, training_code/README,
DATA_ACCESS.json đúng URL BTC tải được, package reproduce đủ file, seed/config
và base model pins. GPU inference thành công không thay cho kiểm train lại.

Upload **zaic_team_52.tar.gz** lên Drive, bật Anyone with the link, nộp link
FILE và MD5 của chính archive theo thông báo nộp chính thức BTC. Không đưa link
model/training ZIP hay link folder thay Docker archive. Không đổi file đang
được BTC tải/chạy. Mọi sửa image sau export cần archive và MD5 mới.

Người phụ trách tự thao tác portal nộp chính thức; quy trình này không dùng mục
Test Solution. Kiểm thời hạn/hồ sơ kèm theo theo thông báo BTC đang áp dụng.

Gửi nhóm build.log, predict.log, gpu.txt, image.json, pip-freeze.txt, VRAM monitor,
notebook executed, hai submission, diagnostics/server logs, link archive và MD5.
Không push weights, data, URL private hoặc log thô lên Git.

Nguồn contract: Docker Guide BTC, mục4–8:
https://dl-challenge.zalo.ai/2026/lifemate_argfzdgert/Docker_Guide_ZAC2026_LifeMate.pdf
Và `llm/baseline_52_release/DOCKER_HANDOFF.md`,
`llm/baseline_52_release/TRAINING_REPRODUCTION.md`.
