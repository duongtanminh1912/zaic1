# Hoàn thiện bài nộp dự phòng 50,2 từ image cũ

Dùng khi đội chọn Medium V3 + Qwen V2 đã chạy trước đó. Không đổi lựa chọn
model52 hiện tại của đội. Các lệnh chạy trong Ubuntu/WSL Bash trên máy Kiên,
Python>=3.11 và Docker đã nhận GPU. Nếu Docker cần sudo, thay các lệnh docker
bằng sudo docker; riêng build_submission.py chạy bằng sudo python3 khi cần.

## 1. Lấy code và giữ image gốc

Clone main vào thư mục mới hoặc cập nhật checkout sạch. Không xóa image/archive
hoặc kết quả cũ. Đặt ROOT ngoài repo và SMOKE vào đúng smoke_data của lượt50,2
hôm trước (split TRAIN-fit, có SDK/audio). Không dùng smoke52 thay thế.

```bash
export ROOT=/mnt/d/ZAIC/lifemate50-2-final-training-v1
export SMOKE=/home/kien/lifemate-handoff/smoke_data
mkdir -p "$ROOT"
git clone --branch main --single-branch https://github.com/duongtanminh1912/zaic1.git "$ROOT/source"
cd "$ROOT/source"
git rev-parse HEAD
docker image inspect zaic-team:rtx4060-test1-fix2 --format '{{.Id}}'
```

Nếu image không còn local, tải archive50,2 đã bàn giao hôm trước, kiểm MD5
`bbcad81bdf9d0a53ecaf10110adbc4fc` rồi `docker load -i PATH/zaic_team.tar.gz`.
Không load ZIP model/training bằng docker load. Builder kiểm đúng config/index
ID của image đã audit; image khác sẽ bị từ chối trước build.

## 2. Cung cấp hai URL training cho BTC

Nhóm đã có hai file, không cần train lại hoặc tải lại weights inference:

- `LIFEMATE_BASELINE_50_2_TRAINING_CHECKPOINTS.zip` trong bộ baseline50,2 cũ;
  743.551.608bytes, SHA256 bắt đầu2ef1dab2.
- `03_TRAINING_BASE50_9.zip` trong bộ52 mới (cũng được phát hành tên
  `LIFEMATE_50_9_TRAINING_PRIVATE.zip`);2.775.709.761bytes, SHA256 bắt đầu33373adf.

Lấy link **file**, bảo đảm BTC tải được đúng byte sau nộp và giữ nguyên file.
Không dùng link folder hay link model inference. Script ghi URL ngoài Git vào
image, kèm SHA256 đầy đủ trong ARCHIVES.json. Hai ASR clip manifests/split và
Qwen data/adapter đã đối chiếu khớp50,2; adapter Large trong ZIP chung không
được dùng cho bản này. Không bắt buộc giải nén training trên máy inference.

```bash
read -r -p 'Link file checkpoint 50,2 cho BTC: ' CHECKPOINTS_URL
read -r -p 'Link file training chung ZIP03 cho BTC: ' SHARED_TRAINING_URL
python3 -B ops/packaging/baseline_50_2_final/prepare_context.py \
  --checkpoints-url "$CHECKPOINTS_URL" \
  --shared-training-url "$SHARED_TRAINING_URL" \
  --destination "$ROOT/context"
```

Script tạo context code gọn: training50,2, các helper/Qwen V2 cố định, README
phương pháp và DATA_ACCESS. Dữ liệu/adapter vẫn ở URL riêng có checksum.

## 3. Build lớp bổ sung

```bash
set -o pipefail
python3 -B ops/packaging/baseline_50_2_final/build_submission.py \
  --context "$ROOT/context" --tag zaic-team:50-2-final-training-v1 \
  2>&1 | tee "$ROOT/build.log"
docker run --rm --network none zaic-team:50-2-final-training-v1 \
  python3 -B /code/submission_audit/audit_base.py
docker run --rm --network none zaic-team:50-2-final-training-v1 \
  cat /code/training_code/DATA_ACCESS.json
docker run --rm --network none zaic-team:50-2-final-training-v1 python3 -m pip check
```

Docker kế thừa image cũ, chỉ COPY training code/hồ sơ và README. Không cài lại
Torch, tải lại model hoặc thay source inference. Build kiểm hash128file kế
thừa, gồm model assets, Linux runtime, inference code/notebook/scripts.
Không dùng tag cũ cho image bổ sung. Mọi lỗi kiểm hash phải được điều tra,
không bỏ guard để tiếp tục.

## 4. Kiểm lại CLI và notebook trên 4060

Dùng output mới, giữ nguyên đầu ra cũ. Chép expected IDs trực tiếp từ manifest
smoke trước inference, không lấy IDs từ output để che thiếu mẫu:

```bash
python3 - "$SMOKE" "$ROOT/expected_ids.jsonl" <<'PY'
import json, pathlib, sys
root, dest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
paths = list(root.rglob('train.jsonl'))
assert len(paths) == 1, 'Set SMOKE to the old 50.2 TRAIN smoke mount'
rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
assert len(rows) == 5 and len({r['id'] for r in rows}) == 5
with dest.open('x') as out:
    for row in rows:
        out.write(json.dumps({'id': row['id']}) + '\n')
PY
mkdir "$ROOT/result"
docker run --rm --gpus '"device=0"' --network none \
  -v "$SMOKE:/data:ro" -v "$ROOT/result:/result" \
  zaic-team:50-2-final-training-v1 /bin/bash /code/predict.sh \
  2>&1 | tee "$ROOT/predict.log"
docker run --rm --gpus '"device=0"' -p 127.0.0.1:9777:9777 \
  -v "$SMOKE:/data:ro" -v "$ROOT/result:/result" \
  zaic-team:50-2-final-training-v1 /bin/bash /code/start_jupyter.sh
```

Mở localhost:9777, tokenzac2026, kernel mới, Run All predict_notebook.ipynb.
Dừng container notebook bằng Ctrl+C sau khi xong. Kiểm bằng checker chung
(file nằm trong folder50,9 nhưng không phụ thuộc model/profile):

```bash
python3 -B ops/packaging/baseline_50_9/check_results.py \
  --result "$ROOT/result" --expected "$ROOT/expected_ids.jsonl"
docker image inspect zaic-team:50-2-final-training-v1 > "$ROOT/image.json"
nvidia-smi > "$ROOT/gpu.txt"
```

Checker cần đủ5ID, hai output khớp calls/hash, thời gian<=30.000ms và diagnostics
không có failure/error. Exit0/đủ5dòng một mình chưa chứng minh đạt. Giữ lỗi và
logs để sửa nếu có; không sửa tay hay chọn output của từng lần chạy. Smoke
không phải bảo đảm toàn bộ hidden split/RTX5090 đạt. Chưa có lượt GPU cho image
bổ sung trong đợt làm code trên máy Mac này.

## 5. Export, kiểm load và nộp chính thức

Chỉ export đúng tag đã kiểm. Bật pipefail để docker save lỗi không tạo archive
trông có vẻ thành công. Dùng thư mục mới, không đè archive cũ:

```bash
set -euo pipefail
mkdir "$ROOT/submission"
docker save zaic-team:50-2-final-training-v1 | gzip > "$ROOT/submission/zaic_team.tar.gz"
gzip -t "$ROOT/submission/zaic_team.tar.gz"
md5sum "$ROOT/submission/zaic_team.tar.gz" > "$ROOT/submission/zaic_team.tar.gz.md5"
docker load -i "$ROOT/submission/zaic_team.tar.gz"
cat "$ROOT/submission/zaic_team.tar.gz.md5"
```

Output nộp: **zaic_team.tar.gz + MD5 mới + link tải đúng file archive**. Upload
Drive và cho phép BTC tải bằng link, kiểm đúng archive/version trước điền vào
mục nộp solution chính thức. Training URLs nằm trong image cũng phải tải được.
Đội tự thực hiện thao tác Submit. Không dùng MD5 cũ, index digest/imageID hoặc
SHA256 model thay MD5 archive. Nếu đổi image sau kiểm, export/MD5 phải làm lại.

README phương pháp và bộ training đã được bổ sung từ tài liệu/source thực tế;
VALIDATION.json ghi rõ CPU/data audit đã chạy và full retrain/merge/GPU chưa
chạy. Không coi hướng dẫn này là chứng nhận BTC chấp nhận bài nộp.
