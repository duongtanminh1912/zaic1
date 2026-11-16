# Docker52 — đo timeout trước khi điều chỉnh cấu hình

Ảnh/log nhóm báo: WSL10GB RAM/8CPU,5outputs nhưng4startup timeouts,1mẫu27,71s.
Chưa có raw logs trên workspace Ops để kết luận RAM/swap, I/O hay model loading
là nguyên nhân. Không xuất bản đó như một image đã qua smoke.

Bản diagnostics giữ weights52, ASR batch2, context16384, serial placement,
không warmup, deadline30s. Không tăng deadline hoặc tự đổi resident/offload.
Bổ sung đo wall-clock và Linux counters, có chi phí nhỏ nằm trong timer mẫu.

## 1. Giữ kết quả cũ, tạo lớp image chẩn đoán

Giữ `result_ram10_cpu8`, predict/server logs và imageID. Chỉ thực hiện một lượt
TRAIN-fit5mẫu để chẩn đoán; không rerun chọn riêng các mẫu đạt. Đặt TODAY theo
thư mục52 đang dùng, SOURCE là clone repo đang dùng:

```bash
export TODAY=/mnt/d/ZAIC/lifemate52-final-20261116
cd "$TODAY/source"
git switch main
git pull --ff-only
# Nếu image hiện tại dùng tag khác, thay đúng tag đã chạy52 ở BASE_IMAGE.
sudo docker build --platform linux/amd64 \
  -f ops/packaging/baseline_52/Dockerfile.diagnostics \
  --build-arg BASE_IMAGE=zaic-team:52-candidate \
  -t zaic-team:52-diagnostics1 ops/packaging/baseline_52
```

Chỉ copy predict.py/notebook mới lên image52 có sẵn, dùng lại weights/libraries,
không tải lại model hoặc thay code inference frozen. Không dùng image50,9 làm base.
Mọi image thay đổi là artifact mới; nếu cuối cùng chọn nộp phải export/MD5 mới.

## 2. Monitor trên Ubuntu/WSL

Tạo thư mục mới; không dùng lại result_ram10_cpu8:

```bash
mkdir "$TODAY/result_timing1"
free -h > "$TODAY/result_timing1/host-memory-before.txt"
sudo docker info > "$TODAY/result_timing1/docker-info.txt"
df -h > "$TODAY/result_timing1/disk.txt"
```

Terminal thứ hai (cùng biến TODAY), ghi RAM/swap/CPU/I/O trong lúc chạy:

```bash
vmstat 1 > "$TODAY/result_timing1/vmstat.log"
```

Terminal thứ ba ghi GPU:

```bash
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
  --format=csv -l 1 > "$TODAY/result_timing1/gpu.csv"
```

Dừng hai monitor bằng Ctrl+C sau lượt chạy. vmstat là số toàn WSL, GPU memory.used
là toàn GPU; không tự coi đó là memory của riêng model. Dòng vmstat đầu tiên là
trung bình từ lúc boot, không phải số tức thời của lượt inference.

## 3. Chạy đúng5TRAIN-fit đã chuẩn bị

Dùng đúng data mount smoke đã chạy5mẫu; nếu đặt ở chỗ khác, đổi path mount.
Giữ nguyên cấu hình10GB/8CPU trong lượt đo để có cơ sở so sánh, không vừa đo vừa
đổi quota. Container name và output đều phải mới:

```bash
set -o pipefail
sudo docker run --name lifemate52-timing1 --gpus '"device=0"' --network none \
  -v "$TODAY/smoke/data:/data:ro" -v "$TODAY/result_timing1:/result" \
  zaic-team:52-diagnostics1 /bin/bash /code/predict.sh \
  2>&1 | tee "$TODAY/result_timing1/predict.log"
run_exit=${PIPESTATUS[0]}
printf '%s\n' "$run_exit" > "$TODAY/result_timing1/exit-code.txt"
sudo docker inspect lifemate52-timing1 > "$TODAY/result_timing1/container.json"
free -h > "$TODAY/result_timing1/host-memory-after.txt"
```

Không dùng --rm để giữ metadata container. Không xóa output khi exit2.
CLI hiện trả **exit2 nếu có bất kỳ failure/timeout**, sau khi ghi đủ output và
summary. Exit0 yêu cầu không có lỗi kỹ thuật trong những mẫu thực sự chạy;
không chứng minh accuracy, full split hoặc thời gian máy BTC.

Notebook dùng cùng engine và báo lỗi sau ghi output nếu có technical failures;
chưa cần chạy notebook khi CLI vẫn timeout. Giữ kết quả để phân tích trước.

## 4. Đọc bằng chứng

- submission.jsonl: vẫn đủ5ID và authoritative tool_calls/state_hash/time.
- submission.jsonl.diagnostics.jsonl: mỗi mẫu có stage_ms, failure_phase,
  qwen_budget_ms và memory_before/before_qwen/after.
- submission.jsonl.summary.json: structure coverage, technical_failures/timeouts,
  technical_smoke_passed; không coi valid:true là technical pass.
- predict_server_N/server.log, predict_errors.log: startup/CUDA/model errors.
- vmstat.log, gpu.csv, container.json, docker-info.txt và exit-code.txt.

Stage milliseconds: input_prepare_ms riêng; asr_to_gpu_ms, asr_transcribe_ms,
asr_to_cpu_ms, qwen_startup_ms, agent_ms, qwen_shutdown_ms, asr_cleanup_ms,
session_export_ms. Chỉ stage đã chạy mới có trường. Các stage không bao trùm
mọi Python/memory-counter overhead; predict_ms là thời gian toàn mẫu.

qwen_budget_ms là budget còn lại **trước khởi động**, không cấp mới30s. Khi
startup thất bại, backend gốc tự cleanup trước khi ném exception; do đó
qwen_startup_ms gồm cả cleanup nội bộ đó, cờ startup_failure_includes_backend_cleanup
được ghi true. Không gọi số này là pure model load time khi server chưa healthy.

memory MemAvailable/SwapTotal/SwapFree tính kB; pswpin/pswpout là counters pages
cộng dồn của WSL; cgroup memory.current/max/swap.current là bytes, memory.events
là counters text. So chênh lệch counters/bằng chứng vmstat, không kết luận swap
chỉ vì một snapshot SwapFree thấp. Container snapshots có thể rỗng trên non-Linux.

## 5. Quyết định tiếp theo

Gửi toàn bộ result_timing1 cho Ops/LLM theo kênh riêng. Nếu budget trước Qwen
đã ít, tách ASR và transfers; nếu Qwen startup chiếm chính, đối chiếu server log,
RAM/swap, CPU và I/O. Chỉ điều chỉnh một yếu tố có bằng chứng, ghi tag/profile mới.
Resident/cắt context/đổi layers/batch/warmup là đổi execution profile, cần version
và kiểm hành vi/calls/hash/VRAM/timing; không tự coi tương đương serial gốc.

4060/WSL không phải RTX5090/BTC: số đo không chứng nhận pass/fail trên máy BTC.
Chỉ export bản cuối sau đánh giá output/timing và các mục nộp chính thức. Không
thay weights/prompt/SDK hoặc kéo deadline để biến timeout thành pass.
