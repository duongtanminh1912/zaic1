# Training Code/Data — Medium V3 + Qwen V2

Entrypoint được bàn giao: `baseline_50_2_training_repro/reproduce.py`.
Đọc `baseline_50_2_training_repro/README.md` và `DATA_ACCESS.json` tại đây.

- Tải đúng hai ZIP trong DATA_ACCESS, kiểm byte count/SHA256, giải nén bằng
  lệnh `unpack` của entrypoint vào hai thư mục mới.
- Bộ shared mang tên50,9 nhưng raw TRAIN/split, ASR clips và Qwen V2 được
  kiểm trùng với hồ sơ50,2. Wrapper Medium không train hoặc merge adapter Large.
- `baseline_50_9_training_repro/` là dependency cố định cung cấp helper
  prepare/SFT, Qwen V2 và converter orchestration; luôn giữ cạnh package50,2.
- Base revision, môi trường, hyperparameters, seed và lệnh prepare/train/
  merge/export nằm trong README cùng metadata của package50,2.
- Training chạy trong môi trường riêng có quyền tải base/data. Image inference
  có đủ weights để chạy offline, không dùng các URL này khi predict.

Những file historical training_reference cũ còn tại root chỉ để đối chiếu;
chạy entrypoint mới ở trên để tránh dependency theo đường dẫn máy cũ.
