# Pipeline build map (per-frame intrinsics + depth scale, không dùng pose cảm biến)

Pipeline này biến session raw (frames + depth + sensors.txt) thành map pose metric cho localization. Toàn bộ đều chạy bằng CLI, không cần COLMAP GUI trừ khi muốn xem nhanh.

## Yêu cầu
- Python >= 3.9, các package trong `pyproject.toml` đã có `numpy`, `pycolmap`, `pyyaml`; cần thêm `Pillow` nếu chưa cài (`pip install Pillow`).
- COLMAP CLI sẵn sàng trong PATH (hoặc chỉ định bằng `--colmap-bin`).
- Dataset layout (ví dụ `capture/UET_G2/raw/phone/2025-12-22_16.43.15`):
  - `frames/*.jpg`
  - `depth/*.bin`
  - `depth/*.confidence.png` (mask, tùy chọn)
  - `sensors.txt` (timestamp, pose, intrinsics)

## Chạy nhanh (ví dụ cho UET_G2 phone)
```bash
SESSION=capture/UET_G2/raw/phone/2025-12-22_16.43.15
KF_OUT=build_map/output_keyframes
SFM_OUT=build_map/output_sfm
SFM_SCALED=build_map/output_sfm_scaled
POSES=build_map/poses.txt

# 1) Chọn keyframe + copy/symlink dữ liệu cần thiết
python build_map/pipeline.py select-keyframes \
  --session-root $SESSION \
  --output-root $KF_OUT \
  --stride 5

# 2) Tạo database COLMAP (mỗi frame = 1 camera)
python build_map/pipeline.py create-db \
  --keyframes-root $KF_OUT \
  --database $KF_OUT/colmap.db

# 3) Feature + sequential matching + mapper
python build_map/pipeline.py run-sfm \
  --database $KF_OUT/colmap.db \
  --image-path $KF_OUT/images \
  --output-model $SFM_OUT \
  --colmap-bin colmap

# 4) Ước lượng scale từ depth, ghi model mới
python build_map/pipeline.py scale-from-depth \
  --model-path $SFM_OUT/0 \
  --intrinsics $KF_OUT/intrinsics.json \
  --depth-dir $KF_OUT/depth \
  --output-model $SFM_SCALED

# 5) Xuất poses (camera-to-world, quaternion qx qy qz qw)
python build_map/pipeline.py export-poses \
  --model-path $SFM_SCALED \
  --output $POSES
```

## Giải thích các bước
- `select-keyframes`: đọc `sensors.txt` chỉ để lấy intrinsics; chọn keyframe đơn giản theo stride. Xuất `images/`, `depth/`, `intrinsics.json`. Dùng `--copy-mode symlink|copy|hardlink` tùy ý.
- `create-db`: tạo `colmap.db` với model PINHOLE, mỗi frame một camera, intrinsics cố định.
- `run-sfm`: gọi COLMAP CLI (SIFT) với sequential matcher (overlap=4). BA khóa intrinsics (`ba_refine_* = 0`). Model nằm trong thư mục `output-model/0`.
- `scale-from-depth`: load model, đọc depth theo timestamp, lấy các 3D points đã đăng ký trong từng ảnh, so sánh depth dự đoán (SfM) với depth đo được, tính tỷ lệ median → scale toàn bộ tvec + point cloud rồi ghi model mới.
- `export-poses`: xuất pose camera-to-world (tx ty tz, quaternion qx qy qz qw).

## Lưu ý
- Depth `.bin` được autodetect dtype: thử float32, float16, rồi uint16 (chia 1000.0). Nếu định dạng khác sẽ báo lỗi.
- Confidence mask: dùng `*.confidence.png` nếu có, ngưỡng mặc định 64 (`--conf-threshold`).
- Nếu máy không có GPU, thêm `--cpu` ở bước `run-sfm`.
- Nếu muốn SuperPoint thay SIFT, chạy feature/matching bằng tool riêng rồi mapper; script này giữ SIFT để tương thích COLMAP mặc định.
- Để kiểm tra scale, xem thống kê p5/p95 được in ra ở bước `scale-from-depth`. Nếu số mẫu < `min-samples` thì cần tăng overlap hoặc giảm lọc.

## Đầu ra
- Model gốc: `$SFM_OUT/0` (up-to-scale).
- Model đã scale: `$SFM_SCALED`.
- Poses: `$POSES` với định dạng `image_name tx ty tz qx qy qz qw` (camera-to-world, mét).
