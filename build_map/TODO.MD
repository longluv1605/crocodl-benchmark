OK, mình sẽ **trình bày lại TOÀN BỘ giải pháp một cách đầy đủ, mạch lạc, và “đóng đinh” cho đúng dataset của bạn**, không giả định mơ hồ nữa.
Bạn có thể coi đây là **spec + pipeline chính thức để build map poses** cho hệ thống visual localization của bạn.

---

# MỤC TIÊU CUỐI

* Sinh **camera poses 6DoF (qx, qy, qz, qw, tx, ty, tz)**
* **Offline**
* **Metric (đúng scale)**
* **Tất cả poses nằm trong cùng 1 world frame**

  * Frame đầu tiên = gốc O
* Dùng để **build map → query localization sau này**

---

# DỮ LIỆU ĐẦU VÀO (CHUẨN HÓA)

Bạn có raw data (/home/long/Workspace/crocodl-benchmark/capture/UET_G2/raw/phone/2025-12-22_16.43.15):

```
frames/            RGB images (timestamped)
depth/*.bin        Depth maps (same timestamp as frames)
depth/*.confidence.png
sensors.txt        Camera intrinsics per timestamp:
                   w, h, fx, fy, cx, cy, exposure
accelerometer.txt  (không dùng cho mapping)
gyroscope.txt      (không dùng cho mapping)
fused_imu.txt      (không dùng cho mapping)
location.txt       (không dùng cho mapping)
```

### Các giả định ĐÚNG:

* RGB ↔ Depth ↔ Intrinsics **đã sync theo timestamp**
* Depth **pixel-aligned** với RGB
* Depth **đơn vị meter**

---

# TỔNG QUAN PIPELINE

```
RAW DATA
   ↓
Keyframe Selection
   ↓
COLMAP SfM (per-frame intrinsics)
   ↓
Depth-based Scale Fixing
   ↓
Pose Refinement (PnP + BA)
   ↓
Export Map Poses
```

---

# STEP 0 — PARSE & CHUẨN HÓA DỮ LIỆU

## 0.1 Parse `sensors.txt`

Mỗi dòng:

```
timestamp w h fx fy cx cy exposure
```

Tạo map:

```
timestamp → K_i
```

## 0.2 Match dữ liệu theo timestamp

Với mỗi frame `T`:

* `frames/T.jpg`
* `depth/T.bin`
* `depth/T.confidence.png`
* `K_T` từ sensors.txt

---

# STEP 1 — KEYFRAME SELECTION (BẮT BUỘC)

### Lý do

* 10 fps → quá dày
* SfM cần **baseline + parallax**, không cần fps cao

### 1.1 Downsample thô

* Giữ 1 frame mỗi **5 frame**
  → ~2 fps (~700–800 frames)

### 1.2 Lọc tinh (khuyên rất mạnh)

Giữ frame `i` nếu:

* Dịch chuyển > **0.2–0.3 m** (ước lượng bằng depth)
* HOẶC xoay > **7–10°**
* HOẶC overlap feature < **75%**

### Output STEP 1:

```
keyframes/
  ├── images/
  ├── depth/
  └── intrinsics.json   (per frame)
```

🎯 Mục tiêu:

* **500–1000 keyframes**
* Phân bố đều không gian

---

# STEP 2 — CHUẨN BỊ COLMAP (PER-FRAME INTRINSICS)

## 2.1 Camera model

* Model: **PINHOLE**
* **Mỗi frame = 1 camera riêng**
* Intrinsics **FIXED**, không optimize

Lý do:

* Intrinsics thay đổi theo timestamp
* Chia camera giúp giảm reprojection error

## 2.2 Import vào COLMAP database

Trong database:

```
cameras:
  camera_id_i → (fx_i, fy_i, cx_i, cy_i)

images:
  image_i → camera_id_i
```

⚠️ KHÔNG dùng 1 camera chung

---

# STEP 3 — STRUCTURE FROM MOTION (RGB ONLY)

### Cấu hình

* Feature: SuperPoint
* Matching: **Sequential**
* Reconstruction: Incremental SfM
* Optimize:

  * Pose + structure
  * ❌ Không optimize intrinsics

### Kết quả:

* Pose mỗi frame:

  * Chung **1 world frame**
  * Frame đầu tiên = gốc
* Nhưng:

  * Translation **chưa metric** (scale chưa đúng)

---

# STEP 4 — DEPTH PROCESSING

## 4.1 Decode depth `.bin`

* Convert → depth map (meters)

## 4.2 Mask depth kém

* Dùng `confidence.png`
* Bỏ pixel depth không tin cậy

---

# STEP 5 — FIX SCALE BẰNG DEPTH (THEN CHỐT)

### 5.1 Back-project depth

Với mỗi frame `i`:

$X^i_c = K_i^{-1} [u, v, 1]^T \cdot Z$

→ Point cloud trong camera frame

### 5.2 Align SfM ↔ Depth

* SfM points: up-to-scale
* Depth points: metric

Tìm **scale s** sao cho:

$s \cdot t_{sfm} \approx t_{depth}$

Cách làm:

* Median ratio
* Least squares
* Umeyama (scale-only)

### 5.3 Apply scale

```
t_i_metric = s * t_i_sfm
```

📌 Rotation giữ nguyên

---

# STEP 6 — POSE REFINEMENT (RẤT KHUYÊN DÙNG)

Với mỗi frame:

* 3D points từ depth
* 2D keypoints từ RGB

→ Solve **PnP (fixed K_i)**
→ Optional local BA

### Kết quả:

* Pose metric
* Noise thấp
* Chuẩn cho localization

---

# STEP 7 — EXPORT MAP POSES

Format khuyên dùng:

```
frame_id timestamp
tx ty tz
qx qy qz qw
```

📌 World frame:

* Frame đầu tiên trong SfM
* Không thay đổi

---

# SENSOR KHÁC (IMU, GPS) — KHÔNG TRỘN VÀO

| Data       | Dùng cho mapping? | Ghi chú             |
| ---------- | ----------------- | ------------------- |
| Intrinsics | ✅ BẮT BUỘC        | Per-frame           |
| Depth      | ✅ BẮT BUỘC        | Metric              |
| IMU        | ❌                 | Chỉ debug           |
| GPS        | ❌                 | Align multi-session |

---

# KẾT LUẬN CUỐI (CHỐT)

> **Giải pháp ĐÚNG cho dataset của bạn là:
> SfM (COLMAP, per-frame intrinsics) + Depth scale fixing + Pose refinement.**

✔ Offline
✔ Pose chính xác
✔ Metric
✔ Chung world frame
✔ Dùng trực tiếp cho visual localization

---

<!-- ## BƯỚC TIẾP THEO NÊN LÀM NGAY

Mình đề xuất đi theo thứ tự này:

1️⃣ **Viết script import intrinsics per-frame vào COLMAP**
2️⃣ **Chạy SfM và kiểm tra trajectory**
3️⃣ **Code fix scale bằng depth**

👉 Bạn muốn mình bắt đầu **viết code cho bước nào**? -->
