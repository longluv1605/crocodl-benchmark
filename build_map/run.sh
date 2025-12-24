SESSION=capture/UET_G2/raw/phone/2025-12-22_16.43.15
KF_OUT=build_map/output_keyframes
SFM_OUT=build_map/output_sfm
SFM_SCALED=build_map/output_sfm_scaled
POSES=build_map/poses.txt

# 1) Chọn keyframe + copy/symlink dữ liệu cần thiết
python build_map/pipeline.py select-keyframes \
  --session-root $SESSION \
  --output-root $KF_OUT \
  --stride 2

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
python3 build_map/pipeline.py export-poses \
  --model-path $SFM_SCALED \
  --output $POSES