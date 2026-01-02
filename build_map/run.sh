SESSION=capture/UET_G2/raw/phone/2025-12-22_16.43.15
OUT_DIR=build_map/outputs
RUN_DIR=$OUT_DIR/run_no_rescale_normalized
KF_OUT=$OUT_DIR/output_keyframes_s10
SFM_OUT=$RUN_DIR/output_sfm
SFM_SCALED=$RUN_DIR/output_sfm_scaled
SFM_NORMALIZED=$RUN_DIR/output_sfm_normalized
POSES=$RUN_DIR/poses.txt

# 1) Chọn keyframe + copy/symlink dữ liệu cần thiết
python build_map/pipeline.py select-keyframes \
  --session-root $SESSION \
  --output-root $KF_OUT \
  --stride 10 \
  --copy-mode copy

# # 2) Tạo database COLMAP (mỗi frame = 1 camera)
# python build_map/pipeline.py create-db \
#   --keyframes-root $KF_OUT \
#   --database $KF_OUT/colmap.db

# # 3) Feature + sequential matching + mapper
# python build_map/pipeline.py run-sfm \
#   --database $KF_OUT/colmap.db \
#   --image-path $KF_OUT/images \
#   --output-model $SFM_OUT \
#   --colmap-bin colmap

# # 4) Ước lượng scale từ depth, ghi model mới
# python build_map/pipeline.py scale-from-depth \
#   --model-path $SFM_OUT/0 \
#   --intrinsics $KF_OUT/intrinsics.json \
#   --depth-dir $KF_OUT/depth \
#   --output-model $SFM_SCALED

# # 5) Chuẩn hóa hệ trục tọa độ: gốc tại camera đầu tiên, X=phải, Y=trên, Z=sau
# python build_map/pipeline.py normalize-coordinate-system \
#   --model-path $SFM_SCALED \
#   --output-model $SFM_NORMALIZED

# # 6) Xuất poses (camera-to-world, quaternion qx qy qz qw)
# python3 build_map/pipeline.py export-poses \
#   --model-path $SFM_NORMALIZED \
#   --output $POSES
