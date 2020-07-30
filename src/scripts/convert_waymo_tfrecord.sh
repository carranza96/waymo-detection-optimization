python utils_tf_record/create_camera_waymo_tf_record.py \
          --set=training \
          --preprocessing=rgb \
          --data_dir=waymo_raw_data/ \
          --output_path=data/camera_data/ \
          --frames_to_skip=3 \
          --num_cores=8