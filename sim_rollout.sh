#!usr/bin/bash

python gesim_video_gen_examples/infer_gesim.py \
    --config_file=configs/cosmos_model/acwm_cosmos.yaml \
    --image_root=gesim_video_gen_examples/process_000 \
    --extrinsic_root=gesim_video_gen_examples/process_000 \
    --intrinsic_root=gesim_video_gen_examples/process_000 \
    --action_path=gesim_video_gen_examples/process_000/actions.npy \
    --output_path=gesim_video_gen_examples/process_000/output
