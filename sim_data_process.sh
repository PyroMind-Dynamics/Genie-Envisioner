#!/usr/bin/bash

python gesim_video_gen_examples/get_example_gesim_inputs.py \
	--data_root=/workspace/AgiBotWorld-Alpha \
	--task_id=327 \
	--episode_id=648642 \
	--save_root=gesim_video_gen_examples/process_000 \
	--valid_start=0 \
	--valid_end=300
