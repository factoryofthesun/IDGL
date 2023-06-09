#!/bin/bash
. env.sh
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN post_LN --lr 0.001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN post_LN --lr 0.001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN post_LN --lr 0.001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN post_LN --lr 0.001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN post_LN --lr 0.0001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN post_LN --lr 0.0001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN post_LN --lr 0.0001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN post_LN --lr 0.0001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN post_LN --lr 1e-05 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN post_LN --lr 1e-05 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN post_LN --lr 1e-05 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN post_LN --lr 1e-05 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN pre_LN --lr 0.001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN pre_LN --lr 0.001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN pre_LN --lr 0.001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN pre_LN --lr 0.001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN pre_LN --lr 0.0001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN pre_LN --lr 0.0001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN pre_LN --lr 0.0001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN pre_LN --lr 0.0001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN pre_LN --lr 1e-05 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN pre_LN --lr 1e-05 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN pre_LN --lr 1e-05 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN pre_LN --lr 1e-05 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN No --lr 0.001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN No --lr 0.001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN No --lr 0.001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN No --lr 0.001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN No --lr 0.0001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN No --lr 0.0001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN No --lr 0.0001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN No --lr 0.0001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN No --lr 1e-05 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN No --lr 1e-05 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN No --lr 1e-05 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN full --LN No --lr 1e-05 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN post_LN --lr 0.001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN post_LN --lr 0.001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN post_LN --lr 0.001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN post_LN --lr 0.001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN post_LN --lr 0.0001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN post_LN --lr 0.0001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN post_LN --lr 0.0001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN post_LN --lr 0.0001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN post_LN --lr 1e-05 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN post_LN --lr 1e-05 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN post_LN --lr 1e-05 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN post_LN --lr 1e-05 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN pre_LN --lr 0.001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN pre_LN --lr 0.001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN pre_LN --lr 0.001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN pre_LN --lr 0.001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN pre_LN --lr 0.0001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN pre_LN --lr 0.0001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN pre_LN --lr 0.0001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN pre_LN --lr 0.0001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN pre_LN --lr 1e-05 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN pre_LN --lr 1e-05 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN pre_LN --lr 1e-05 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN pre_LN --lr 1e-05 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN No --lr 0.001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN No --lr 0.001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN No --lr 0.001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN No --lr 0.001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN No --lr 0.0001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN No --lr 0.0001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN No --lr 0.0001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN No --lr 0.0001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN No --lr 1e-05 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN No --lr 1e-05 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN No --lr 1e-05 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first --LN No --lr 1e-05 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN post_LN --lr 0.001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN post_LN --lr 0.001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN post_LN --lr 0.001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN post_LN --lr 0.001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN post_LN --lr 0.0001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN post_LN --lr 0.0001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN post_LN --lr 0.0001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN post_LN --lr 0.0001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN post_LN --lr 1e-05 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN post_LN --lr 1e-05 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN post_LN --lr 1e-05 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN post_LN --lr 1e-05 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN pre_LN --lr 0.001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN pre_LN --lr 0.001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN pre_LN --lr 0.001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN pre_LN --lr 0.001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN pre_LN --lr 0.0001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN pre_LN --lr 0.0001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN pre_LN --lr 0.0001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN pre_LN --lr 0.0001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN pre_LN --lr 1e-05 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN pre_LN --lr 1e-05 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN pre_LN --lr 1e-05 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN pre_LN --lr 1e-05 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN No --lr 0.001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN No --lr 0.001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN No --lr 0.001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN No --lr 0.001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN No --lr 0.0001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN No --lr 0.0001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN No --lr 0.0001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN No --lr 0.0001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN No --lr 1e-05 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN No --lr 1e-05 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN No --lr 1e-05 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_last --LN No --lr 1e-05 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN post_LN --lr 0.001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN post_LN --lr 0.001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN post_LN --lr 0.001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN post_LN --lr 0.001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN post_LN --lr 0.0001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN post_LN --lr 0.0001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN post_LN --lr 0.0001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN post_LN --lr 0.0001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN post_LN --lr 1e-05 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN post_LN --lr 1e-05 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN post_LN --lr 1e-05 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN post_LN --lr 1e-05 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN pre_LN --lr 0.001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN pre_LN --lr 0.001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN pre_LN --lr 0.001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN pre_LN --lr 0.001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN pre_LN --lr 0.0001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN pre_LN --lr 0.0001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN pre_LN --lr 0.0001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN pre_LN --lr 0.0001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN pre_LN --lr 1e-05 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN pre_LN --lr 1e-05 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN pre_LN --lr 1e-05 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN pre_LN --lr 1e-05 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN No --lr 0.001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN No --lr 0.001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN No --lr 0.001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN No --lr 0.001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN No --lr 0.0001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN No --lr 0.0001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN No --lr 0.0001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN No --lr 0.0001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN No --lr 1e-05 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN No --lr 1e-05 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN No --lr 1e-05 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN not_first_not_last --LN No --lr 1e-05 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN post_LN --lr 0.001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN post_LN --lr 0.001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN post_LN --lr 0.001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN post_LN --lr 0.001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN post_LN --lr 0.0001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN post_LN --lr 0.0001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN post_LN --lr 0.0001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN post_LN --lr 0.0001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN post_LN --lr 1e-05 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN post_LN --lr 1e-05 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN post_LN --lr 1e-05 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN post_LN --lr 1e-05 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN pre_LN --lr 0.001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN pre_LN --lr 0.001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN pre_LN --lr 0.001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN pre_LN --lr 0.001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN pre_LN --lr 0.0001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN pre_LN --lr 0.0001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN pre_LN --lr 0.0001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN pre_LN --lr 0.0001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN pre_LN --lr 1e-05 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN pre_LN --lr 1e-05 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN pre_LN --lr 1e-05 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN pre_LN --lr 1e-05 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN No --lr 0.001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN No --lr 0.001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN No --lr 0.001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN No --lr 0.001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN No --lr 0.0001 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN No --lr 0.0001 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN No --lr 0.0001 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN No --lr 0.0001 --init None
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN No --lr 1e-05 --init extend_ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN No --lr 1e-05 --init ortho
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN No --lr 1e-05 --init extend_eye
python prompt_generator.py --texture yarn --object hamburger  --num_prompts 2 --partition greg-gpu --num_layers 6 --hidden_dim 64   --skip --WN None --LN No --lr 1e-05 --init None
