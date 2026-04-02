accelerate launch --num_processes 8 --mixed_precision bf16 test.py   \
  --fusion_mode dsab \
  --dsab_stage pre_warpers \
  --dsab_mix_space logprob \
  --dsab_alpha_gamma 1.0 \
  --dsab_alpha_use_d2 \
  --dsab_use_safeguards \
  --dsab_lite \
  --dsab_alpha_min 0.1 \
  --dsab_alpha_max 0.9 \

  
  # --sim_mode alpha_bias \
  # --sim_beta 0.1 \
  # --sim_center 0.5 \
  # --sim_mode alpha \
  # --sim_pool max \
  # --sim_alpha_weight 0.8
