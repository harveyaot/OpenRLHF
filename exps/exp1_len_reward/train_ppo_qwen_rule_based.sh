#!/bin/bash
set -x

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
python exp1_len_reward/reward_server.py &


read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
   --pretrain Qwen/Qwen2.5-0.5B-Instruct \
   --save_path ./checkpoint/qwen-0.5b-rlhf \
   --save_steps 100 \
   --logging_steps 1 \
   --eval_steps 500 \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 512 \
   --max_epochs 1 \
   --prompt_max_len 512 \
   --generate_max_len 512 \
   --zero_stage 1 \
   --bf16 \
   --actor_learning_rate 1e-5 \
   --critic_learning_rate 1e-4 \
   --init_kl_coef 0.1 \
   --prompt_data data/OpenRLHF/prompt-collection-v0.1-dev-100k \
   --input_key context_messages \
   --apply_chat_template \
   --max_samples 5000 \
   --normalize_reward \
   --gradient_checkpointing \
   --remote_rm_url http://localhost:5000/reward
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
