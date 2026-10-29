set -x

# Isolated env (torch 2.7.1+cu128 for RTX 5090 / sm_120, robosuite 1.4.1 for LIBERO); see DEPENDENCY_CHANGES.md
source /opt/simplevla/.venv/bin/activate
# Run from the repo root so `python -m verl...` (and Ray workers) import SimpleVLA-RL's verl, not upstream veRL
cd "$(dirname "$(readlink -f "$0")")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH}"
# torch>=2.6 defaults torch.load(weights_only=True); LIBERO's *.pruned_init files need the old behaviour
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# Container has pids.max=2048; Ray prestarting one worker per CPU (32) exhausts it ("thread: Resource temporarily unavailable")
export RAY_prestart_worker_first_driver=0
export RAY_enable_worker_prestart=0
# CPU rendering for LIBERO. EGL (GPU) is ~7x faster per sim step, but env creation failed with EGL_BAD_ALLOC
# during RL rollouts (likely GPU memory contention). Try MUJOCO_GL=egl / PYOPENGL_PLATFORM=egl when the GPU is not shared.
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

export NCCL_DEBUG=WARN
export WANDB_API_KEY='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'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export ROBOT_PLATFORM=LIBERO  # Use LIBERO: ROBOT_PLATFORM=LIBERO  Use Robotwin ROBOT_PLATFORM=ALOHA
PROJECT_NAME='SimpleVLA-RL'
EXPERIMENT_NAME="openvla-oft_libero10_lora"
# For openvla-oft Libero-Long traj1 SFT or traj all SFT models can be find in https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86
SFT_MODEL_PATH="/home/DockerShared/ECE780/SimpleVLA-RL/models/Openvla-oft-SFT-libero10-traj1"
CKPT_PATH="/home/DockerShared/ECE780/SimpleVLA-RL/examples/trained_checkpoints"
# DATASET_NAME can be libero_10 (libero_Long), libero_90, libero_spatial, libero_object, libero_goal
DATASET_NAME="libero_10"
VLA_NAME="openvla-oft"
NUM_GPUS=1
# If you want to use 2*8 GPU to RL. Set NUM_NODES=2
NUM_NODES=1 
ALIGN_PATH="$(pwd)/align.json"
# LoRA RL (single 32 GB GPU): frozen base loaded in bf16 (~15 GB), only LoRA adapters trained (kept in fp32).
# Full fine-tuning needs ~120 GB (fp32 weights + grads + Adam) and does not fit.
LORA_RANK=32
LORA_ALPHA=32
# LoRA usually needs a higher LR than full fine-tuning (the original full-FT script used 5e-6); tune as needed
LR=5e-5
# log_prob_micro_batch_size is per-node and gets divided by NUM_GPUS. Each micro-batch must hold trajectories
# of a single task (same prompt padding, else dp_rob.py raises "Padding error!"), so the per-GPU value must divide n_samples.
bash /home/DockerShared/ECE780/SimpleVLA-RL/examples/overwrite_vla_ckpt_utils.sh $SFT_MODEL_PATH 

HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=50 \
    data.n_samples=8 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=64 \
    data.val_batch_size=496 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.use_proprio=False \
    actor_rollout_ref.rollout.val_micro_batch_size=8 \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.unnorm_key=$DATASET_NAME \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=25 \
    trainer.test_freq=4 \
    trainer.total_epochs=100 \
    trainer.val_only=False \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.wandb_mode=online \
    trainer.val_before_train=True \


