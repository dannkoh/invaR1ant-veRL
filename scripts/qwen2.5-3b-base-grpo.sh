set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/ConStruct-veRL/data/train-base-v2.parquet \
    data.val_files=$HOME/ConStruct-veRL/data/test-base-v2.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=614 \
    data.max_prompt_length=1000 \
    data.max_response_length=1000 \
    data.instruct=False \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    +trainer.val_before_train=False \
    trainer.project_name=ConStruct \
    trainer.experiment_name="qwen2.5-3b-base-grpo" \
    trainer.n_gpus_per_node=2 \
    trainer.save_freq=3 \
    trainer.test_freq=1 \
    trainer.total_epochs=15 $@