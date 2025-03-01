set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/ConStruct-veRL/data/train.parquet \
    data.val_files=$HOME/ConStruct-veRL/data/test.parquet \
    data.train_batch_size=1024 \
    data.val_batch_size=1024 \
    data.max_prompt_length=1800 \
    data.max_response_length=4096 \
    data.instruct=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=80 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    +trainer.val_before_train=False \
    trainer.project_name=ConStruct \
    trainer.experiment_name='qwen2.5-1.5b-instruct-grpo' \
    trainer.n_gpus_per_node=2 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=15 $@