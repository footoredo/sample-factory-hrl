def robosuite_override_defaults(env, parser):
    parser.set_defaults(
        batched_sampling=False,
        num_workers=1,
        num_envs_per_worker=1,
        worker_num_splits=1,
        train_for_env_steps=1000000,
        encoder_mlp_layers=[64, 64],
        env_frameskip=1,
        nonlinearity="relu",
        batch_size=256,
        kl_loss_coeff=0.1,
        use_rnn=False,
        adaptive_stddev=False,
        policy_initialization="torch_default",
        reward_scale=1,
        rollout=64,
        max_grad_norm=3.5,
        num_epochs=2,
        num_batches_per_epoch=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=1.3,
        exploration_loss_coeff=0.0,
        learning_rate=0.00295,
        lr_schedule="linear_decay",
        shuffle_minibatches=False,
        gamma=0.99,
        gae_lambda=0.95,
        with_vtrace=False,
        recurrence=1,
        normalize_input=True,
        normalize_returns=True,
        value_bootstrap=True,
        experiment_summaries_interval=3,
        save_every_sec=15,
        serial_mode=False,
        async_rl=False,
        stats_avg=10
    )


# noinspection PyUnusedLocal
def add_robosuite_env_args(env, parser):
    # in case we need to add more args in the future
    pass
