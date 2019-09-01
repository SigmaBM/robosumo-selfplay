import tensorflow as tf


def get_default_params():
    return dict(
        nsteps=8192,
        nminibatches=32,
        lam=0.95,
        gamma=0.995,
        noptepochs=6,
        log_interval=1,
        save_interval=1,
        ent_coef=0.0,
        lr=1e-3,
        cliprange=0.2,
        value_network='copy',
        anneal_bound=1000,
        num_hidden=64,
        activation=tf.nn.relu,
    )