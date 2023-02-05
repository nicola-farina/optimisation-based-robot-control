import orc.assignment_03.src.environment.pendulum as environment
from orc.assignment_03.src.network import DQNet, Network


def main():
    hyper_params = DQNet.HyperParams(
        replay_size=5000,
        replay_start=500,
        discount=0.9,
        max_episodes=500,
        max_steps_per_episode=100,
        steps_for_target_update=100,
        epsilon_start=1.,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        batch_size=256,
        learning_rate=0.09,
        display_every_episodes=25
    )

    num_joints = 1
    num_controls = 32
    name = "single"

    env = environment.SinglePendulum(num_controls=num_controls, max_torque=10.)
    model = Network.get_model(num_joints*2, num_controls, name)

    dq = DQNet(model, hyper_params, env)

    best_model = dq.train()

    if best_model is not None:
        best_model.save(f"models/{name}")


if __name__ == "__main__":
    main()
