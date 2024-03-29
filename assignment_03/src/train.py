import argparse
from enum import Enum

import numpy.random as random
from dqn.algorithm import DQL
from dqn.network import Network
from environment.double_pendulum import UnderactDoublePendulumEnv
from environment.single_pendulum import SinglePendulumEnv


class PendulumType(Enum):
    SINGLE = "single"
    DOUBLE = "double"


def main(
        hyper_params: DQL.HyperParams,
        name: str,
        num_controls: int,
        max_vel: float,
        max_torque: float,
        pend_type: PendulumType,
        rng_seed: int = None
):
    if rng_seed is not None:
        rng = random.default_rng(seed=rng_seed)
    else:
        rng = random.default_rng()

    if pend_type == pend_type.SINGLE:
        num_joints = 1
        model = Network.get_single_model(num_joints * 2, num_controls)
        env = SinglePendulumEnv(
            num_controls, max_vel, max_torque, rng
        )
    elif pend_type == pend_type.DOUBLE:
        num_joints = 2
        model = Network.get_double_model(num_joints * 2, num_controls)
        env = UnderactDoublePendulumEnv(
            num_controls, max_vel, max_torque, rng
        )
    else:
        print("Not supported.")
        return

    dql = DQL(name, model, hyper_params, env, rng)

    last_model = dql.train()
    last_model.save_weights(f"{dql.model_folder}/last_weights.h5")


if __name__ == "__main__":
    possible_types = [str(x.value) for x in PendulumType]

    parser = argparse.ArgumentParser()

    parser.add_argument("type", help="the type of pendulum to use", type=str, choices=possible_types)
    parser.add_argument("name", help="the name of the experiment", type=str)

    parser.add_argument("--controls", required=False, help="how many controls to use", type=int, default=11)
    parser.add_argument("--max-vel", required=False, help="the maximum velocity", type=float, default=5.0)
    parser.add_argument("--max-torque", required=False, help="the maximum torque", type=float, default=5.0)
    parser.add_argument("--seed", required=False, help="the rng seed", type=int)

    parser.add_argument("--replay-size", required=False, help="the replay buffer size", type=int, default=10000)
    parser.add_argument(
        "--replay-start", required=False, help="how many steps to start replay training", type=int, default=1000
    )
    parser.add_argument("--discount", required=False, help="the discount factor", type=float, default=0.99)
    parser.add_argument("--max-episodes", required=False, help="the maximum number of episodes", type=int, default=100)
    parser.add_argument(
        "--max-steps", required=False, help="the maximum number of steps per episode", type=int, default=500
    )
    parser.add_argument(
        "--sync-target", required=False, help="how often (steps) to sync target network", type=int, default=1000
    )
    parser.add_argument("--eps-start", required=False, help="the starting value of epsilon", type=float, default=1.0)
    parser.add_argument(
        "--eps-decay", required=False, help="the decay of epsilon (eps=eps*decay)", type=float, default=0.995
    )
    parser.add_argument("--eps-min", required=False, help="the minimum value of epsilon", type=float, default=0.005)
    parser.add_argument("--batch-size", required=False, help="the size of an experience batch", type=int, default=128)
    parser.add_argument("--lr", required=False, help="the initial learning rate", type=float, default=0.001)
    parser.add_argument("--display-rate", required=False, help="how often (episodes) to display", type=int, default=10)

    args = parser.parse_args()

    hp = DQL.HyperParams(
        replay_size=args.replay_size,
        replay_start=args.replay_start,
        discount=args.discount,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps,
        steps_for_target_update=args.sync_target,
        epsilon_start=args.eps_start,
        epsilon_decay=args.eps_decay,
        epsilon_min=args.eps_min,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        display_every_episodes=args.display_rate
    )

    main(
        hyper_params=hp,
        name=args.name,
        num_controls=args.controls,
        max_vel=args.max_vel,
        max_torque=args.max_torque,
        pend_type=PendulumType(args.type),
        rng_seed=args.seed
    )
