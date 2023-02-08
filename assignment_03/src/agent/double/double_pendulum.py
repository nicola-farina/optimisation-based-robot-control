import time
from typing import Tuple

from example_robot_data.robots_loader import load
import numpy as np
import numpy.typing as npt
import pinocchio as pin
from agent.double.robot_simulator import RobotSimulator
from agent.double.robot_wrapper import RobotWrapper
from agent.pendulum import PendulumAgent
from agent.utils import NumpyUtils


class UnderactDoublePendulumAgent(PendulumAgent):

    def __init__(
            self,
            max_vel: float,
            max_torque: float,
            sim_time_step: float
    ):
        super(UnderactDoublePendulumAgent, self).__init__(
            2, max_vel, max_torque, sim_time_step
        )

        # Load robot agent and wrap it
        r = load("double_pendulum")
        self._robot = RobotWrapper(r.model, r.collision_model, r.visual_model)

        # Simulation wrapper on the robot
        self._simu = RobotSimulator(sim_time_step, self._robot)

    @property
    def joint_angles_size(self):
        return self._robot.nq

    @property
    def joint_velocities_size(self):
        return self._robot.nv

    @property
    def state_size(self):
        return self.joint_angles_size + self.joint_velocities_size

    @property
    def control_size(self):
        return self.joint_velocities_size

    def display(self, joint_angles: npt.NDArray):
        self._simu.display(joint_angles)
        time.sleep(self._sim_time_step)

    def dynamics(
            self,
            state: npt.NDArray,
            control: npt.NDArray
    ) -> Tuple[npt.NDArray, float]:
        nq = self._robot.nq
        nv = self._robot.nv

        q = NumpyUtils.modulo_pi(np.copy(state[:nq]))
        v = np.copy(state[nq:])
        u = np.clip(np.reshape(np.copy(control), nq), -self._max_torque, self._max_torque)

        self._robot.computeAllTerms(q, v)

        model = self._robot.model
        data = self._robot.data

        dx = np.zeros(nq+nv)

        ddq = pin.aba(model, data, q, v, u)

        dx[nv:] = ddq
        v_mean = v + 0.5 * self.sim_time_step * ddq
        dx[:nv] = v_mean

        new_state = np.copy(state) + self.sim_time_step * dx

        new_state[:nq] = NumpyUtils.modulo_pi((new_state[:nq]))
        new_state[nq:] = np.clip(new_state[nq:], -self._max_vel, self._max_vel)

        cost = self.cost_function(new_state, u)

        return new_state, cost

