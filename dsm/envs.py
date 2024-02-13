import functools
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dsm.types import Environment


class WindyCartesianEnv(gym.Env):
    """Windy grid world."""

    def __init__(
        self,
        size: float = 10.0,
        vel: float = 1.0,
        wind_scale: float = 0.25,
        max_steps: int = 200,
        reward_fn: Callable[[np.ndarray, float], float] | None = None,
    ):
        super().__init__()
        self.size = size
        self.vel = vel
        self.wind_scale = wind_scale
        self.max_steps = max_steps
        self.state = np.zeros(2)
        self.clock = 0
        if reward_fn is None:
            reward_fn = self._default_reward_fn

        self.reward_fn = reward_fn

        self._action_to_vel = {
            0: np.array([-1.0, 0.0]),  # Left
            1: np.array([1.0, 0.0]),  # Right
            2: np.array([0.0, -1.0]),  # Down
            3: np.array([0.0, 1.0]),  # Up
        }

    def _default_reward_fn(self, x: np.ndarray, a: float) -> float:
        return 0.0

    @functools.cached_property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            shape=(2,),
            dtype=np.float32,
            low=-self.size * np.ones(2) / 2,
            high=self.size * np.ones(2) / 2,
        )

    @functools.cached_property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(4)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[Any, Any] | None = None,
    ) -> tuple[Any, dict[Any, Any]]:
        self.state = np.zeros(2)
        self.clock = 0
        return self.state, {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[Any, Any]]:
        r = self.reward_fn(self.state, action)
        vel = self._action_to_vel[action] * self.vel
        wind_dir = np.sign(self.state)
        wind = np.random.uniform(size=(2,)) * self.wind_scale
        self.state = (self.state + vel + wind * wind_dir).clip(-self.size / 2, self.size / 2)
        self.clock += 1
        done = self.clock >= self.max_steps

        return self.state, r, False, done, {}


def make(env_id: Environment) -> gym.Env:
    match env_id:
        case "Pendulum-v1":
            return gym.make(env_id)
        case "WindyGridWorld-v0" | "WindyGridWorld-top-v0" | "WindyGridWorld-bottom-v0":
            return WindyCartesianEnv()
        case _:
            raise NotImplementedError
