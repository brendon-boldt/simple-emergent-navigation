from typing import Tuple, Any, Dict
from abc import ABC, abstractmethod

import gym  # type: ignore
from gym import spaces
import numpy as np  # type: ignore

StepResult = Tuple[Any, float, bool, Dict]

rng = np.random.default_rng()
GOLDEN_RATIO = (np.sqrt(5) + 1) / 2
GOLDEN_ANGLE = 2 * np.pi * (2 - GOLDEN_RATIO)


def get_norm(x):
    return np.sqrt((x ** 2).sum(-1))


def cosine_similarity(x, y) -> float:
    return (x * y).sum() / max(get_norm(x) * get_norm(y), 1e-5)


class Navigation(gym.Env, ABC):
    def __init__(
        self,
        *,
        is_eval: bool,
        goal_radius: float,
        world_radius: float,
        max_step_scale: float,
        sparsity: float,
        biased_reward_shaping: bool,
        **kwargs,
    ) -> None:
        super().__init__()
        self.goal_radius = goal_radius
        self.world_radius = world_radius
        self.is_eval = is_eval
        self.max_step_scale = max_step_scale
        self.sparsity = sparsity
        self.biased_reward_shaping = biased_reward_shaping

        self.max_steps = int(self.world_scale * self.max_step_scale)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        # For type purposes
        self.num_steps = 0
        self.stop = False
        self.location = np.zeros(self.observation_space.shape)
        self.prev_location = self.location.copy()

    def _take_action(self, action: np.ndarray) -> None:
        try:
            assert type(action) == np.ndarray
            assert action.shape == (2,)
        except Exception as e:
            print(type(action))
            print(action)
            print(action.shape)
            raise e
        act_norm = get_norm(action)
        if act_norm > 1:
            action /= act_norm
        action /= self.world_scale
        self.location += action

    def get_observation(self) -> np.ndarray:
        return self.location

    def _get_step_result(self, action: np.ndarray) -> StepResult:
        info: Dict[str, Any] = {}

        observation = self.get_observation()

        at_goal = self._at_goal()

        if (self.num_steps > 0 and at_goal) or self.num_steps > self.max_steps:
            self.stop = True

        # prev_vec = action - self.location
        # cosine_sim = cosine_similarity(prev_vec, action)
        if self.is_eval:
            reward = float(at_goal)
            info["at_goal"] = at_goal
        else:
            info["at_goal"] = at_goal
            reward = 0.0
            # reward += (cosine_sim - 1) / self.sparsity
            # Shaped reward of +1 for moving towards the goal one step
            reward += self._get_shaped_reward() / self.sparsity
            if self.stop and at_goal:
                reward += 1.0
        return observation, reward, self.stop, info

    def step(self, action: np.ndarray) -> StepResult:
        if self.stop:
            raise Exception("Cannot take action after the agent has stopped")
        # If the max has been reached, force the agent to stop
        self.num_steps += 1
        self.prev_location = self.location.copy()
        self._take_action(action)
        obs, reward, done, info = self._get_step_result(action)
        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_location()
        self.stop = False
        self.num_steps = 0
        return self.get_observation()

    @abstractmethod
    def fib_disc_init(self, i, n) -> np.ndarray:
        pass

    @abstractmethod
    def _get_shaped_reward(self) -> float:
        pass

    @abstractmethod
    def _at_goal(self) -> bool:
        pass

    @abstractmethod
    def _reset_location(self) -> None:
        pass

class NavToCenter(Navigation):
    def __init__(self, *, world_radius: float, **kwargs) -> None:
        self.world_scale = world_radius
        super().__init__(world_radius=world_radius, **kwargs)

    def _get_shaped_reward(self) -> float:
        cur = get_norm(self.location)
        prev = get_norm(self.prev_location)
        return (prev - cur) * self.world_scale

    def _at_goal(self) -> bool:
        return get_norm(self.location) <= self.goal_radius / self.world_scale

    def _reset_location(self) -> None:
        # Pulled from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        n_dim = 2
        u = rng.normal(0, 1, n_dim)
        norm = (u ** 2).sum() ** 0.5
        radius = np.sqrt(rng.uniform((self.goal_radius / self.world_scale) ** 2, 1.0))
        self.location = radius * u / norm
        self.prev_location = self.location.copy()

    def fib_disc_init(self, i: int, n: int) -> np.ndarray:
        theta = i * GOLDEN_ANGLE
        g_rad = self.goal_radius / self.world_radius
        lo = int(np.ceil(n * g_rad ** 2))
        hi = int(np.ceil(n / (1 - g_rad ** 2)))
        r = np.sqrt((i + lo) / hi)
        if self._at_goal():
            raise ValueError(f"Index i={i} initializes agent within the goal.")
        self.location = r * np.array([np.cos(theta), np.sin(theta)])
        return self.get_observation()



class NavToEdges(Navigation):
    def __init__(self, *, goal_radius: float, **kwargs) -> None:
        self.world_scale = goal_radius
        super().__init__(goal_radius=goal_radius, **kwargs)

    def _get_shaped_reward(self):
        if self.biased_reward_shaping:
            cur = get_norm(self.location[0])
            prev = get_norm(self.prev_location[0])
        else:
            cur = get_norm(self.location)
            prev = get_norm(self.prev_location)
        return (cur - prev) * self.world_scale

    def _at_goal(self) -> bool:
        return get_norm(self.location) >= 1

    def _reset_location(self) -> None:
        # Pulled from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        n_dim = 2
        u = rng.normal(0, 1, n_dim)
        norm = (u ** 2).sum() ** 0.5
        radius = np.sqrt(rng.uniform(0, self.world_radius / self.world_scale))
        self.location = radius * u / norm
        self.prev_location = self.location.copy()
        self.prev_location = self.location.copy()

    def fib_disc_init(self, i: int, n: int) -> np.ndarray:
        theta = i * GOLDEN_ANGLE
        r = np.sqrt(i / n) * self.world_radius / self.world_scale
        if self._at_goal():
            raise ValueError(f"Index i={i} initializes agent within the goal.")
        self.location = r * np.array([np.cos(theta), np.sin(theta)])
        return self.get_observation()
