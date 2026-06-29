from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Iterable


ACTIONS: tuple[tuple[int, int], ...] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


@dataclass(frozen=True)
class StepResult:
    observation: tuple[int, ...]
    reward: float
    done: bool
    info: dict[str, int | tuple[int, int]]


class BabyGrid:
    """Tiny grid environment for fast RL smoke tests.

    The agent receives a raw flattened grid with an agent marker, a goal marker,
    and distractor objects. The environment is deliberately small so every host
    in the fleet can run the same verification loop without dependencies.
    """

    def __init__(
        self,
        size: int = 7,
        max_steps: int = 60,
        seed: int = 0,
        obstacle_count: int = 0,
        toy_count: int = 3,
        step_penalty: float = -0.01,
        collision_penalty: float = -0.05,
    ) -> None:
        if size < 4:
            raise ValueError("size must be at least 4")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if obstacle_count < 0:
            raise ValueError("obstacle_count must be non-negative")
        if toy_count < 0:
            raise ValueError("toy_count must be non-negative")
        if obstacle_count + toy_count > size * size - 2:
            raise ValueError("obstacle_count + toy_count leaves no room for agent and goal")
        self.size = size
        self.max_steps = max_steps
        self.obstacle_count = obstacle_count
        self.toy_count = toy_count
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty
        self.rng = Random(seed)
        self.agent = (0, 0)
        self.goal = (size - 1, size - 1)
        self.toys: tuple[tuple[int, int], ...] = ()
        self.walls: tuple[tuple[int, int], ...] = ()
        self.steps = 0

    def reset(self, seed: int | None = None) -> tuple[int, ...]:
        if seed is not None:
            self.rng.seed(seed)
        self.steps = 0
        self.agent = self._sample_empty(())
        self.goal = self._sample_empty((self.agent,))
        self.walls = self._sample_walls((self.agent, self.goal))
        toys: list[tuple[int, int]] = []
        while len(toys) < self.toy_count:
            cell = self._sample_empty((self.agent, self.goal, *self.walls, *toys))
            toys.append(cell)
        self.toys = tuple(toys)
        return self.observation()

    def step(self, action: int) -> StepResult:
        if action < 0 or action >= len(ACTIONS):
            raise ValueError(f"invalid action: {action}")
        dx, dy = ACTIONS[action]
        x, y = self.agent
        nx = min(self.size - 1, max(0, x + dx))
        ny = min(self.size - 1, max(0, y + dy))
        blocked = (nx, ny) in self.walls
        if not blocked:
            self.agent = (nx, ny)
        self.steps += 1
        done = self.agent == self.goal or self.steps >= self.max_steps
        if self.agent == self.goal:
            reward = 1.0
        elif blocked:
            reward = self.collision_penalty
        else:
            reward = self.step_penalty
        return StepResult(
            observation=self.observation(),
            reward=reward,
            done=done,
            info={"steps": self.steps, "agent": self.agent, "goal": self.goal, "blocked": int(blocked)},
        )

    def observation(self) -> tuple[int, ...]:
        cells = [0] * (self.size * self.size)
        for wall in self.walls:
            cells[self._idx(wall)] = 1
        for toy in self.toys:
            cells[self._idx(toy)] = 4
        cells[self._idx(self.goal)] = 3
        cells[self._idx(self.agent)] = 5 if self.agent == self.goal else 2
        return tuple(cells)

    def _idx(self, cell: tuple[int, int]) -> int:
        x, y = cell
        return y * self.size + x

    def _sample_empty(self, occupied: Iterable[tuple[int, int]]) -> tuple[int, int]:
        occupied_set = set(occupied)
        while True:
            cell = (self.rng.randrange(self.size), self.rng.randrange(self.size))
            if cell not in occupied_set:
                return cell

    def _sample_walls(self, occupied: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
        walls: list[tuple[int, int]] = []
        max_attempts = max(100, self.obstacle_count * 50)
        occupied_set = set(occupied)
        attempts = 0
        while len(walls) < self.obstacle_count and attempts < max_attempts:
            attempts += 1
            candidate = self._sample_empty((*occupied_set, *walls))
            trial = [*walls, candidate]
            if self._path_exists(set(trial)):
                walls.append(candidate)
        if len(walls) != self.obstacle_count:
            raise RuntimeError("could not sample reachable wall layout")
        return tuple(walls)

    def _path_exists(self, walls: set[tuple[int, int]]) -> bool:
        queue: deque[tuple[int, int]] = deque([self.agent])
        seen = {self.agent}
        while queue:
            x, y = queue.popleft()
            if (x, y) == self.goal:
                return True
            for dx, dy in ACTIONS:
                nx = min(self.size - 1, max(0, x + dx))
                ny = min(self.size - 1, max(0, y + dy))
                cell = (nx, ny)
                if cell in walls or cell in seen:
                    continue
                seen.add(cell)
                queue.append(cell)
        return False


class FeatureEncoder:
    """Perception module used by the v0 experiment conditions."""

    def __init__(self, size: int, mode: str) -> None:
        if mode not in {"raw", "coarse"}:
            raise ValueError(f"unknown encoder mode: {mode}")
        self.size = size
        self.mode = mode

    def encode(self, observation: tuple[int, ...]) -> tuple[int, ...]:
        if self.mode == "raw":
            return observation
        agent = self._find(observation, (2, 5))
        goal = self._find(observation, (3, 5))
        dx = goal[0] - agent[0]
        dy = goal[1] - agent[1]
        return (
            self._sign(dx),
            self._sign(dy),
            min(abs(dx) + abs(dy), 4),
            self._toy_quadrant(observation, agent),
            self._blocked_mask(observation, agent),
        )

    def _find(self, observation: tuple[int, ...], values: tuple[int, ...]) -> tuple[int, int]:
        idx = next(i for i, value in enumerate(observation) if value in values)
        return (idx % self.size, idx // self.size)

    def _toy_quadrant(self, observation: tuple[int, ...], agent: tuple[int, int]) -> int:
        toy_indices = [i for i, value in enumerate(observation) if value == 4]
        if not toy_indices:
            return 0
        tx = sum(i % self.size for i in toy_indices) / len(toy_indices)
        ty = sum(i // self.size for i in toy_indices) / len(toy_indices)
        return (1 if tx >= agent[0] else 0) + (2 if ty >= agent[1] else 0)

    def _blocked_mask(self, observation: tuple[int, ...], agent: tuple[int, int]) -> int:
        mask = 0
        for bit, (dx, dy) in enumerate(ACTIONS):
            nx = min(self.size - 1, max(0, agent[0] + dx))
            ny = min(self.size - 1, max(0, agent[1] + dy))
            if observation[ny * self.size + nx] == 1:
                mask |= 1 << bit
        return mask

    def _sign(self, value: int) -> int:
        if value < 0:
            return -1
        if value > 0:
            return 1
        return 0
