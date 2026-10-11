"""Online-only uniform replay, with episode-safe three-step returns.

The source return equation is preserved; truncated tails use gamma**actual_length.
Lazy chunks preserve the configured capacity without allocating 10M rows at startup.
"""

from collections import deque
import numpy as np


class NStepReplay:
    def __init__(
        self,
        capacity,
        observation_dim,
        action_dim=3,
        *,
        gamma=0.992,
        n_step=3,
        seed=0,
        chunk_size=8192,
    ):
        if any(
            type(v) is not int or v < 1
            for v in (capacity, observation_dim, action_dim, n_step, chunk_size)
        ):
            raise ValueError(
                "Replay dimensions, capacity and return length must be positive integers"
            )
        if not np.isfinite(gamma) or not 0 < gamma < 1:
            raise ValueError("gamma must be in (0, 1)")
        self.capacity, self.observation_dim, self.action_dim = capacity, observation_dim, action_dim
        self.gamma, self.n_step = gamma, n_step
        self.chunk_size = min(capacity, chunk_size)
        self.chunks = {}
        self.pending = deque()
        self.position = self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def append(self, observation, action, reward, next_observation, terminated, truncated=False):
        arrays = [np.asarray(x, np.float32) for x in (observation, action, next_observation)]
        if (
            arrays[0].shape != (self.observation_dim,)
            or arrays[1].shape != (self.action_dim,)
            or arrays[2].shape != (self.observation_dim,)
            or not all(np.isfinite(x).all() for x in arrays)
            or not np.isfinite(reward)
        ):
            raise ValueError("Replay transition has invalid dimensions or nonfinite values")
        self.pending.append(
            (*(x.copy() for x in arrays[:2]), float(reward), arrays[2].copy(), bool(terminated))
        )
        if len(self.pending) == self.n_step:
            self._emit()
        if terminated or truncated:
            self.flush()

    def flush(self):
        """End a rollout without connecting it to the next reset; preserve bootstrap on cutoffs."""
        while self.pending:
            self._emit()

    def _emit(self):
        length = len(self.pending)
        reward = sum(self.gamma**i * entry[2] for i, entry in enumerate(self.pending))
        first, last = self.pending[0], self.pending[-1]
        transition = (first[0], first[1], reward, last[3], 0.0 if last[4] else self.gamma**length)
        chunk_id, index = divmod(self.position, self.chunk_size)
        if chunk_id not in self.chunks:
            count = min(self.chunk_size, self.capacity - chunk_id * self.chunk_size)
            self.chunks[chunk_id] = (
                np.empty((count, self.observation_dim), np.float32),
                np.empty((count, self.action_dim), np.float32),
                np.empty((count, 1), np.float32),
                np.empty((count, self.observation_dim), np.float32),
                np.empty((count, 1), np.float32),
            )
        for array, value in zip(self.chunks[chunk_id], transition):
            array[index] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.pending.popleft()

    def get(self, indices):
        indices = np.asarray(indices, dtype=int)
        if indices.ndim != 1 or (indices < 0).any() or (indices >= self.size).any():
            raise ValueError("Invalid replay indices")
        arrays = [
            np.empty((len(indices), dim), np.float32)
            for dim in (self.observation_dim, self.action_dim, 1, self.observation_dim, 1)
        ]
        for chunk_id in np.unique(indices // self.chunk_size):
            selected = indices // self.chunk_size == chunk_id
            local = indices[selected] % self.chunk_size
            for output, source in zip(arrays, self.chunks[int(chunk_id)]):
                output[selected] = source[local]
        return tuple(arrays)

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("Cannot sample empty replay")
        return self.get(self.rng.integers(self.size, size=batch_size))
