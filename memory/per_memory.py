# memory/per_memory.py

import numpy as np
from .sumtree import SumTree


class PERMemory:
    """
    Prioritized Experience Replay buffer.

    Experiences are stored as tuples:
        (state, action, reward, next_state, done)
    """

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

        # PER hyperparameters
        self.epsilon = 0.01     # small value to prevent zero priority
        self.alpha = 0.6        # how much prioritization is used
        self.beta = 0.4         # importance-sampling exponent
        self.beta_increment = 0.001
        self.abs_err_upper = 1.0

    def store(self, experience):
        """Store a new experience with maximal priority."""
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.abs_err_upper
        self.tree.add(max_priority, experience)

    def sample(self, n):
        """
        Sample a mini-batch of size n.

        Returns:
            idxs (np.ndarray): tree indices of sampled experiences
            batch (list): list of experiences
            is_weights (np.ndarray): importance-sampling weights
        """
        idxs = np.empty((n,), dtype=np.int32)
        is_weights = np.empty((n, 1), dtype=np.float32)
        batch = []

        total_priority = self.tree.total_priority
        priority_segment = total_priority / n

        # Update beta toward 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / total_priority
        max_weight = (p_min * n) ** (-self.beta)

        for i in range(n):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            v = np.random.uniform(a, b)

            idx, priority, data = self.tree.get_leaf(v)
            sampling_prob = priority / total_priority

            is_weights[i, 0] = (n * sampling_prob) ** (-self.beta) / max_weight
            idxs[i] = idx
            batch.append(data)

        return idxs, batch, is_weights

    def update_batch(self, idxs, abs_errors):
        """Update priorities for a batch of transitions."""
        abs_errors = abs_errors + self.epsilon
        clipped = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped, self.alpha)

        for idx, p in zip(idxs, ps):
            self.tree.update(idx, p)
