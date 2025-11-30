# memory/per_memory.py

import numpy as np
from .sumtree import SumTree


class PERMemory:
    """Prioritized Experience Replay buffer with SumTree backend."""

    def __init__(self, capacity: int):
        # SumTree stores (priority, data) pairs
        self.tree = SumTree(capacity)

        # PER hyperparameters
        self.epsilon = 0.01     # small constant to avoid zero priority
        self.alpha = 0.6        # how much prioritization is used (0: no PER, 1: full PER)
        self.beta = 0.4         # importance-sampling exponent
        self.beta_increment = 0.001
        self.abs_err_upper = 1.0  # clip TD error

    def store(self, experience):
        """Store a new experience with maximal priority."""
        # Guard: experience should be a 5-tuple (s, a, r, s', done)
        if not isinstance(experience, tuple) or len(experience) != 5:
            raise ValueError(
                "Experience must be a 5-tuple (state, action, reward, next_state, done). "
                f"Got type={type(experience)}, value={experience}"
            )

        # Get current max priority among leaves
        leaf_priorities = self.tree.tree[-self.tree.capacity:]
        max_priority = np.max(leaf_priorities)

        # If tree is empty or all zeros, use upper bound
        if max_priority <= 0.0:
            max_priority = self.abs_err_upper

        # Add to tree
        self.tree.add(max_priority, experience)

    def sample(self, n: int):
        """
        Sample a mini-batch of size n.

        Returns:
            idxs (np.ndarray): tree indices of sampled experiences
            batch (list): list of (s, a, r, s', done)
            is_weights (np.ndarray): importance-sampling weights
        """
        idxs = np.empty((n,), dtype=np.int32)
        is_weights = np.empty((n, 1), dtype=np.float32)
        batch = []

        total_priority = self.tree.total_priority
        if total_priority <= 0.0:
            # No valid transitions have been stored yet
            raise RuntimeError("SumTree total_priority is zero. "
                               "You are sampling before storing any transitions.")

        # Increase beta toward 1 (more correction over time)
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Use only non-zero leaf priorities to compute p_min
        leaf_priorities = self.tree.tree[-self.tree.capacity:]
        non_zero = leaf_priorities[leaf_priorities > 0.0]

        if non_zero.size == 0:
            # Fallback: assume uniform minimal probability
            p_min = 1.0 / self.tree.capacity
        else:
            p_min = np.min(non_zero) / total_priority

        max_weight = (p_min * n) ** (-self.beta)

        # Segment-based sampling (standard PER)
        priority_segment = total_priority / n

        for i in range(n):
            a = priority_segment * i
            b = priority_segment * (i + 1)

            # Keep drawing until we get a valid experience tuple
            while True:
                v = np.random.uniform(a, b)
                idx, priority, data = self.tree.get_leaf(v)

                # Skip empty / invalid slots (e.g., initial zeros)
                if not isinstance(data, tuple) or len(data) != 5:
                    continue

                # Avoid zero probability
                if priority <= 0.0:
                    sampling_prob = p_min
                else:
                    sampling_prob = priority / total_priority

                w = (n * sampling_prob) ** (-self.beta) / max_weight

                idxs[i] = idx
                batch.append(data)
                is_weights[i, 0] = w
                break

        return idxs, batch, is_weights

    def update_batch(self, idxs, abs_errors):
        """Update priorities for a batch of transitions."""
        abs_errors = abs_errors + self.epsilon
        clipped = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped, self.alpha)

        for idx, p in zip(idxs, ps):
            self.tree.update(idx, p)
