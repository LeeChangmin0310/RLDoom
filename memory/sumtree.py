# memory/sumtree.py

import numpy as np


class SumTree:
    """Binary Sum Tree for Prioritized Experience Replay."""

    def __init__(self, capacity):
        # Number of leaf nodes
        self.capacity = capacity
        # Tree has 2 * capacity - 1 nodes
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        # Data array holds experiences
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, priority, data):
        """Add a new experience with given priority."""
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        """Update priority and propagate the change upwards."""
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        """
        Retrieve leaf for cumulative sum v.

        Returns:
            leaf_index (int)
            priority (float)
            data (object)
        """
        parent_index = 0

        while True:
            left = 2 * parent_index + 1
            right = left + 1

            if left >= len(self.tree):
                leaf_index = parent_index
                break

            if v <= self.tree[left]:
                parent_index = left
            else:
                v -= self.tree[left]
                parent_index = right

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        """Return the sum of all priorities (root node)."""
        return self.tree[0]
