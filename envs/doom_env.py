# envs/doom_env.py

import numpy as np
from collections import deque
from vizdoom import DoomGame
from skimage import transform


class DoomEnv:
    """Wrapper around VizDoom deadly_corridor environment."""

    def __init__(
        self,
        config_path,
        scenario_path,
        frame_height=100,
        frame_width=120,
        stack_size=4,
    ):
        # Initialize Doom game
        self.game = DoomGame()
        self.game.load_config(config_path)
        self.game.set_doom_scenario_path(scenario_path)
        self.game.init()

        # Possible actions as one-hot vectors
        self.possible_actions = np.identity(
            self.game.get_available_buttons_size(), dtype=np.int32
        ).tolist()

        self.frame_height = frame_height
        self.frame_width = frame_width
        self.stack_size = stack_size

        # Deque for frame stacking
        self.stacked_frames = deque(
            [
                np.zeros((frame_height, frame_width), dtype=np.float32)
                for _ in range(stack_size)
            ],
            maxlen=stack_size,
        )

    def _preprocess_frame(self, frame):
        """Crop, normalize, and resize the raw frame."""
        # Crop to remove top and bottom area
        cropped = frame[15:-5, 20:-20]
        normalized = cropped / 255.0
        processed = transform.resize(
            normalized,
            (self.frame_height, self.frame_width),
            anti_aliasing=True,
        ).astype(np.float32)
        return processed

    def _stack_frames(self, frame, is_new_episode):
        """Stack frames over time to encode motion."""
        processed = self._preprocess_frame(frame)

        if is_new_episode:
            # Reset deque with same frame repeated
            self.stacked_frames = deque(
                [
                    np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
                    for _ in range(self.stack_size)
                ],
                maxlen=self.stack_size,
            )
            for _ in range(self.stack_size):
                self.stacked_frames.append(processed)
        else:
            self.stacked_frames.append(processed)

        # Return stacked as (C, H, W)
        stacked_state = np.stack(self.stacked_frames, axis=0)
        return stacked_state

    def reset(self):
        """Reset environment and return initial stacked state."""
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        stacked_state = self._stack_frames(state, is_new_episode=True)
        return stacked_state

    def step(self, action_idx):
        """
        Take one step in the environment.

        Args:
            action_idx (int): index of action to take.

        Returns:
            next_state (np.ndarray): stacked frames (C, H, W)
            reward (float)
            done (bool)
        """
        action = self.possible_actions[action_idx]
        reward = self.game.make_action(action)
        done = self.game.is_episode_finished()

        if done:
            next_state = np.zeros(
                (self.stack_size, self.frame_height, self.frame_width),
                dtype=np.float32,
            )
        else:
            frame = self.game.get_state().screen_buffer
            next_state = self._stack_frames(frame, is_new_episode=False)

        return next_state, reward, done

    def close(self):
        """Close the game instance."""
        self.game.close()
