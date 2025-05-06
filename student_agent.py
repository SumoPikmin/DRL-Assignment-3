# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import gym
import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque 

# ------------------------------------------------------------
# Frame Preprocessing
# ------------------------------------------------------------
def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert RGB (240x256x3) frame to 84×84 grayscale uint8."""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame

# ------------------------------------------------------------
# DQN – Deep Q‑Network Architecture
# (Based on the 2015 Nature DQN paper)
# ------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # 84 → 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 20 → 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 9 → 7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ------------------------------------------------------------
# Setup: Environment, Action Space, and Network Initialization
# ------------------------------------------------------------
action_space = gym.spaces.Discrete(12)  # COMPLEX_MOVEMENT
n_actions = action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_PATH = "mario_dqn_final.pth"

policy_net = DQN(n_actions).to(device)

try:
    checkpoint = torch.load(WEIGHT_PATH, map_location=device)

    # If it's a checkpoint with multiple components
    if "policy_net" in checkpoint:
        state_dict = checkpoint["policy_net"]
    else:
        state_dict = checkpoint  # raw state_dict

    policy_net.load_state_dict(state_dict)
    policy_net.eval()
    use_network = True
except FileNotFoundError:
    print(f"[student_agent] WARNING: '{WEIGHT_PATH}' not found. Agent will act randomly.")
    use_network = False


# ------------------------------------------------------------
# Agent for Evaluation – Acts Greedily with Respect to Q
# ------------------------------------------------------------
class Agent:
    """DQN Agent for Super Mario Bros (COMPLEX_MOVEMENT)."""

    def __init__(self):
        self.skip_count = 0
        self.last_action = 0
        self.frames: deque[np.ndarray] = deque(maxlen=4)

    def act(self, observation: np.ndarray) -> int:
        """Return action for a single environment step."""
        if not use_network:
            return action_space.sample()

        # Reuse last action if skipping
        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action

        # Preprocess and store frame
        frame = _preprocess_frame(observation)
        self.frames.append(frame)

        # Pad to 4 frames if needed (initial steps)
        while len(self.frames) < 4:
            self.frames.append(frame)

        # Prepare state tensor: shape (1, 4, 84, 84)
        state = np.stack(self.frames, axis=0)
        state = torch.from_numpy(state).unsqueeze(0).to(device, dtype=torch.float32) / 255.0

        with torch.no_grad():
            q_values = policy_net(state)
            action = int(q_values.argmax(dim=1).item())

        self.last_action = action
        self.skip_count = 3  # Repeat this action for next 3 steps
        return action
