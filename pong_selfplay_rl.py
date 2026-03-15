# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║         SELF-PLAY PONG  ·  DUAL-AGENT PPO  ·  GOOGLE COLAB GPU            ║
# ║                                                                              ║
# ║  Two independent PPO agents learn Pong from scratch against each other.     ║
# ║  Training is fully recorded → exported as a fast-forwarded MP4 video.      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# HOW TO USE IN COLAB:
#   1. Runtime → Change runtime type → GPU (T4 recommended)
#   2. Copy this file or upload it, then: !python pong_selfplay_rl.py
#      OR paste each cell block into separate Colab cells.
#
# OUTPUT:  ./pong_project/pong_training_cinematic.mp4   ← download this!
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 │ Install Dependencies
# ══════════════════════════════════════════════════════════════════════════════
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("📦 Installing dependencies...")
install("imageio")
install("imageio-ffmpeg")
print("✅ Dependencies ready.")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 │ Imports & Global Config
# ══════════════════════════════════════════════════════════════════════════════
import os, time, math, random
from collections import deque, namedtuple
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

import imageio
from IPython.display import FileLink, display, HTML

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎮 Training device : {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"   GPU             : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Output directory ──────────────────────────────────────────────────────────
PROJECT_DIR = "./pong_project"
os.makedirs(PROJECT_DIR, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 │ Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # ── Environment ───────────────────────────────────────────────────────────
    env_width:          int   = 120       # logical game width
    env_height:         int   = 90        # logical game height
    paddle_height:      int   = 16        # paddle height in game units
    paddle_width:       int   = 4
    ball_radius:        float = 3.5
    ball_speed_init:    float = 3.5
    ball_speed_max:     float = 7.0
    paddle_speed:       float = 4.0
    max_steps_per_ep:   int   = 800

    # ── PPO ───────────────────────────────────────────────────────────────────
    lr:                 float = 3e-4
    gamma:              float = 0.99
    gae_lambda:         float = 0.95
    clip_eps:           float = 0.2
    entropy_coef:       float = 0.02
    value_coef:         float = 0.5
    max_grad_norm:      float = 0.5
    ppo_epochs:         int   = 6
    batch_size:         int   = 256
    rollout_steps:      int   = 1024     # steps to collect before each update

    # ── Training schedule ─────────────────────────────────────────────────────
    total_timesteps:    int   = 600_000  # total env steps (both agents)
    warmup_episodes:    int   = 20       # episodes before self-play switch
    lr_anneal:          bool  = True

    # ── Video recording ───────────────────────────────────────────────────────
    record_every_steps: int   = 15_000   # record a demo episode every N steps
    video_fps:          int   = 30
    video_speedup:      float = 3.0      # final video speed multiplier
    frame_w:            int   = 960
    frame_h:            int   = 680

    # ── Network ───────────────────────────────────────────────────────────────
    obs_dim:            int   = 8        # per-agent observation size
    hidden_dim:         int   = 256
    n_actions:          int   = 3        # UP / STAY / DOWN

CFG = Config()

Transition = namedtuple('Transition', ['obs', 'action', 'logp', 'reward', 'done', 'value'])


# ══════════════════════════════════════════════════════════════════════════════
# CELL 4 │ Pong Environment
# ══════════════════════════════════════════════════════════════════════════════
class PongEnv:
    """
    Two-player Pong.
    Agent 1 controls the LEFT  paddle (blue).
    Agent 2 controls the RIGHT paddle (red).
    Each agent receives its own mirrored 8-dim observation.

    Observation vector (normalised to [-1, 1] or [0, 1]):
        [ball_x, ball_y, ball_vx, ball_vy,
         own_paddle_y, opp_paddle_y, ball_approaching, own_advantage]
    """

    ACTIONS = {0: -1, 1: 0, 2: 1}   # UP=0  STAY=1  DOWN=2

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.W, self.H = cfg.env_width, cfg.env_height
        self.reset()

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        self.ball_x  = self.W / 2
        self.ball_y  = self.H / 2
        angle = random.uniform(-math.pi / 4, math.pi / 4)
        direction = random.choice([-1, 1])
        self.ball_vx = direction * cfg.ball_speed_init * math.cos(angle)
        self.ball_vy = cfg.ball_speed_init * math.sin(angle)

        self.p1_y = self.H / 2     # left paddle
        self.p2_y = self.H / 2     # right paddle
        self.score1 = 0
        self.score2 = 0
        self.steps  = 0
        self.rally  = 0            # consecutive hits in current rally
        return self._obs()

    # ── Observation ───────────────────────────────────────────────────────────
    def _obs(self) -> Tuple[np.ndarray, np.ndarray]:
        W, H = self.W, self.H
        bx  = self.ball_x / W
        by  = self.ball_y / H
        bvx = self.ball_vx / self.cfg.ball_speed_max
        bvy = self.ball_vy / self.cfg.ball_speed_max

        # Relative distance of ball to each paddle (how many frames to reach)
        dist1 = (self.ball_x - self.cfg.paddle_width) / W
        dist2 = (W - self.cfg.paddle_width - self.ball_x) / W

        p1 = self.p1_y / H
        p2 = self.p2_y / H

        obs1 = np.array([bx, by, bvx, bvy, p1, p2, dist1, bvx], dtype=np.float32)
        # Agent 2 sees a horizontally mirrored world
        obs2 = np.array([1 - bx, by, -bvx, bvy, p2, p1, dist2, -bvx], dtype=np.float32)
        return obs1, obs2

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, a1: int, a2: int) -> Tuple[np.ndarray, np.ndarray, float, float, bool, dict]:
        cfg = self.cfg
        half_ph = cfg.paddle_height / 2

        # Move paddles
        self.p1_y += self.ACTIONS[a1] * cfg.paddle_speed
        self.p2_y += self.ACTIONS[a2] * cfg.paddle_speed
        self.p1_y = float(np.clip(self.p1_y, half_ph, self.H - half_ph))
        self.p2_y = float(np.clip(self.p2_y, half_ph, self.H - half_ph))

        # Move ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        r1, r2 = 0.0, 0.0
        done = False
        info = {'scored': None, 'hit': None}

        # Top / bottom wall bounce
        if self.ball_y - cfg.ball_radius <= 0:
            self.ball_vy  = abs(self.ball_vy)
            self.ball_y   = cfg.ball_radius
        elif self.ball_y + cfg.ball_radius >= self.H:
            self.ball_vy  = -abs(self.ball_vy)
            self.ball_y   = self.H - cfg.ball_radius

        # Left paddle hit (agent 1)
        p1_x = cfg.paddle_width
        if (self.ball_x - cfg.ball_radius <= p1_x + cfg.ball_radius and
                self.ball_vx < 0 and
                abs(self.ball_y - self.p1_y) <= half_ph + cfg.ball_radius):
            self.ball_vx = abs(self.ball_vx) * 1.04
            # Add spin based on where the ball hit the paddle
            offset = (self.ball_y - self.p1_y) / half_ph
            self.ball_vy += offset * 1.5
            self.ball_vx = min(self.ball_vx, cfg.ball_speed_max)
            self.ball_vy = float(np.clip(self.ball_vy, -cfg.ball_speed_max, cfg.ball_speed_max))
            self.ball_x  = p1_x + cfg.ball_radius + cfg.ball_radius
            r1 += 0.05          # small reward for returning
            self.rally += 1
            info['hit'] = 1

        # Right paddle hit (agent 2)
        p2_x = self.W - cfg.paddle_width
        if (self.ball_x + cfg.ball_radius >= p2_x - cfg.ball_radius and
                self.ball_vx > 0 and
                abs(self.ball_y - self.p2_y) <= half_ph + cfg.ball_radius):
            self.ball_vx = -abs(self.ball_vx) * 1.04
            offset = (self.ball_y - self.p2_y) / half_ph
            self.ball_vy += offset * 1.5
            self.ball_vx = max(self.ball_vx, -cfg.ball_speed_max)
            self.ball_vy = float(np.clip(self.ball_vy, -cfg.ball_speed_max, cfg.ball_speed_max))
            self.ball_x  = p2_x - cfg.ball_radius - cfg.ball_radius
            r2 += 0.05
            self.rally += 1
            info['hit'] = 2

        # Scoring — ball exits left or right
        if self.ball_x < 0:
            r1 -= 1.0; r2 += 1.0
            self.score2 += 1
            done = True
            info['scored'] = 2
        elif self.ball_x > self.W:
            r1 += 1.0; r2 -= 1.0
            self.score1 += 1
            done = True
            info['scored'] = 1

        # Bonus reward for long rallies
        if done and self.rally >= 5:
            bonus = min(self.rally * 0.02, 0.3)
            r1 += bonus; r2 += bonus

        self.steps += 1
        if self.steps >= cfg.max_steps_per_ep:
            done = True

        obs1, obs2 = self._obs()
        return obs1, obs2, r1, r2, done, info


# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 │ Actor-Critic Network
# ══════════════════════════════════════════════════════════════════════════════
class ActorCritic(nn.Module):
    """
    Shared-trunk Actor-Critic.
    The trunk is a 3-layer MLP with LayerNorm + GELU activations.
    Actor head outputs action logits; critic head outputs a scalar value.
    """
    def __init__(self, obs_dim: int = CFG.obs_dim,
                 hidden: int = CFG.hidden_dim,
                 n_actions: int = CFG.n_actions):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        self.actor_head  = nn.Linear(hidden // 2, n_actions)
        self.critic_head = nn.Linear(hidden // 2, 1)

        # Orthogonal init (standard for PPO)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        feat = self.trunk(x)
        return self.actor_head(feat), self.critic_head(feat).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        x    = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, value = self(x)
        dist  = Categorical(logits=logits)
        action = dist.sample()
        logp   = dist.log_prob(action)
        return action.item(), logp.item(), value.item()

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, values = self(obs)
        dist    = Categorical(logits=logits)
        logp    = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, values, entropy


# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 │ PPO Rollout Buffer
# ══════════════════════════════════════════════════════════════════════════════
class RolloutBuffer:
    """Stores one agent's trajectory for PPO update."""

    def __init__(self):
        self.obs:     List[np.ndarray] = []
        self.actions: List[int]        = []
        self.logps:   List[float]      = []
        self.rewards: List[float]      = []
        self.dones:   List[bool]       = []
        self.values:  List[float]      = []

    def push(self, obs, action, logp, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.logps.append(logp)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)

    def compute_returns(self, last_value: float, gamma: float, lam: float):
        """Generalised Advantage Estimation (GAE)."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            nxt_val = last_value if t == n - 1 else self.values[t + 1]
            nxt_non_term = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * nxt_val * nxt_non_term - self.values[t]
            gae   = delta + gamma * lam * nxt_non_term * gae
            advantages[t] = gae
        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns


# ══════════════════════════════════════════════════════════════════════════════
# CELL 7 │ PPO Agent
# ══════════════════════════════════════════════════════════════════════════════
class PPOAgent:
    def __init__(self, name: str, cfg: Config = CFG):
        self.name   = name
        self.cfg    = cfg
        self.net    = ActorCritic().to(DEVICE)
        self.opt    = optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.buffer = RolloutBuffer()

        # Logging
        self.ep_returns:  List[float] = []
        self.ep_lengths:  List[int]   = []
        self.win_log:     List[int]   = []   # 1=win, 0=loss, 0.5=draw
        self.total_steps: int         = 0
        self.updates:     int         = 0

    # ── PPO Update ────────────────────────────────────────────────────────────
    def update(self, last_value: float, progress: float = 0.0):
        cfg = self.cfg
        buf = self.buffer

        if len(buf) == 0:
            return {}

        advantages, returns = buf.compute_returns(last_value, cfg.gamma, cfg.gae_lambda)

        # Tensors
        obs_t   = torch.tensor(np.array(buf.obs),     dtype=torch.float32, device=DEVICE)
        act_t   = torch.tensor(np.array(buf.actions), dtype=torch.long,    device=DEVICE)
        logp_t  = torch.tensor(np.array(buf.logps),   dtype=torch.float32, device=DEVICE)
        adv_t   = torch.tensor(advantages,            dtype=torch.float32, device=DEVICE)
        ret_t   = torch.tensor(returns,               dtype=torch.float32, device=DEVICE)

        # Normalise advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Annealed learning rate
        if cfg.lr_anneal:
            lr_now = cfg.lr * (1.0 - progress)
            for pg in self.opt.param_groups:
                pg['lr'] = max(lr_now, 1e-5)

        metrics = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'approx_kl': []}
        n = len(buf)

        for _ in range(cfg.ppo_epochs):
            idxs = torch.randperm(n, device=DEVICE)
            for start in range(0, n, cfg.batch_size):
                mb = idxs[start: start + cfg.batch_size]
                new_logp, new_val, entropy = self.net.evaluate(obs_t[mb], act_t[mb])

                ratio    = torch.exp(new_logp - logp_t[mb])
                adv_mb   = adv_t[mb]

                # Clipped surrogate objective
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_mb
                p_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                v_loss = nn.functional.mse_loss(new_val, ret_t[mb])

                # Total loss
                loss = p_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy.mean()

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    kl = ((logp_t[mb] - new_logp)**2).mean().item() / 2
                metrics['policy_loss'].append(p_loss.item())
                metrics['value_loss'].append(v_loss.item())
                metrics['entropy'].append(entropy.mean().item())
                metrics['approx_kl'].append(kl)

        self.updates += 1
        buf.clear()
        return {k: np.mean(v) for k, v in metrics.items()}

    @property
    def win_rate(self) -> float:
        recent = self.win_log[-100:] if self.win_log else []
        return np.mean(recent) if recent else 0.5


# ══════════════════════════════════════════════════════════════════════════════
# CELL 8 │ Visual Renderer
# ══════════════════════════════════════════════════════════════════════════════
# Colour palette
C = dict(
    bg        = '#050510',
    court     = '#0b0b20',
    net       = '#1a1a3a',
    p1        = '#00c8ff',      # agent 1 — neon blue
    p2        = '#ff3860',      # agent 2 — neon red
    ball      = '#ffffff',
    ball_glow = '#ffffcc',
    text      = '#e0e0ff',
    grid      = '#111128',
    win1      = '#00c8ff',
    win2      = '#ff3860',
)

def render_frame(env: PongEnv,
                 agent1: PPOAgent,
                 agent2: PPOAgent,
                 step: int,
                 total_steps: int,
                 stats1: Dict,
                 stats2: Dict,
                 cfg: Config = CFG) -> np.ndarray:
    """
    Renders a 960×680 composite frame:
      Left panel  – the Pong game court
      Right panel – live training dashboard (win rates, rewards, loss curves)
    """
    fig = plt.figure(figsize=(cfg.frame_w / 100, cfg.frame_h / 100),
                     facecolor=C['bg'], dpi=100)
    gs  = GridSpec(4, 2, figure=fig,
                   left=0.01, right=0.99, top=0.93, bottom=0.04,
                   hspace=0.45, wspace=0.08,
                   width_ratios=[1.1, 0.9])

    # ─── Header ───────────────────────────────────────────────────────────────
    progress = step / total_steps
    header_txt = (f"SELF-PLAY PONG  ·  PPO  ·  "
                  f"Step {step:,} / {total_steps:,}  "
                  f"({100*progress:.1f}%)")
    fig.text(0.5, 0.97, header_txt, ha='center', va='top',
             fontsize=9, color=C['text'], fontfamily='monospace',
             fontweight='bold')

    # ─── Game Court ───────────────────────────────────────────────────────────
    ax_game = fig.add_subplot(gs[:, 0], facecolor=C['court'])
    ax_game.set_xlim(0, env.W)
    ax_game.set_ylim(0, env.H)
    ax_game.set_aspect('equal')
    ax_game.axis('off')

    # Court border
    for spine_pos in ['top', 'bottom', 'left', 'right']:
        ax_game.spines[spine_pos].set_visible(False)
    border = patches.FancyBboxPatch((0, 0), env.W, env.H,
                                     boxstyle="round,pad=0",
                                     linewidth=2, edgecolor='#222244',
                                     facecolor='none')
    ax_game.add_patch(border)

    # Net (dashed centre line)
    for i in np.arange(2, env.H - 2, 7):
        ax_game.add_patch(patches.Rectangle(
            (env.W / 2 - 0.7, i), 1.4, 3.5, color=C['net'], zorder=1))

    # Paddles with glow effect
    ph   = cfg.paddle_height
    pw   = cfg.paddle_width
    p1_x = 0
    p2_x = env.W - pw

    for (px, py, colour) in [(p1_x, env.p1_y, C['p1']),
                              (p2_x, env.p2_y, C['p2'])]:
        # glow layer
        ax_game.add_patch(patches.FancyBboxPatch(
            (px - 1.5, py - ph/2 - 1.5), pw + 3, ph + 3,
            boxstyle="round,pad=1", linewidth=0,
            facecolor=colour, alpha=0.18, zorder=2))
        # paddle body
        ax_game.add_patch(patches.FancyBboxPatch(
            (px, py - ph/2), pw, ph,
            boxstyle="round,pad=0.5", linewidth=0,
            facecolor=colour, zorder=3))

    # Ball with glow
    bx, by = env.ball_x, env.ball_y
    br     = cfg.ball_radius
    ax_game.add_patch(plt.Circle((bx, by), br * 2.5,
                                  color=C['ball'], alpha=0.08, zorder=2))
    ax_game.add_patch(plt.Circle((bx, by), br * 1.5,
                                  color=C['ball'], alpha=0.25, zorder=2))
    ax_game.add_patch(plt.Circle((bx, by), br,
                                  color=C['ball'], zorder=4))

    # Scores
    ax_game.text(env.W * 0.28, env.H - 6, str(env.score1),
                 ha='center', va='top', fontsize=20, color=C['p1'],
                 fontweight='bold', fontfamily='monospace', zorder=5)
    ax_game.text(env.W * 0.72, env.H - 6, str(env.score2),
                 ha='center', va='top', fontsize=20, color=C['p2'],
                 fontweight='bold', fontfamily='monospace', zorder=5)

    # Agent labels
    ax_game.text(pw + 1, 3, "AGENT 1", fontsize=5.5, color=C['p1'],
                 fontfamily='monospace', va='bottom')
    ax_game.text(env.W - pw - 1, 3, "AGENT 2", fontsize=5.5, color=C['p2'],
                 fontfamily='monospace', va='bottom', ha='right')

    # ─── Dashboard ────────────────────────────────────────────────────────────
    dash_axes = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    for ax in dash_axes:
        ax.set_facecolor(C['court'])
        ax.tick_params(colors=C['text'], labelsize=6)
        for sp in ax.spines.values():
            sp.set_color('#1a1a3a')

    # Panel 0 – Win Rate
    ax = dash_axes[0]
    w  = 60  # smoothing window
    if len(agent1.win_log) > 1:
        wr1 = [np.mean(agent1.win_log[max(0, i-w):i+1])
               for i in range(len(agent1.win_log))]
        wr2 = [np.mean(agent2.win_log[max(0, i-w):i+1])
               for i in range(len(agent2.win_log))]
        x   = np.arange(len(wr1))
        ax.plot(x, wr1, color=C['p1'], lw=1.2, label='Agent 1')
        ax.plot(x, wr2, color=C['p2'], lw=1.2, label='Agent 2')
        ax.axhline(0.5, color='#444', lw=0.6, ls='--')
        ax.fill_between(x, wr1, 0.5, alpha=0.12, color=C['p1'])
        ax.fill_between(x, wr2, 0.5, alpha=0.12, color=C['p2'])
    ax.set_ylim(0, 1)
    ax.set_title('Win Rate (60-ep rolling)', color=C['text'], fontsize=7,
                 pad=2, fontfamily='monospace')
    ax.legend(fontsize=5.5, loc='upper left',
              labelcolor=C['text'], framealpha=0.15)

    # Panel 1 – Episode Return
    ax = dash_axes[1]
    if len(agent1.ep_returns) > 1:
        r1_smooth = np.convolve(agent1.ep_returns,
                                np.ones(min(30, len(agent1.ep_returns))) /
                                min(30, len(agent1.ep_returns)), mode='valid')
        r2_smooth = np.convolve(agent2.ep_returns,
                                np.ones(min(30, len(agent2.ep_returns))) /
                                min(30, len(agent2.ep_returns)), mode='valid')
        ax.plot(r1_smooth, color=C['p1'], lw=1.0, label='A1')
        ax.plot(r2_smooth, color=C['p2'], lw=1.0, label='A2')
        ax.axhline(0, color='#444', lw=0.6, ls='--')
    ax.set_title('Episode Return (30-ep smooth)', color=C['text'], fontsize=7,
                 pad=2, fontfamily='monospace')
    ax.legend(fontsize=5.5, loc='upper left',
              labelcolor=C['text'], framealpha=0.15)

    # Panel 2 – Policy Loss
    ax = dash_axes[2]
    if stats1.get('policy_loss_log') and len(stats1['policy_loss_log']) > 1:
        ax.plot(stats1['policy_loss_log'], color=C['p1'], lw=1.0, label='A1')
        ax.plot(stats2['policy_loss_log'], color=C['p2'], lw=1.0, label='A2')
    ax.set_title('Policy Loss', color=C['text'], fontsize=7,
                 pad=2, fontfamily='monospace')
    ax.legend(fontsize=5.5, loc='upper right',
              labelcolor=C['text'], framealpha=0.15)

    # Panel 3 – Entropy
    ax = dash_axes[3]
    if stats1.get('entropy_log') and len(stats1['entropy_log']) > 1:
        ax.plot(stats1['entropy_log'], color=C['p1'], lw=1.0, label='A1')
        ax.plot(stats2['entropy_log'], color=C['p2'], lw=1.0, label='A2')
    ax.set_title('Policy Entropy', color=C['text'], fontsize=7,
                 pad=2, fontfamily='monospace')
    ax.legend(fontsize=5.5, loc='upper right',
              labelcolor=C['text'], framealpha=0.15)

    # Step/time indicator
    bar_w  = progress
    bar_ax = fig.add_axes([0.01, 0.015, 0.98, 0.012])
    bar_ax.set_xlim(0, 1); bar_ax.set_ylim(0, 1)
    bar_ax.axis('off')
    bar_ax.add_patch(patches.Rectangle((0, 0), 1, 1,
                                        color='#111128', zorder=0))
    bar_ax.add_patch(patches.Rectangle((0, 0), bar_w, 1,
                                        color='#2244aa', zorder=1))
    fig.text(0.5, 0.004, f"Training Progress — {100*progress:.1f}%",
             ha='center', fontsize=6, color='#888', fontfamily='monospace')

    fig.canvas.draw()
    buf_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img     = buf_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# CELL 9 │ Demo Episode Recorder (captures frames of one full game)
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def record_episode(env: PongEnv,
                   agent1: PPOAgent,
                   agent2: PPOAgent,
                   step: int,
                   total_steps: int,
                   stats1: Dict,
                   stats2: Dict,
                   fps_every: int = 3) -> List[np.ndarray]:
    """
    Runs one full episode with greedy actions and captures a frame every
    `fps_every` game steps. Returns list of rendered frames.
    """
    obs1, obs2 = env.reset()
    frames = []
    done   = False

    agent1.net.eval()
    agent2.net.eval()

    t = 0
    while not done:
        # Greedy actions (sample from distribution, not argmax — still varied)
        a1, _, _ = agent1.net.act(obs1)
        a2, _, _ = agent2.net.act(obs2)
        obs1, obs2, _, _, done, _ = env.step(a1, a2)

        if t % fps_every == 0:
            frame = render_frame(env, agent1, agent2,
                                 step, total_steps, stats1, stats2)
            frames.append(frame)
        t += 1

    agent1.net.train()
    agent2.net.train()
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# CELL 10 │ Main Training Loop
# ══════════════════════════════════════════════════════════════════════════════
def train():
    cfg = CFG
    env = PongEnv(cfg)

    agent1 = PPOAgent("Agent-1", cfg)
    agent2 = PPOAgent("Agent-2", cfg)

    # Stats logs for dashboard (updated every PPO update)
    stats1 = {'policy_loss_log': [], 'entropy_log': []}
    stats2 = {'policy_loss_log': [], 'entropy_log': []}

    # Video frame accumulator
    video_frames: List[np.ndarray] = []

    # Training state
    obs1, obs2 = env.reset()
    ep_ret1, ep_ret2 = 0.0, 0.0
    ep_len          = 0

    total_steps  = cfg.total_timesteps
    step         = 0
    last_record  = -cfg.record_every_steps   # force first recording at step 0

    t0 = time.time()
    print("=" * 70)
    print("  DUAL-AGENT PPO SELF-PLAY PONG  ·  TRAINING STARTED")
    print(f"  Device: {DEVICE}   Total steps: {total_steps:,}")
    print("=" * 70)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while step < total_steps:

        # ── Collect rollout ──────────────────────────────────────────────────
        for _ in range(cfg.rollout_steps):
            if step >= total_steps:
                break

            a1, lp1, v1 = agent1.net.act(obs1)
            a2, lp2, v2 = agent2.net.act(obs2)

            n_obs1, n_obs2, r1, r2, done, _ = env.step(a1, a2)

            agent1.buffer.push(obs1, a1, lp1, r1, done, v1)
            agent2.buffer.push(obs2, a2, lp2, r2, done, v2)

            obs1, obs2 = n_obs1, n_obs2
            ep_ret1 += r1
            ep_ret2 += r2
            ep_len  += 1
            step    += 1

            agent1.total_steps += 1
            agent2.total_steps += 1

            if done:
                agent1.ep_returns.append(ep_ret1)
                agent2.ep_returns.append(ep_ret2)
                agent1.ep_lengths.append(ep_len)
                agent2.ep_lengths.append(ep_len)

                # Log wins
                if ep_ret1 > ep_ret2:
                    agent1.win_log.append(1); agent2.win_log.append(0)
                elif ep_ret2 > ep_ret1:
                    agent1.win_log.append(0); agent2.win_log.append(1)
                else:
                    agent1.win_log.append(0.5); agent2.win_log.append(0.5)

                obs1, obs2 = env.reset()
                ep_ret1 = ep_ret2 = 0.0
                ep_len  = 0

            # ── Record demo episode ──────────────────────────────────────────
            if step - last_record >= cfg.record_every_steps:
                last_record = step
                frames = record_episode(env, agent1, agent2,
                                        step, total_steps, stats1, stats2)
                video_frames.extend(frames)
                obs1, obs2 = env.reset()   # fresh state after recording
                ep_ret1 = ep_ret2 = 0.0
                ep_len  = 0

        # ── PPO Update ───────────────────────────────────────────────────────
        progress = step / total_steps
        _, last_v1 = agent1.net(
            torch.tensor(obs1, dtype=torch.float32, device=DEVICE).unsqueeze(0))
        _, last_v2 = agent2.net(
            torch.tensor(obs2, dtype=torch.float32, device=DEVICE).unsqueeze(0))

        m1 = agent1.update(last_v1.item(), progress)
        m2 = agent2.update(last_v2.item(), progress)

        if m1:
            stats1['policy_loss_log'].append(m1['policy_loss'])
            stats1['entropy_log'].append(m1['entropy'])
            stats2['policy_loss_log'].append(m2['policy_loss'])
            stats2['entropy_log'].append(m2['entropy'])

        # ── Console log ──────────────────────────────────────────────────────
        n_ep = len(agent1.ep_returns)
        if n_ep > 0 and n_ep % 50 == 0:
            elapsed = time.time() - t0
            sps     = step / elapsed
            wr1     = agent1.win_rate
            wr2     = agent2.win_rate
            ret1    = np.mean(agent1.ep_returns[-50:])
            ret2    = np.mean(agent2.ep_returns[-50:])
            kl1     = m1.get('approx_kl', 0) if m1 else 0
            print(f"  step={step:>7,} | ep={n_ep:>5,} | "
                  f"WR1={wr1:.2f} WR2={wr2:.2f} | "
                  f"ret1={ret1:+.2f} ret2={ret2:+.2f} | "
                  f"kl={kl1:.4f} | {sps:.0f} sps | "
                  f"{elapsed/60:.1f} min")

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"  TRAINING COMPLETE  ·  {elapsed/60:.1f} min  ·  {step:,} steps")
    print(f"  Agent-1 final win rate : {agent1.win_rate:.3f}")
    print(f"  Agent-2 final win rate : {agent2.win_rate:.3f}")
    print(f"  Recorded frames        : {len(video_frames)}")
    print("=" * 70)

    return agent1, agent2, env, video_frames, stats1, stats2


# ══════════════════════════════════════════════════════════════════════════════
# CELL 11 │ Video Export
# ══════════════════════════════════════════════════════════════════════════════
def export_video(frames: List[np.ndarray],
                 base_fps: int = CFG.video_fps,
                 speedup: float = CFG.video_speedup,
                 out_dir: str = PROJECT_DIR) -> str:
    """
    Encodes all captured frames into a cinematic MP4.
    Speed is multiplied by `speedup` so the viewer sees the full
    training arc in a fraction of the real time.
    """
    if not frames:
        print("⚠️  No frames to encode.")
        return ""

    out_fps  = int(base_fps * speedup)
    out_path = os.path.join(out_dir, "pong_training_cinematic.mp4")

    print(f"\n🎬 Encoding video …")
    print(f"   Frames   : {len(frames)}")
    print(f"   Base FPS : {base_fps}  ×  {speedup}× speed  →  {out_fps} FPS output")
    duration = len(frames) / out_fps
    print(f"   Duration : ~{duration:.1f} s  ({duration/60:.2f} min)")

    writer = imageio.get_writer(
        out_path,
        fps=out_fps,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p',
        output_params=['-crf', '18', '-preset', 'fast']
    )

    for i, frame in enumerate(frames):
        writer.append_data(frame)
        if (i + 1) % 200 == 0:
            print(f"   {i+1}/{len(frames)} frames written …")

    writer.close()
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"✅ Video saved → {out_path}  ({size_mb:.1f} MB)")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# CELL 12 │ Save Models
# ══════════════════════════════════════════════════════════════════════════════
def save_models(agent1: PPOAgent, agent2: PPOAgent, out_dir: str = PROJECT_DIR):
    torch.save({
        'model_state': agent1.net.state_dict(),
        'optimizer_state': agent1.opt.state_dict(),
        'win_log': agent1.win_log,
        'ep_returns': agent1.ep_returns,
    }, os.path.join(out_dir, 'agent1_final.pt'))

    torch.save({
        'model_state': agent2.net.state_dict(),
        'optimizer_state': agent2.opt.state_dict(),
        'win_log': agent2.win_log,
        'ep_returns': agent2.ep_returns,
    }, os.path.join(out_dir, 'agent2_final.pt'))
    print(f"💾 Models saved to {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 13 │ Download Helper (Colab)
# ══════════════════════════════════════════════════════════════════════════════
def colab_download(path: str):
    """Provides a clickable download link inside Colab."""
    try:
        from google.colab import files
        print(f"\n📥 Triggering browser download for: {path}")
        files.download(path)
    except ImportError:
        # Not in Colab — just show the path
        print(f"\n📁 File ready at: {os.path.abspath(path)}")
        try:
            display(FileLink(path))
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# CELL 14 │ Post-Training Stats Plot
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_summary(agent1: PPOAgent, agent2: PPOAgent,
                          out_dir: str = PROJECT_DIR):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=C['bg'])
    fig.suptitle("Training Summary — Self-Play Pong PPO",
                 color=C['text'], fontsize=13, fontweight='bold', y=1.02)

    for ax in axes:
        ax.set_facecolor(C['court'])
        ax.tick_params(colors=C['text'])
        for sp in ax.spines.values():
            sp.set_color('#1a1a3a')

    w = 50
    # Win rate
    def smooth(data, k):
        return np.convolve(data, np.ones(k)/k, mode='valid') if len(data) >= k else data

    axes[0].plot(smooth(agent1.win_log, w), color=C['p1'], label='Agent 1')
    axes[0].plot(smooth(agent2.win_log, w), color=C['p2'], label='Agent 2')
    axes[0].axhline(0.5, color='#555', ls='--', lw=0.8)
    axes[0].set_title('Win Rate', color=C['text']); axes[0].legend(labelcolor=C['text'])

    axes[1].plot(smooth(agent1.ep_returns, w), color=C['p1'], label='Agent 1')
    axes[1].plot(smooth(agent2.ep_returns, w), color=C['p2'], label='Agent 2')
    axes[1].axhline(0, color='#555', ls='--', lw=0.8)
    axes[1].set_title('Episode Return', color=C['text']); axes[1].legend(labelcolor=C['text'])

    axes[2].plot(smooth(agent1.ep_lengths, w), color=C['p1'], label='Agent 1')
    axes[2].plot(smooth(agent2.ep_lengths, w), color=C['p2'], label='Agent 2')
    axes[2].set_title('Episode Length', color=C['text']); axes[2].legend(labelcolor=C['text'])

    for ax in axes:
        ax.xaxis.label.set_color(C['text'])
        ax.yaxis.label.set_color(C['text'])

    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'training_summary.png')
    plt.savefig(plot_path, dpi=120, bbox_inches='tight', facecolor=C['bg'])
    plt.show()
    print(f"📊 Summary plot saved → {plot_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 15 │ Run Everything
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── 1. Train ──────────────────────────────────────────────────────────────
    agent1, agent2, env, video_frames, stats1, stats2 = train()

    # ── 2. Save models ────────────────────────────────────────────────────────
    save_models(agent1, agent2)

    # ── 3. Summary plot ───────────────────────────────────────────────────────
    plot_training_summary(agent1, agent2)

    # ── 4. Encode & export video ──────────────────────────────────────────────
    video_path = export_video(
        video_frames,
        base_fps=CFG.video_fps,
        speedup=CFG.video_speedup,
    )

    # ── 5. Download in Colab ──────────────────────────────────────────────────
    if video_path:
        colab_download(video_path)

    print("\n" + "═" * 70)
    print("  ALL DONE 🎉")
    print(f"  Video  : {PROJECT_DIR}/pong_training_cinematic.mp4")
    print(f"  Models : {PROJECT_DIR}/agent1_final.pt")
    print(f"           {PROJECT_DIR}/agent2_final.pt")
    print(f"  Plot   : {PROJECT_DIR}/training_summary.png")
    print("═" * 70)
