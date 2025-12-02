"""
Soft Actor-Critic algorithm for the DeepMind Control Suite cartpole/swingup problem.

by Julia Beiferman
"""

import argparse
import os
import random
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dm_control import suite


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class DMCCartpoleSwingupEnv:
    def __init__(self, seed: int):
        self.env = suite.load(
            domain_name="cartpole",
            task_name="swingup",
            task_kwargs={"random": seed},
        )
        self.action_spec = self.env.action_spec()
        self.obs_spec = self.env.observation_spec()

    @property
    def obs_dim(self):
        #return sum(np.asarray(v).size for v in self.obs_spec.values())
        return sum(int(np.prod(v.shape)) for v in self.obs_spec.values())

    @property
    def act_dim(self):
        return int(np.prod(self.action_spec.shape))

    @property
    def act_low(self):
        return float(self.action_spec.minimum[0])

    @property
    def act_high(self):
        return float(self.action_spec.maximum[0])

    def _flatten_obs(self, timestep):
        obs_list = []
        for key in sorted(timestep.observation.keys()):
            obs_list.append(np.asarray(timestep.observation[key]).ravel())
        return np.concatenate(obs_list).astype(np.float32)

    def reset(self):
        ts = self.env.reset()
        return self._flatten_obs(ts)

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_spec.minimum, self.action_spec.maximum)
        ts = self.env.step(action)
        obs = self._flatten_obs(ts)
        reward = float(ts.reward if ts.reward is not None else 0.)
        done = bool(ts.last())
        return obs, reward, done

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            act=self.acts_buf[idxs],
            rew=self.rews_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    #gaussian policy with tanh squash
    def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation=nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

        #set limits
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.act_limit = act_limit

    def forward(self, obs):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        normal = torch.distributions.Normal(mu, std)

        x = normal.rsample()
        y = torch.tanh(x)
        action = self.act_limit * y

        # log prob
        log_prob = normal.log_prob(x).sum(dim=-1)
        log_prob -= torch.sum(torch.log(1 - y.pow(2) + 1e-6), dim=-1)
        mu_action = self.act_limit * torch.tanh(mu)
        
        return action, log_prob, mu_action


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation=nn.ReLU)

    def forward(self, obs, act):
        
        return self.q(torch.cat([obs, act], dim=-1)).squeeze(-1)



@dataclass
class SACConfig:
    # set our hyperparameters 
    total_steps: int = 100000
    batch_size: int = 256
    buffer_size: int = 200_000
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    start_steps: int = 2000
    update_every: int = 1
    alpha: float = 0.2       # temp     
    hidden_sizes: tuple = (256, 256)
    eval_interval: int = 5000


# evaluate the policy 
def evaluate_policy(actor, env, device, act_limit, episodes=10):
    returns = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = actor.forward(obs_t)
                action = act_limit * torch.tanh(mu)
            action = action.detach().cpu().numpy()[0]
            obs, rew, done = env.step(action)
            ep_ret += rew
        returns.append(ep_ret)
    return float(np.mean(returns))


def train(args):
    set_seed(args.seed)
    cfg = SACConfig()

    # save the seed file 
    run_name = f"sac_cartpole_swingup_seed{args.seed}"
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cpu")
    print(f"Using device: {device}, Seed: {args.seed}")

    env = DMCCartpoleSwingupEnv(seed=args.seed)           # training env
    eval_env = DMCCartpoleSwingupEnv(seed=args.eval_seed) # evaluation env

    obs_dim = env.obs_dim
    act_dim = env.act_dim
    act_limit = env.act_high

    # define the actor and the q values and targets 
    actor = SquashedGaussianActor(obs_dim, act_dim, cfg.hidden_sizes, act_limit).to(device)
    q1 = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
    q2 = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
    q1_targ = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
    q2_targ = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(device)
    q1_targ.load_state_dict(q1.state_dict())
    q2_targ.load_state_dict(q2.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=cfg.lr)
    q1_opt = optim.Adam(q1.parameters(), lr=cfg.lr)
    q2_opt = optim.Adam(q2.parameters(), lr=cfg.lr)

    # Replay buffer
    replay = ReplayBuffer(obs_dim, act_dim, cfg.buffer_size)
    train_returns, train_steps = [], []
    eval_returns, eval_steps = [], []

    obs = env.reset()
    ep_ret, ep_len = 0, 0

    start_time = time.time()

    for step in range(1, cfg.total_steps + 1):
        if step < cfg.start_steps:
            act = np.random.uniform(env.act_low, env.act_high, size=act_dim).astype(np.float32)
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            a, _, _ = actor.sample(obs_t)
            act = a.detach().cpu().numpy()[0]

        next_obs, rew, done = env.step(act)
        replay.store(obs, act, rew, next_obs, float(done))

        obs = next_obs
        ep_ret += rew
        ep_len += 1

        if done:
            train_returns.append(ep_ret)
            train_steps.append(step)
            obs = env.reset()
            ep_ret, ep_len = 0, 0

        # SAC updates
        if step >= cfg.start_steps and step % cfg.update_every == 0:
            batch = replay.sample(cfg.batch_size)
            o = batch["obs"].to(device)
            a = batch["act"].to(device)
            r = batch["rew"].to(device)
            o2 = batch["next_obs"].to(device)
            d = batch["done"].to(device)

            # target actions
            with torch.no_grad():
                a2, logp_a2, _ = actor.sample(o2)
                q1_targ_v = q1_targ(o2, a2)
                q2_targ_v = q2_targ(o2, a2)
                q_targ_v = torch.min(q1_targ_v, q2_targ_v) - cfg.alpha * logp_a2
                backup = r + cfg.gamma * (1 - d) * q_targ_v

            # test against these q losses 
            q1_loss = ((q1(o, a) - backup) ** 2).mean()
            q2_loss = ((q2(o, a) - backup) ** 2).mean()

            q1_opt.zero_grad()
            q1_loss.backward()
            q1_opt.step()

            q2_opt.zero_grad()
            q2_loss.backward()
            q2_opt.step()

            # define the loss q values for the actor 
            a_pi, logp_pi, _ = actor.sample(o)
            q1_pi = q1(o, a_pi)
            q2_pi = q2(o, a_pi)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (cfg.alpha * logp_pi - q_pi).mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            # update the values 
            with torch.no_grad():
                for p, p_targ in zip(q1.parameters(), q1_targ.parameters()):
                    p_targ.data.mul_(1 - cfg.tau)
                    p_targ.data.add_(cfg.tau * p.data)
                for p, p_targ in zip(q2.parameters(), q2_targ.parameters()):
                    p_targ.data.mul_(1 - cfg.tau)
                    p_targ.data.add_(cfg.tau * p.data)

        # evaluate the parametrs 
        if step % cfg.eval_interval == 0:
            eval_ret = evaluate_policy(actor, eval_env, device, act_limit, episodes=args.eval_episodes)
            eval_returns.append(eval_ret)
            eval_steps.append(step)
            print(
                f"Step {step} | TrainEpRet: {train_returns[-1] if train_returns else 0:.1f} | "
                f"EvalRet: {eval_ret:.1f} | Time: {time.time()-start_time:.1f}s"
            )

    # record the data 
    np.savez(
        os.path.join(save_dir, "logs.npz"),
        train_returns=np.array(train_returns),
        train_steps=np.array(train_steps),
        eval_returns=np.array(eval_returns),
        eval_steps=np.array(eval_steps),
    )

    torch.save(actor.state_dict(), os.path.join(save_dir, "actor.pt"))

    print(f"Training complete. Logs saved to: {save_dir}/logs.npz")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
