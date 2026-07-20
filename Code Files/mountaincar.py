# ===============================
# IMPORTS & SETUP
# ===============================
import os
import gc
import csv
import json
import torch
import warnings
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from collections import deque
from openai import OpenAI

# Set your API Key here
os.environ["OPENAI_API_KEY"] = "" # Ensure your actual key is set

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc.collect(); torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

# ===============================
# WRAPPERS (Strictly Baseline)
# ===============================
class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.low = env.observation_space.low
        self.high = env.observation_space.high
    def observation(self, obs):
        return (obs - self.low) / (self.high - self.low)

class StepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, action):
        state, _, terminated, truncated, info = self.env.step(action)
        pos, vel = state
        
        # Reward Shaping
        vel = np.interp(vel, [-0.07, 0.07], [-0.5, 0.5])
        rad = np.deg2rad(pos * 360) 
        
        reward = 0.2 * (np.cos(rad) + 2 * abs(vel)) - 0.5

        if pos > 0.98: reward += 20
        elif pos > 0.92: reward += 10
        elif pos > 0.82: reward += 6
        elif pos > 0.65: reward += 1 - np.exp(-2 * pos)
        
        if vel > 0.3 and pos > 0.40842572 + 0.1:
            reward += 1 + 2 * pos
            
        return state, reward, terminated, truncated, info

# ===============================
# REPLAY MEMORY
# ===============================
class ReplayMemory:
    def __init__(self, capacity):
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def store(self, s, a, ns, r, d):
        self.states.append(s)
        self.actions.append(a)
        self.next_states.append(ns)
        self.rewards.append(r)
        self.dones.append(d)

    def sample(self, batch):
        idx = np.random.choice(len(self.dones), batch, replace=False)
        return (
            torch.tensor(np.array([self.states[i] for i in idx]), dtype=torch.float32, device=device),
            torch.tensor([self.actions[i] for i in idx], dtype=torch.long, device=device),
            torch.tensor(np.array([self.next_states[i] for i in idx]), dtype=torch.float32, device=device),
            torch.tensor([self.rewards[i] for i in idx], dtype=torch.float32, device=device),
            torch.tensor([self.dones[i] for i in idx], dtype=torch.bool, device=device)
        )
    def __len__(self): return len(self.dones)

# ===============================
# DQN NETWORK
# ===============================
class DQN(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, act_dim)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    def forward(self, x): return self.net(x)

# ===============================
# LLM POLICY MODULES
# ===============================
class RuleBasedPolicy:
    def __init__(self, params):
        self.params = params
        self.vel_split = params.get("vel_split", 0.5)
        self.pos_split = params.get("pos_split", 0.4)
        self.action_low = params.get("action_low_energy", 0) 
        self.action_high = params.get("action_high_energy", 2) 
        self.conflict_count = 0
        self.total_calls = 0

    def get_action(self, state):
        self.total_calls += 1
        pos, vel = state
        
        is_low_energy = (pos < self.pos_split) and (vel < self.vel_split)
        is_high_energy = (pos >= self.pos_split) or (vel >= self.vel_split)
        
        if is_low_energy and not is_high_energy:
            return self.action_low
        elif is_high_energy and not is_low_energy:
            return self.action_high
        else:
            self.conflict_count += 1
            return np.random.choice([0, 1, 2])

    def get_conflict_stats(self):
        if self.total_calls == 0: return 0.0
        return (self.conflict_count / self.total_calls) * 100

class LLMOracle:
    def __init__(self, hp):
        self.client = OpenAI()
        self.policy_history = deque(maxlen=hp["past_policies_len"])
        
        self.current_params = {
            "vel_split": 0.5, "pos_split": 0.4, 
            "action_low_energy": 0, "action_high_energy": 2
        }
        self.current_policy = RuleBasedPolicy(self.current_params)

    def consult_gpt(self, training_trajectories, current_epsilon, episode_num, eval_score, coverage_info):
        returns = [sum([x[2] for x in t]) for t in training_trajectories]
        avg_r = np.mean(returns) if returns else 0
        conflict_pct = self.current_policy.get_conflict_stats()
        
        traj_summary = "\n".join([f"- Run {i}: R={r:.2f}, Len={len(t)}" for i, (r, t) in enumerate(zip(returns, training_trajectories))])
        
        self.policy_history.append({
            "policy": self.current_params, 
            "avg_r": avg_r,
            "epsilon": current_epsilon,
            "coverage": coverage_info
        })
        history_json = json.dumps(list(self.policy_history), indent=2)

        system_prompt = f"""
        You are controlling a car in 'Mountain Car'. 
        Goal: Reach the flag at position > 0.5.
        Actions: 0 (Left), 1 (Nothing), 2 (Right).
        
        Physics:
        - Gravity pulls down.
        - Engine is weak; you must build momentum by rocking back and forth.
        
        Task: 
        Define a "Non Conflicting Policy" that splits state space (Velocity, Position).
        - Low Energy State: Needs momentum building.
        - High Energy State: Drive up the hill.

        Coverage Statistics:

        Low Energy Region:
        {coverage_info["low_energy"]} %

        High Energy Region:
        {coverage_info["high_energy"]} %

        Use BOTH evaluation score and coverage statistics.
        """

        user_prompt = f"""
        Status (Ep {episode_num}):
        Eval Score (Greedy): {eval_score:.2f}
        Train Trajectories (Exploration): 
        {traj_summary}
        
        Conflict Rate: {conflict_pct:.1f}%
        
        History of Policies:
        {history_json}
        
        Provide updated splits in JSON:
        {{
            "vel_split": float,
            "pos_split": float,
            "action_low_energy": int,
            "action_high_energy": int,
            "reasoning": "string"
        }}
        """

        try:
            print(f"   [LLM] Consulting GPT (Eps: {current_epsilon:.2f})...")
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                response_format={"type": "json_object"},
                seed=42
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            
            self.current_params = data
            self.current_policy = RuleBasedPolicy(data)
            
            reasoning = data.get("reasoning", "")
            print(f"   [LLM] New Splits: Vel={data.get('vel_split')}, Pos={data.get('pos_split')}")
            
            return True, data, content, user_prompt, reasoning
            
        except Exception as e:
            print(f"   [LLM Error] {e}")
            return False, {}, str(e), user_prompt, ""

# ===============================
# AGENT
# ===============================
class Agent:
    def __init__(self, env, hp, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        self.gamma = hp["discount"]
        self.eps = hp["epsilon_max"]
        self.eps_min = hp["epsilon_min"]
        self.anneal = hp["epsilon_anneal_episodes"]
        self.episode = 0
        self.clip = hp["clip_grad_norm"] 

        self.mem = ReplayMemory(hp["memory_capacity"])
        self.q = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
        self.qt = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
        self.qt.load_state_dict(self.q.state_dict()); self.qt.eval()
        self.opt = optim.Adam(self.q.parameters(), lr=hp["learning_rate"])
        self.loss_fn = nn.MSELoss()
        
        self.llm_oracle = LLMOracle(hp)
        
        self.losses = []
        self._loss_acc = 0.0
        self._loss_count = 0

    def act(self, s, use_llm=True):
        if use_llm and np.random.rand() < self.eps:
            return self.llm_oracle.current_policy.get_action(s)
        
        with torch.no_grad():
            return torch.argmax(self.q(torch.tensor(s, dtype=torch.float32, device=device))).item()

    def learn(self, batch_size, done):
        if len(self.mem) < batch_size: return
        s, a, ns, r, d = self.mem.sample(batch_size)
        
        q = self.q(s).gather(1, a.unsqueeze(1))
        with torch.no_grad():
            q_next = self.qt(ns).max(1, keepdim=True)[0]
            q_next[d.unsqueeze(1)] = 0.0 
        
        loss = self.loss_fn(q, r.unsqueeze(1) + self.gamma * q_next)
        
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.clip) 
        self.opt.step()

        self._loss_acc += loss.item()
        self._loss_count += 1

        if done and self._loss_count > 0:
            self.losses.append(self._loss_acc / self._loss_count)
            self._loss_acc = 0.0
            self._loss_count = 0

    def update_target(self): self.qt.load_state_dict(self.q.state_dict())
    
    def update_epsilon(self):
        self.episode += 1
        frac = min(self.episode / self.anneal, 1.0)
        start = 1.0 
        end = 0.05  
        self.eps = start - frac * (start - end)

# ===============================
# EVALUATION
# ===============================
def evaluate_policy(agent, env_fn, n, seed):
    env = env_fn()
    scores = []
    with torch.no_grad():
        for i in range(n):
            s, _ = env.reset(seed=seed + 1000 + i)
            done = trunc = False
            ep_r = 0
            while not done and not trunc:
                a = torch.argmax(
                    agent.q(torch.tensor(s, dtype=torch.float32, device=device))
                ).item()
                s, r, done, trunc, _ = env.step(a)
                ep_r += r
            scores.append(ep_r)
    env.close()
    return np.mean(scores)
# ==========================
# Coverage Evaluation
# ==========================

def evaluate_policy_with_coverage(agent, env_fn, n, seed):

    env = env_fn()

    scores = []

    low_count = 0
    high_count = 0
    total = 0

    with torch.no_grad():

        for i in range(n):

            s, _ = env.reset(seed=seed + 1000 + i)

            done = trunc = False
            ep_r = 0

            while not done and not trunc:

                pos, vel = s

                is_low = (
                    pos < agent.llm_oracle.current_policy.pos_split
                    and
                    vel < agent.llm_oracle.current_policy.vel_split
                )

                if is_low:
                    low_count += 1
                else:
                    high_count += 1

                total += 1

                a = torch.argmax(
                    agent.q(
                        torch.tensor(
                            s,
                            dtype=torch.float32,
                            device=device
                        )
                    )
                ).item()

                s, r, done, trunc, _ = env.step(a)

                ep_r += r

            scores.append(ep_r)

    env.close()

    coverage_info = {
        "low_energy":
            round(100 * low_count / max(total,1),2),
        "high_energy":
            round(100 * high_count / max(total,1),2)
    }

    return np.mean(scores), coverage_info
# ===============================
# TRAINER
# ===============================
class Trainer:
    def __init__(self, hp, seed):
        self.hp = hp
        self.seed = seed
        self.env = self._make_env(seed)
        self.agent = Agent(self.env, hp, seed)
        
        self.save_path = f"dqn_mountaincar_seed_{seed}.pth"
        self.llm_log_path = f"llm_logs_mountaincar_seed_{seed}.jsonl"

        def make_env(seed_offset=10_000):
            def _make():
                e = gym.make("MountainCar-v0", max_episode_steps=hp["max_steps"])
                e.reset(seed=seed + seed_offset)
                e.action_space.seed(seed + seed_offset)
                return StepWrapper(ObservationWrapper(e))
            return _make
        
        self.eval_env_fn = make_env()
        self.training_traj_buffer = deque(maxlen=hp["n_trajectories_llm"])

    def _make_env(self, seed):
        e = gym.make("MountainCar-v0", max_episode_steps=self.hp["max_steps"])
        e = ObservationWrapper(e)
        e = StepWrapper(e)
        e.reset(seed=seed)
        return e

    def train(self):
        rows = []
        threshold_ep = None
        total_steps = 0
        
        # Initialize JSONL file
        with open(self.llm_log_path, "w") as f:
            f.write("")

        for ep in range(self.hp["max_episodes"]):
            s, _ = self.env.reset()
            done = False; trunc = False
            ep_r = 0
            current_traj = [] 
            
            while not done and not trunc:
                a = self.agent.act(s)
                ns, r, done, trunc, _ = self.env.step(a)
                
                self.agent.mem.store(s, a, ns, r, done or trunc)
                current_traj.append((s, a, r))

                if len(self.agent.mem) > self.hp["batch_size"]:
                    self.agent.learn(self.hp["batch_size"], done or trunc)
                    if total_steps % self.hp["update_frequency"] == 0:
                        self.agent.update_target()
                
                s = ns; ep_r += r; total_steps += 1

            self.agent.update_epsilon()
            self.training_traj_buffer.append(current_traj)

            # Evaluate
            eval_score, coverage_info = evaluate_policy_with_coverage(self.agent, self.eval_env_fn, self.hp["eval_episodes"], self.seed)
            loss = self.agent.losses[-1] if self.agent.losses else np.nan
            
            # LLM Update
            llm_response_text = "N/A"
            if ep > 0 and (ep + 1) % self.hp["llm_call_frequency"] == 0:
                success, data, raw_text, user_prompt, reasoning = self.agent.llm_oracle.consult_gpt(
                    list(self.training_traj_buffer), 
                    self.agent.eps,
                    ep + 1,
                    eval_score,
                    coverage_info
                )
                
                if success:
                    llm_response_text = raw_text
                    log_entry = {
                        "episode": ep + 1,
                        "user_prompt": user_prompt,
                        "raw_json_response": raw_text,
                        "parsed_policy": data,
                        "reasoning": reasoning,
                        "current_eval_score": eval_score,
                        "current_epsilon": self.agent.eps
                    }
                    with open(self.llm_log_path, "a") as f:
                        f.write(json.dumps(log_entry, default=float) + "\n")

            # Logging
            rows.append([ep + 1, ep_r, eval_score, self.agent.eps, loss, llm_response_text])
            print(f"Seed {self.seed} | Ep {ep+1:3d} | TrainR: {ep_r:.2f} | EvalR: {eval_score:.2f} | Eps: {self.agent.eps:.2f}")

            # Saving mechanism (Adapted from CliffWalker)
            if eval_score >= self.hp["early_stop_threshold"] and threshold_ep is None:
                threshold_ep = ep + 1
                torch.save({
                    "episode": threshold_ep,
                    "model_state_dict": self.agent.q.state_dict(),
                    "eval_return": eval_score,
                    "seed": self.seed,
                    "hyperparameters": self.hp,
                }, self.save_path)
                print(f">>> EARLY STOPPING at episode {threshold_ep}")
                break

        if threshold_ep is None:
            torch.save({
                "episode": self.hp["max_episodes"],
                "model_state_dict": self.agent.q.state_dict(),
                "eval_return": eval_score,
                "seed": self.seed,
                "hyperparameters": self.hp,
            }, self.save_path)
            print(f"Training finished | Model saved to {self.save_path}")

        return rows, threshold_ep

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    HP = {
        "learning_rate": 7.5e-4,     
        "discount": 0.96,            
        "batch_size": 64,
        "update_frequency": 20,      
        "max_episodes": 250,
        "max_steps": 200,
        "epsilon_max": 1.0,
        "epsilon_min": 0.05,
        "epsilon_anneal_episodes": 200,
        "memory_capacity": 125_000,  
        "clip_grad_norm": 5.0,       
        "eval_episodes": 10,
        "early_stop_threshold": 150.0,
        
        # --- NEW LLM Hyperparameters (From CliffWalker) ---
        "llm_call_frequency": 5,      
        "past_policies_len": 10,       
        "n_trajectories_llm": 3       
    }

    N_RUNS = 10
    BASE_SEED = 1
    crossed = []
    
    for i in range(N_RUNS):
        seed = BASE_SEED + i
        print(f"\n==== RUN {i+1}/{N_RUNS} | SEED {seed} ====")
        trainer = Trainer(HP, seed)
        rows, cross_ep = trainer.train()

        with open(f"run_mountaincar_seed_{seed}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "train_return", "eval_return", "epsilon", "loss", "llm_full_response"])
            writer.writerows(rows)

        if cross_ep is not None:
            crossed.append(cross_ep)

    with open("summary_mountaincar_final.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mean_cross_episode", np.mean(crossed) if crossed else "N/A"])
        writer.writerow(["std_cross_episode", np.std(crossed) if crossed else "N/A"])

    print("\nAll runs complete. Logs saved.")
