# ===============================
# IMPORTS & GLOBAL SETUP
# ===============================
import os
import gc
import csv
import json
import re
import ast
import torch
import warnings
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from collections import deque

# === IMPORT OPENAI ===
# pip install openai
import openai 

device = torch.device("cpu")  # CPU only (reproducible)

gc.collect()
warnings.filterwarnings("ignore", category=UserWarning)

# ===============================
# UTILS
# ===============================
def one_hot(state, n_states):
    vec = np.zeros(n_states, dtype=np.float32)
    vec[state] = 1.0
    return vec

# ===============================
# LLM ADVISOR
# ===============================
class LLMPolicyAdvisor:
    def __init__(self):
        # =====================================================
        # TODO: PASTE YOUR OPENAI API KEY BELOW
        # =====================================================
        self.client = openai.OpenAI(api_key="") 
        
        self.action_map = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

    # def format_trajectory(self, traj):
    #     """Converts a list of (s, a, r, ns) into a readable string."""
    #     path_str = []
    #     for step in traj:
    #         s, a, r, ns = step
    #         path_str.append(f"(State {s} -> {self.action_map[a]} -> State {ns} -> Reward {r})")
    #     return " -> ".join(path_str)

    def format_trajectory(self, traj):
        n = len(traj)

        if n <= 15:
            sampled = traj
            sections = [("Trajectory", sampled)]
        else:
            first = traj[:5]

            mid_start = max((n // 2) - 2, 5)
            middle = traj[mid_start:mid_start + 5]

            last = traj[-5:]

            sections = [
                ("Beginning", first),
                ("Middle", middle),
                ("End", last),
            ]

        lines = []

        for title, part in sections:
            lines.append(f"--- {title} ---")
            for s, a, r, ns in part:
                lines.append(
                    f"State {s} --{self.action_map[a]}--> State {ns}, Reward={r}"
                )

        return "\n".join(lines)

    def get_policy(self, trajectories, returns, current_epsilon, avg_eval_reward, last_eval_reward, coverage_info, seed_val, past_policies):
        """
        Constructs the prompt exactly as requested.
        """
        
        # 1. Prepare History JSON
        history_data = []
        for i, (traj, ret) in enumerate(zip(trajectories, returns)):
            history_data.append({
                "run": i + 1,
                "return": ret,
                "trajectory": self.format_trajectory(traj)
            })
        history_json = json.dumps(history_data, indent=2)

        # 2. Prepare Past Policies JSON
        past_policies_json = json.dumps(past_policies, indent=2) if past_policies else "None (First Iteration)"

        # 3. SYSTEM PROMPT
        system_prompt = """
        You are an Expert explorer helping an agent with exploration on FrozenLake-v1 (4x4 Grid).
        The FrozenLake environment is a 2D 4x4 grid world (4 rows, 4 columns) where the agent can take an action 0,1,2 or 3. 
        The state is between 0 and 15, representing the which grid the agent is in. The top row is 0 to 3, 2nd row is 4 to 7, 3rd row is 8 to 11, bottom row is 12 to 15. 
        THe environment is a 4x4 grid as followes:
        S F F F
        F H F H
        F F F H
        H F F G
        S is the starting point and has state 0.
        F is frozen surface.
        H is a hole, if the agent steps on it, the episode ends with 0 reward.
        G is the goal, if the agent reaches it, the episode ends with +1 reward.
        all the borders are walls, if the agent tries to move outside the grid, it stays in the same state.

        The environment is slippery, which means when the agent takes an action, it will go to 3 directions with equal probability.
        -  0, it will pick from [move down, move left, move up] with equal probability.
        -  1, it will pick from [move right, move down, move left] with equal probability.
        -  2, it will pick from [move up, move right, move down] with equal probability.
        -  3, it will pick from [move left, move up, move right] with equal probability.
        IMPORTANT ACTION MAPPING:

        0 = LEFT
        1 = DOWN
        2 = RIGHT
        3 = UP

        You MUST use this mapping when generating the policy.

        Task: Propose policy that takes safe actions to maximize average returns.
        """

        # 4. USER PROMPT (UPDATED WITH PAST POLICIES)
        user_prompt = f"""
        Current Training Status:
        Current Average Evaluation Return: {avg_eval_reward:.2f}
        Previous Average Evaluation Return:{last_eval_reward:.2f}

        Use the change between the current and previous average evaluation returns to determine whether the current exploration policy is improving or degrading. If performance has improved, preserve the current policy unless the trajectories consistently indicate failures. If performance has degraded, identify the states most responsible for repeated failures and modify only those actions.
        Coverage Statistics:

        LEFT Region:
        {coverage_info["left_pct"]:.2f}%

        DOWN Region:
        {coverage_info["down_pct"]:.2f}%

        RIGHT Region:
        {coverage_info["right_pct"]:.2f}%

        UP Region:
        {coverage_info["up_pct"]:.2f}%

        Epsilon: {current_epsilon:.2f}
        
        History of Previous Policies & Their Eval Returns: {past_policies_json}

        Current Greedy Evaluation Trajectories (10):

        The trajectories are the primary source of evidence. Base your decisions primarily on the observed successful and unsuccessful trajectories rather than manually designing a path through the grid.

        Each trajectory contains the sequence of visited states, actions taken by the greedy DQN policy, and the rewards obtained. These trajectories represent the current behaviour of the agent.
        {history_json}
        
        Use the greedy evaluation trajectories, average evaluation return, coverage statistics, and previous policy performance to identify repeated failure patterns and successful behaviours.

        Task:

            Analyze the greedy evaluation trajectories, coverage statistics, and previous policy performance.

            The trajectories are the primary source of evidence. Base your recommendations primarily on repeated trajectory patterns rather than manually designing a path through the grid.

            Preserve actions that consistently appear in successful trajectories.

            If a state appears in both successful and unsuccessful trajectories, preserve its current action unless multiple trajectories consistently indicate that changing it improves performance.

            Do not modify an action unless repeated trajectory evidence suggests it contributes to failure.

            The objective is to increase the average evaluation return over successive LLM updates. If the current policy performs better than the previous one, preserve it with minimal modifications. If performance deteriorates, modify only the actions associated with repeated failures observed in the trajectories.
                    Output JSON only:
        {{
            "policy":List[16 integers between 0 and 3 mapping states to actions],
            "reasoning": "string"
        }}
        """
#
        # 5. Call LLM
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                seed=seed_val,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return content, user_prompt # Return user prompt for logging

        except Exception as e:
            return f"{{\"error\": \"{str(e)}\"}}", user_prompt

    def parse_response(self, text):
        """Parses the JSON response to extract policy and reasoning."""
        try:
            data = json.loads(text)
            policy = data.get("policy", [])
            reasoning = data.get("reasoning", "")
            
            if isinstance(policy, list) and len(policy) == 16 and all(isinstance(x, (int, float)) for x in policy):
                policy = [int(x) for x in policy]
                return policy, reasoning
            else:
                return None, f"Invalid policy format: {policy}"
        except json.JSONDecodeError:
            return None, "JSON Decode Error"
        except Exception as e:
            return None, str(e)

# ===============================
# REPLAY MEMORY (UNCHANGED)
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
            torch.tensor(np.array([self.states[i] for i in idx]), dtype=torch.float32),
            torch.tensor([self.actions[i] for i in idx], dtype=torch.long),
            torch.tensor(np.array([self.next_states[i] for i in idx]), dtype=torch.float32),
            torch.tensor([self.rewards[i] for i in idx], dtype=torch.float32),
            torch.tensor([self.dones[i] for i in idx], dtype=torch.bool),
        )

    def __len__(self):
        return len(self.dones)

# ===============================
# DQN NETWORK (UNCHANGED)
# ===============================
class DQN(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def forward(self, x):
        return self.net(x)

# ===============================
# AGENT (UNCHANGED)
# ===============================
class Agent:
    def __init__(self, obs_dim, act_dim, hp, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.gamma = hp["discount"]
        self.eps_max = hp["epsilon_max"]
        self.eps_min = hp["epsilon_min"]
        self.anneal_episodes = hp["epsilon_anneal_episodes"]

        self.eps = self.eps_max
        self.episode = 0

        self.mem = ReplayMemory(hp["memory_capacity"])

        self.q = DQN(obs_dim, act_dim)
        self.qt = DQN(obs_dim, act_dim)
        self.qt.load_state_dict(self.q.state_dict())
        self.qt.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=hp["learning_rate"])
        self.loss_fn = nn.MSELoss()
        self.clip = hp["clip_grad_norm"]

        self.losses = []
        self._loss_acc = 0.0
        self._loss_count = 0
        
        self.suggested_policy = None

    def act(self, s, action_space):
        # 1. Exploration (Epsilon)
        if np.random.rand() < self.eps:
            if self.suggested_policy is not None:
                state_idx = np.argmax(s) 
                return self.suggested_policy[state_idx]
            else:
                return action_space.sample()

        # 2. Exploitation (Greedy DQN)
        with torch.no_grad():
            return torch.argmax(self.q(torch.tensor(s))).item()

    def learn(self, batch, done):
        s, a, ns, r, d = self.mem.sample(batch)

        qsa = self.q(s).gather(1, a.unsqueeze(1))
        with torch.no_grad():
            q_next = self.qt(ns).max(1, keepdim=True)[0]
            q_next[d.unsqueeze(1)] = 0

        loss = self.loss_fn(qsa, r.unsqueeze(1) + self.gamma * q_next)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.clip)
        self.opt.step()

        self._loss_acc += loss.item()
        self._loss_count += 1

        if done:
            self.losses.append(self._loss_acc / self._loss_count)
            self._loss_acc = 0.0
            self._loss_count = 0

    def update_target(self):
        self.qt.load_state_dict(self.q.state_dict())

    def update_epsilon(self):
        self.episode += 1
        frac = min(self.episode / self.anneal_episodes, 1.0)
        self.eps = self.eps_max - frac * (self.eps_max - self.eps_min)

# ===============================
# EVALUATION & TRAJECTORY UTILS
# ===============================
def evaluate_policy_with_coverage(agent, env_fn, n, seed, n_states):
    env = env_fn()

    scores = []
    left_count = 0
    down_count = 0
    right_count = 0
    up_count = 0
    total_count = 0

    with torch.no_grad():
        for i in range(n):

            s_raw, _ = env.reset(seed=seed + 1000 + i)
            s = one_hot(s_raw, n_states)

            done = trunc = False
            ep_r = 0

            while not done and not trunc:

                if agent.suggested_policy is not None:

                    region_action = agent.suggested_policy[s_raw]

                    if region_action == 0:
                        left_count += 1

                    elif region_action == 1:
                        down_count += 1

                    elif region_action == 2:
                        right_count += 1

                    elif region_action == 3:
                        up_count += 1

                    total_count += 1

                a = torch.argmax(agent.q(torch.tensor(s))).item()

                

                ns_raw, r, done, trunc, _ = env.step(a)

                s_raw = ns_raw
                s = one_hot(ns_raw, n_states)

                ep_r += r

            scores.append(ep_r)

    env.close()

    if total_count > 0:

        coverage_info = {
            "left_pct": 100 * left_count / total_count,
            "down_pct": 100 * down_count / total_count,
            "right_pct": 100 * right_count / total_count,
            "up_pct": 100 * up_count / total_count
        }

    else:

        coverage_info = {
            "left_pct": 0.0,
            "down_pct": 0.0,
            "right_pct": 0.0,
            "up_pct": 0.0
        }

    return np.mean(scores), coverage_info

def evaluate_policy(agent, env_fn, n, seed, n_states):
    env = env_fn()
    scores = []
    with torch.no_grad():
        for i in range(n):
            s, _ = env.reset(seed=seed + 1000 + i)
            s = one_hot(s, n_states)
            done = trunc = False
            ep_r = 0
            while not done and not trunc:
                a = torch.argmax(agent.q(torch.tensor(s))).item()
                ns, r, done, trunc, _ = env.step(a)
                s = one_hot(ns, n_states)
                ep_r += r
            scores.append(ep_r)
    env.close()
    return np.mean(scores)

def collect_greedy_trajectories(agent, env_fn, n_trajs, seed, n_states):
    env = env_fn()
    trajectories = []
    returns = []
    left_count = 0
    down_count = 0
    right_count = 0
    up_count = 0
    total_count = 0
    with torch.no_grad():
        for i in range(n_trajs):
            traj = []
            s_raw, _ = env.reset(seed=seed + 2000 + i)
            s_vec = one_hot(s_raw, n_states)
            done = trunc = False
            ep_r = 0
            while not done and not trunc:
                if agent.suggested_policy is not None:
                    region_action = agent.suggested_policy[s_raw]

                    if region_action == 0:
                        left_count += 1
                    elif region_action == 1:
                        down_count += 1
                    elif region_action == 2:
                        right_count += 1
                    elif region_action == 3:
                        up_count += 1

                    total_count += 1
                a = torch.argmax(agent.q(torch.tensor(s_vec))).item()
                ns_raw, r, done, trunc, _ = env.step(a)
                ns_vec = one_hot(ns_raw, n_states)
                traj.append((s_raw, a, r, ns_raw))
                s_raw = ns_raw
                s_vec = ns_vec
                ep_r += r
            trajectories.append(traj)
            returns.append(ep_r)
    env.close()
    if total_count > 0:
        coverage_info = {
            "left_pct": 100 * left_count / total_count,
            "down_pct": 100 * down_count / total_count,
            "right_pct": 100 * right_count / total_count,
            "up_pct": 100 * up_count / total_count,
        }
    else:
        coverage_info = {
            "left_pct": 0.0,
            "down_pct": 0.0,
            "right_pct": 0.0,
            "up_pct": 0.0,
        }

    avg_return = np.mean(returns)

    return trajectories, returns, avg_return, coverage_info

# ===============================
# TRAINER (UPDATED)
# ===============================
class Trainer:
    def __init__(self, hp, seed):
        env = gym.make("FrozenLake-v1", is_slippery=True)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        self.save_path = f"dqn_frozenlake_seed_{seed}.pth"
        self.llm_log_path = f"llm_logs_seed_{seed}.jsonl"
        self.env = env
        self.n_states = env.observation_space.n
        self.seed = seed
        self.last_eval_reward = 0.0

        self.agent = Agent(
            obs_dim=self.n_states,
            act_dim=env.action_space.n,
            hp=hp,
            seed=seed
        )
        self.hp = hp
        
        # New: Track past policies and their results using HP for length
        self.past_policies = deque(maxlen=hp["past_policies_len"]) 

        def make_env():
            def _make():
                e = gym.make("FrozenLake-v1", is_slippery=True)
                return e
            return _make
        self.eval_env_fn = make_env()
        
        self.llm_advisor = LLMPolicyAdvisor()

    def train(self):
        rows = []
        threshold_ep = None
        total_steps = 0
        
        # Initialize JSONL log
        with open(self.llm_log_path, "w") as f:
            f.write("")

        for ep in range(self.hp["max_episodes"]):
            # 1. Standard Training Interaction
            s, _ = self.env.reset()
            s = one_hot(s, self.n_states)
            done = trunc = False
            ep_r = 0

            while not done and not trunc:
                a = self.agent.act(s, self.env.action_space)
                ns, r, done, trunc, _ = self.env.step(a)
                ns = one_hot(ns, self.n_states)

                self.agent.mem.store(s, a, ns, r, done)

                if len(self.agent.mem) > self.hp["batch_size"]:
                    self.agent.learn(self.hp["batch_size"], done or trunc)
                    if total_steps % self.hp["update_frequency"] == 0:
                        self.agent.update_target()

                s = ns
                ep_r += r
                total_steps += 1

            self.agent.update_epsilon()

            # 2. Evaluation
            # eval_r, coverage_info = evaluate_policy_with_coverage(
            #     self.agent,
            #     self.eval_env_fn,
            #     self.hp["eval_episodes"],
            #     self.seed,
            #     self.n_states
            # )
            loss = self.agent.losses[-1] if self.agent.losses else np.nan
            # Evaluate every episode using 10 greedy trajectories
            trajs, rets, eval_r, coverage_info = collect_greedy_trajectories(
                self.agent,
                self.eval_env_fn,
                n_trajs=self.hp["n_trajectories_llm"],
                seed=self.seed,
                n_states=self.n_states
            )

            llm_response_text = "N/A"
            # 3. LLM Interaction (Using HP for Frequency)
            llm_response_text = "N/A"
            # eval_r = np.nan
            # coverage_info = None
            if (ep + 1) % self.hp["llm_call_frequency"] == 0:
                print(f"   [LLM] Calling GPT-4o for Episode {ep+1}...")
                
                # UPDATE HISTORY: 
                # If we had a suggested policy running during these last few episodes,
                # record its performance (eval_r) now.
                # trajs, rets, eval_r, coverage_info = collect_greedy_trajectories(
                #     self.agent,
                #     self.eval_env_fn,
                #     n_trajs=self.hp["n_trajectories_llm"],
                #     seed=self.seed,
                #     n_states=self.n_states
                # )
                if self.agent.suggested_policy is not None:
                    self.past_policies.append({
                        "proposed_policy": self.agent.suggested_policy,
                        "result_eval_return": eval_r
                    })
                
                # Collect trajectories (Using HP for Count)
               
                
                # Call LLM with current Epsilon, Avg Eval Reward AND Past Policies
                llm_response_text, full_user_prompt = self.llm_advisor.get_policy(
                    trajectories=trajs, 
                    returns=rets, 
                    current_epsilon=self.agent.eps,
                    avg_eval_reward=eval_r,
                    last_eval_reward=self.last_eval_reward,
                    coverage_info=coverage_info,
                    seed_val=self.seed,
                    past_policies=list(self.past_policies) # Pass history
                )
                self.last_eval_reward = eval_r
                print("\n===== RAW GPT RESPONSE =====")
                print(llm_response_text)
                print("============================\n")
                parsed_policy, reasoning = self.llm_advisor.parse_response(llm_response_text)
                
                if parsed_policy:
                    self.agent.suggested_policy = parsed_policy
                    print(f"   [LLM] Applied Policy: {parsed_policy}")
                    print(f"   [LLM] Reasoning snippet: {reasoning[:60]}...")
                else:
                    print(f"   [LLM] Failed to parse policy. Error: {reasoning}")

                # Log detailed info to JSONL
                log_entry = {
                    "episode": ep + 1,
                    "user_prompt": full_user_prompt,
                    "raw_json_response": llm_response_text,
                    "parsed_policy": parsed_policy,
                    "reasoning": reasoning,
                    "trajectories_returns": rets,
                    "current_eval_score": eval_r,
                    "current_epsilon": self.agent.eps
                }
                with open(self.llm_log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            # 4. Standard Logging (CSV)
            rows.append([ep + 1, ep_r, eval_r, self.agent.eps, loss, llm_response_text])
            print(
                f"Seed {self.seed} | "
                f"Ep {ep+1:3d} | "
                f"TrainR {ep_r:.2f} | "
                f"EvalR {eval_r:.2f} | "
                f"Eps {self.agent.eps:.3f}"
            )

            if eval_r >= self.hp["early_stop_threshold"] and threshold_ep is None:
                threshold_ep = ep + 1
                torch.save({
                    "episode": threshold_ep,
                    "model_state_dict": self.agent.q.state_dict(),
                    "eval_return": eval_r,
                    "seed": self.seed,
                    "hyperparameters": self.hp,
                }, self.save_path)
                print(f">>> EARLY STOPPING at episode {threshold_ep}")
                break

        if threshold_ep is None:
            torch.save({
                "episode": self.hp["max_episodes"],
                "model_state_dict": self.agent.q.state_dict(),
                "eval_return": eval_r,
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
        # --- Standard RL Hyperparameters ---
        "learning_rate": 6e-4,
        "discount": 0.93,
        "batch_size": 32,
        "update_frequency": 10,
        "max_episodes": 250,
        "epsilon_max": 1.0,
        "epsilon_min": 0.01,
        "epsilon_anneal_episodes": 250,
        "memory_capacity": 4_000,
        "clip_grad_norm": 3.0,
        "eval_episodes": 10,
        "early_stop_threshold": 0.6,

        # --- NEW LLM Hyperparameters ---
        "llm_call_frequency": 5,      # 1) Call LLM every N episodes. 5
        "past_policies_len": 5,       # 2) How many past policies to keep in history  5
        "n_trajectories_llm": 10       # 3) How many trajectories to pass to LLM
    }

    N_RUNS = 1
    BASE_SEED = 10
    crossed = []
    
    for i in range(N_RUNS):
        seed = BASE_SEED + i
        print(f"\n==== RUN {i+1}/{N_RUNS} | SEED {seed} ====")
        trainer = Trainer(HP, seed)
        rows, cross_ep = trainer.train()

        with open(f"run_seed_{seed}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "train_return", "eval_return", "epsilon", "loss", "llm_full_response"])
            writer.writerows(rows)

        if cross_ep is not None:
            crossed.append(cross_ep)

    with open("summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mean_cross_episode", np.mean(crossed) if crossed else "N/A"])
        writer.writerow(["std_cross_episode", np.std(crossed) if crossed else "N/A"])

    print("All runs complete. Logs saved.")
