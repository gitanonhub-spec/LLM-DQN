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
from openai import OpenAI

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
# LLM ADVISOR (Adapted for CliffWalking)
# ===============================
class LLMPolicyAdvisor:
    def __init__(self):
        self.client = OpenAI(api_key="") 
        
        # CliffWalking-v1 Action Map
        # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        self.action_map = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

    def format_trajectory(self, traj):
        """Converts a list of (s, a, r, ns) into a readable string."""
        path_str = deque(maxlen=10)  # Limit to last 10 steps for readability
        for step in traj:
            s, a, r, ns = step
            path_str.append(f"(State {s} -> {self.action_map.get(a, a)} -> Reward {r})")
        return " -> ".join(path_str)

    def get_policy(self, trajectories, returns, coverage_info, current_epsilon, avg_eval_reward, seed_val, past_policies):
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


        

        # 3. SYSTEM PROMPT (Specific to CliffWalking 4x12)
        system_prompt = """
        You are an Expert explorer helping an agent with exploration on CliffWalking-v1.

        Environment: Gymnasium CliffWalking-v1 (is_slippery=False)

        The environment is deterministic. Every action always produces the intended movement.

        Grid: 4x12 → states 0–47

        [KEEP YOUR GRID DESCRIPTION AS IS]

        Rewards:
        -1 per step
        -100 if entering cliff (then reset to state 36)
        Optimal return (shortest safe path) ≈ -13

        Constraints:
        • Never enter cliff states (37–46).
        • You DO NOT need to define actions for all states.
        • Only provide actions for states where improvement is needed.
        • Focus especially on:
            - Cliff states (37–46)
            - States near cliff (25–35)
            - Start state (36)

        # Important Constraints:
        # - Do NOT assign actions to all states in a region
        # - Only modify 2–5 states that are most problematic
        # - Avoid repeatedly suggesting the same states across iterations unless clearly necessary
        # - Try to explore different states if previous suggestions did not improve performance

        Actions:
        0 → UP
        1 → RIGHT
        2 → DOWN
        3 → LEFT

        Guidance:
        • Identify failure patterns from trajectories.
        • Since the environment is deterministic, prefer the shortest safe path to the goal.
        • Avoid entering cliff states.
        • Improve only critical states.

        Output Requirements:
        • Output ONLY a PARTIAL policy (not full 48 states).
        • Return a dictionary mapping state → action.

        Example:
        {
            "partial_policy": {
                "36": 0,
                "37": 0,
                "38": 0
            },
            "reasoning": "..."
        }
        """
        
        coverage_text = ""

        if coverage_info:

            coverage_text = "Coverage Statistics:\n"

            for region, pct in coverage_info.items():
                coverage_text += f"Action {region}: {pct:.2f}%\n"
        # 4. USER PROMPT

        user_prompt = f"""
            Current Training Status:

            Avg Reward: {avg_eval_reward:.2f} 
            Coverage statistics indicate the percentage of states visited during the 10 greedy evaluation trajectories that belong to each region defined by the current exploration policy.
            {coverage_text}

            Epsilon: {current_epsilon:.2f}

            Past Policies and their performance:
            {past_policies_json}

            Current Trajectories of DQN Learning:
            {history_json}

            Interpretation Guidelines:
            - States 37–46 are CLIFF → entering them gives -100 reward.
            - State 36 is the START state.
            - Since the environment is deterministic, every action always produces the intended movement.
            - Repeated visits to the same states or loops indicate poor action choices rather than stochastic transitions.

            Task:
            - Identify where the agent is repeatedly failing or getting stuck.
            - Focus especially on states near the cliff (35–46).
            - Improve only the critical states.
            - Avoid repeating unsuccessful policy modifications.
            - If similar states were suggested previously without improving the evaluation return, consider modifying different states.

            Additional Guidance:
            - The objective is to reach the goal using the shortest safe path.
            - Avoid entering the cliff.
            - Avoid loops or oscillations between states.
            - Prefer moving RIGHT toward the goal whenever it is safe.
            - Use UP only when necessary to safely avoid the cliff.
            - Do NOT assign the same action to all states.
            - Keep the partial policy as small as possible (modify only the states that require improvement).

            Output JSON only:
            {{
                "partial_policy": {{state: action}},
                "reasoning": "string"
            }}
            """

        # 5. Call LLM
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
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
        """Parses the JSON response to extract partial policy and reasoning."""
        try:
            data = json.loads(text)

            partial_policy = data.get("partial_policy", {})
            reasoning = data.get("reasoning", "")

            if isinstance(partial_policy, dict):
                clean_policy = {}

                for k, v in partial_policy.items():
                    try:
                        k_int = int(k)
                        v_int = int(v)

                        if 0 <= k_int < 48 and 0 <= v_int <= 3:
                            clean_policy[k_int] = v_int
                    except:
                        continue

                if len(clean_policy) == 0:
                    return None, "Empty or invalid partial policy"

                return clean_policy, reasoning

            else:
                return None, "Invalid partial policy format"

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
# AGENT (MODIFIED FOR LLM)
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
        
        # New: Store the LLM's suggested policy
        self.suggested_policy = None

    def act(self, s, action_space):
        # Exploration
        if np.random.rand() < self.eps:
            if self.suggested_policy is not None and np.random.rand() < 0.4:
                state_idx = int(np.argmax(s))

                if 0 <= state_idx < len(self.suggested_policy):
                    action = self.suggested_policy[state_idx]

                    if action is not None:
                        return int(action)

            # fallback if no LLM action available
            return action_space.sample()

        # Exploitation (greedy)
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
def extract_regions_from_policy(policy):
    regions = {}

    for s, a in enumerate(policy):
        if a is not None:
            regions.setdefault(a, []).append(s)

    return regions


def compute_coverage(trajectories, policy):

    regions = extract_regions_from_policy(policy)

    counts = {a: 0 for a in regions}
    total = 0

    for traj in trajectories:
        for s, _, _, _ in traj:

            if policy[s] is None:
                continue

            counts[policy[s]] += 1
            total += 1

    if total == 0:
        return {}

    return {
        a: round(100 * counts[a] / total, 2)
        for a in counts
    }
def collect_greedy_trajectories(agent, env_fn, n_trajs, seed, n_states, current_policy=None):
    env = env_fn()
    trajectories = []
    returns = []
    with torch.no_grad():
        for i in range(n_trajs):
            traj = []
            s_raw, _ = env.reset(seed=seed + 2000 + i)
            s_vec = one_hot(s_raw, n_states)
            done = trunc = False
            ep_r = 0
            while not done and not trunc:
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
    coverage = None

    if current_policy is not None:
        coverage = compute_coverage(
            trajectories,
            current_policy
        )

    return trajectories, returns, coverage

# ===============================
# TRAINER (MODIFIED FOR LLM)
# ===============================
class Trainer:
    def __init__(self, hp, seed):
        # NOTE: is_slippery=True is not standard for CliffWalking-v1 in basic Gym, 
        # but included here as requested by user constraints.
        env = gym.make("CliffWalking-v1", is_slippery=False, max_episode_steps=200)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        
        self.save_path = f"dqn_cliffwalking_random_seed_{seed}.pth"
        self.llm_log_path = f"llm_logs_cliff_random_seed_{seed}.jsonl"
        self.env = env
        self.n_states = env.observation_space.n
        self.seed = seed

        self.agent = Agent(
            obs_dim=self.n_states,
            act_dim=env.action_space.n,
            hp=hp,
            seed=seed
        )
        self.hp = hp
        
        # New: Track past policies
        self.past_policies = deque(maxlen=hp["past_policies_len"]) 

        def make_env():
            def _make():
                e = gym.make("CliffWalking-v1", is_slippery=False, max_episode_steps=200)
                # Offset seed for eval to avoid overfitting to specific spawn logic if any
                e.reset(seed=seed + 10_000)
                e.action_space.seed(seed + 10_000)
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
            eval_r = evaluate_policy(
                self.agent,
                self.eval_env_fn,
                self.hp["eval_episodes"],
                self.seed,
                self.n_states
            )
            loss = self.agent.losses[-1] if self.agent.losses else np.nan

            # 3. LLM Interaction
            llm_response_text = "N/A"
            
            if ep > 0 and (ep + 1) % self.hp["llm_call_frequency"] == 0:
                print(f"   [LLM] Calling GPT-4o for Episode {ep+1}...")
                
                # UPDATE HISTORY: 
                if self.agent.suggested_policy is not None:
                    self.past_policies.append({
                        "proposed_policy": self.agent.suggested_policy,
                        "result_eval_return": eval_r
                    })
                
                # Collect trajectories
                trajs, rets, coverage_info = collect_greedy_trajectories(
                    self.agent, 
                    self.eval_env_fn, 
                    n_trajs=self.hp["n_trajectories_llm"], 
                    seed=ep, 
                    n_states=self.n_states,
                    current_policy=self.agent.suggested_policy
                )
                filtered = [(t, r) for t, r in zip(trajs, rets) if r > -80]

                # Use filtered if available, else fallback to best ones
                if len(filtered) >= 2:
                    trajs, rets = zip(*filtered)
                else:
                    # take top 2 best trajectories instead of all bad ones
                    sorted_pairs = sorted(zip(trajs, rets), key=lambda x: x[1], reverse=True)
                    trajs, rets = zip(*sorted_pairs[:2])

                # convert back to list (important)
                trajs = list(trajs)
                rets = list(rets)
                
                # Call LLM
                llm_response_text, full_user_prompt = self.llm_advisor.get_policy(
                    trajectories=trajs, 
                    returns=rets, 
                    coverage_info=coverage_info,
                    current_epsilon=self.agent.eps,
                    avg_eval_reward=eval_r,
                    seed_val=self.seed,
                    past_policies=list(self.past_policies)
                )
                
                parsed_policy, reasoning = self.llm_advisor.parse_response(llm_response_text)

                # SAFETY LIMIT (VERY IMPORTANT)
                MAX_LLM_STATES = 5

                if parsed_policy and len(parsed_policy) > MAX_LLM_STATES:
                    parsed_policy = dict(list(parsed_policy.items())[:MAX_LLM_STATES])
                
                if parsed_policy:
                    # self.agent.suggested_policy = parsed_policy
                    # if self.agent.suggested_policy is None:
                    self.agent.suggested_policy = [None] * self.n_states

                    for state, action in parsed_policy.items():
                        self.agent.suggested_policy[int(state)] = int(action)
                    print(f"   [LLM] Applied Policy! Length: {len(parsed_policy)}")
                    print(f"   [LLM] Reasoning snippet: {reasoning[:80]}...")
                else:
                    print(f"   [LLM] Failed to parse policy. Error: {reasoning}")

                # Log detailed info
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
        # --- Standard RL Hyperparameters (CliffWalking Baseline) ---
        "learning_rate": 1e-3,
        "discount": 0.92,
        "batch_size": 32,
        "update_frequency": 40,
        "max_episodes": 250,
        "epsilon_max": 1.0,
        "epsilon_min": 0.01,
        "epsilon_anneal_episodes": 150,
        "memory_capacity": 10_000,
        "clip_grad_norm": 4.0,
        "eval_episodes": 20,
        "early_stop_threshold": -13.0,  # CliffWalking scale

        # --- NEW LLM Hyperparameters ---
        "llm_call_frequency": 10,      # Call LLM every 5 episodes
        "past_policies_len": 10,       # Keep last 5 policies in history
        "n_trajectories_llm": 10      # Send 2 sample trajectories to LLM
    }

    N_RUNS = 10
    BASE_SEED = 1
    crossed = []
    
    for i in range(N_RUNS):
        seed = BASE_SEED + i
        print(f"\n==== RUN {i+1}/{N_RUNS} | SEED {seed} ====")
        trainer = Trainer(HP, seed)
        rows, cross_ep = trainer.train()

        with open(f"run_cliffwalking_random_seed_{seed}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "train_return", "eval_return", "epsilon", "loss", "llm_full_response"])
            writer.writerows(rows)

        if cross_ep is not None:
            crossed.append(cross_ep)
        else:
            print(f"Seed {seed} DID NOT CONVERGE within {HP['max_episodes']} episodes")
            crossed.append(HP["max_episodes"])

    with open("summary_cliffwalking_radom.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mean_cross_episode", np.mean(crossed) if crossed else "N/A"])
        writer.writerow(["std_cross_episode", np.std(crossed) if crossed else "N/A"])

    print("All runs complete. Logs saved.")
