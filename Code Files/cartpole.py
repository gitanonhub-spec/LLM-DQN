import os
import gc
import csv
import json
import ast
import torch
import warnings
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from collections import deque
import openai 
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

gc.collect()
warnings.filterwarnings("ignore", category=UserWarning)

# ===============================
# RULE EXECUTOR (From cartpole/policy_executor.py)
# ===============================
FEATURE_INDEX = {
    "x": 0,
    "v": 1,
    "theta": 2,
    "omega": 3,
}

class RulePolicyExecutor:
    """Executes an explicit rule policy output by LLM."""
    def __init__(self, action_n: int, fallback: str = "random"):
        self.action_n = action_n
        self.fallback = fallback

    def _fallback_action(self, obs):
        return int(np.random.randint(0, self.action_n))

    def _clause_true(self, obs, clause: dict) -> bool:
        feat = clause.get("feature")
        op = clause.get("op")
        if feat not in FEATURE_INDEX or op is None:
            return False
        v = float(obs[FEATURE_INDEX[feat]])
        if op in ("<", "<=", ">", ">=", "=="):
            target = float(clause.get("value"))
            if op == "<": return v < target
            if op == "<=": return v <= target
            if op == ">": return v > target
            if op == ">=": return v >= target
            if op == "==": return abs(v - target) < 1e-6
        
        if op == "between":

            if (
                isinstance(clause.get("value"), list)
                and len(clause["value"]) == 2
            ):

                low, high = clause["value"]

            else:

                low = clause.get("low")
                high = clause.get("high")

            if low is None or high is None:
                return False

            return (
                v >= float(low)
                and
                v <= float(high)
            )
        return False

    def _rule_matches(self, obs, rule: dict) -> bool:
        conds = rule.get("if", [])
        if not isinstance(conds, list) or len(conds) == 0:
            return False
        for c in conds:
            if not self._clause_true(obs, c):
                return False
        return True

    def act(self, obs, policy_text: str) -> int:
        try:
            if not policy_text:
                return self._fallback_action(obs)
            policy = json.loads(policy_text)
            if policy.get("policy_type") != "rules":
                return self._fallback_action(obs)
            rules = policy.get("rules", [])
            default_action = int(policy.get("default_action", 0))
            default_action = int(np.clip(default_action, 0, self.action_n - 1))
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            if obs.shape[0] != 4:
                return self._fallback_action(obs)
            for rule in rules:
                if self._rule_matches(obs, rule):
                    a = int(rule.get("then", default_action))
                    return int(np.clip(a, 0, self.action_n - 1))
            return self._fallback_action(obs)
        except Exception:
            return self._fallback_action(obs)
        
# ==================================
# REGION EXTRACTION FROM LLM POLICY
# ==================================

def extract_regions_from_policy(policy_json):
    """
    Converts LLM policy into region definitions.

    Region i -> Rule i
    Last region -> Default region
    """

    try:
        policy = json.loads(policy_json)

        regions = []

        for idx, rule in enumerate(policy.get("rules", [])):
            regions.append({
                "name": f"Rule-{idx+1}",
                "conditions": rule.get("if", [])
            })

        regions.append({
            "name": "Default",
            "conditions": None
        })

        return regions

    except Exception:
        return []
    
def region_to_text(region):
    """
    Converts rule conditions to text for prompt.
    """

    if region["conditions"] is None:
        return "Default Region"

    pieces = []

    for c in region["conditions"]:
        feat = c["feature"]
        op = c["op"]

        if op == "between":

            if (
                isinstance(c.get("value"), list)
                and len(c["value"]) == 2
            ):

                low, high = c["value"]

                pieces.append(
                    f"{low} <= {feat} <= {high}"
                )

            else:

                low = c.get("low")
                high = c.get("high")

                pieces.append(
                    f"{low} <= {feat} <= {high}"
                )
        else:
            pieces.append(
                f"{feat} {op} {c['value']}"
            )

    return " AND ".join(pieces)

def get_region_id(obs, regions):
    """
    Returns region index for a state.

    First matching rule wins.
    """

    executor = RulePolicyExecutor(
        action_n=2
    )

    for idx, region in enumerate(regions):

        if region["conditions"] is None:
            continue

        matched = True

        for cond in region["conditions"]:
            if not executor._clause_true(obs, cond):
                matched = False
                break

        if matched:
            return idx

    return len(regions) - 1

# ===============================
# LLM ADVISOR (Adapted for CartPole)
# ===============================
class LLMPolicyAdvisor:
    def __init__(self):
        self.client = openai.OpenAI(api_key="") 
        
    def get_policy(self, episode, policy_history, seed_val):
        """Constructs prompt using policy history."""
        history_text = ""
        if policy_history:
            history_text = "\nPast Policies:\n"
            for i, (p_json, ret_val, coverage_info) in enumerate(policy_history):

                try:
                    p_dict = json.loads(p_json)
                    if "notes" in p_dict:
                        del p_dict["notes"]
                    p_str = json.dumps(p_dict, indent=2)
                    heading = f"--- Policy-{i+1} ---"
                    perf_text = f"Evaluated Return: {ret_val:.2f}"
                    context_text = "This previous explorative policy given by you was used to train my network and currently this is the average return i get on evaluating the training."
                    coverage_text = ""
                    if coverage_info:
                        coverage_text += \
                            "\nCoverage Statistics:\n"

                        for c in coverage_info:

                            coverage_text += (
                                f"- {c['region_name']} : "
                                f"{c['coverage']}%\n"
                            )
                    history_text += (
                        f"{heading}\n"
                        f"{p_str}\n"
                        f"{context_text}\n"
                        f"{perf_text}\n"
                        f"{coverage_text}\n\n"
                    )
                except Exception as e:
                    heading = f"--- Policy-{i+1} ---"
                    perf_text = f"Evaluated Return: {ret_val:.2f}"
                    history_text += f"{heading}\n{p_json}\nEvaluated Return: {ret_val:.2f}\n\n"
        else:
            history_text = "\nNo previous policies yet.\n"

        prompt = f"""
        You are a smart environment explorer working on the CartPole Environment.
        The goal of the agent is to balance a pole on a cart.
        However, YOUR goal is to generate a rule-based EXPLORATION policy to help the DQN agent learn faster.

        CartPole Observation Space (4-dimensional vector):
        1) x: Cart Position
        2) v: Cart Velocity
        3) theta: Pole Angle
        4) omega: Pole Angular Velocity

        CartPole Action Space (Discrete):
        0) Push cart to the left
        1) Push cart to the right

        You are given the history of previous exploration policies you suggested.

        ALONG WITH EACH POLICY, we provide:
        1) The average return obtained by the agent trained using that policy (evaluated over 50 episodes using the greedy DQN policy).
        2) Coverage statistics of the regions defined by that policy.

        The coverage statistics indicate the percentage of visited states that belong to each region during the 50 evaluation episodes.

        Use BOTH the evaluation return and the region coverage statistics to generate your next exploratory policy.
        Please remember that the objective is not exploration itself, but to improve the DQN agent's learning performance and achieve higher evaluation returns.

        {history_text}

        REQUIREMENTS FOR THE NEW POLICY:
        - Output ONLY a JSON string (no markdown, no extra text).
        - Follow the schema exactly (same structure as below).
        - Rules must be non-conflicting. Use mutually exclusive conditions OR rely on ordering (first match applies).
        - Use BOTH the evaluation return and the region coverage statistics
        
        Now generate a NEW, IMPROVED exploration policy in JSON format:
        {{
          "policy_type": "rules",
          "name": "policy_ep_{episode}",
          "rules": [
            {{"if": [ {{"feature":"theta","op":"<","value":-0.1}}, ... ], "then": 1}},
            ...
          ],
          "default_action": 0,
          "notes": "Explain which previous policy influenced the new policy and why. When referencing a previous policy, explicitly mention its evaluated return and coverage statistics."
        }}

        Output ONLY the JSON string.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful RL assistant. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                seed=seed_val,
            )
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return content.strip(), prompt
        except Exception as e:
            return f"{{\"error\": \"{str(e)}\"}}", prompt

    def parse_response(self, text):
        try:
            data = json.loads(text)
            if "rules" in data and "default_action" in data:
                return text, data.get("notes", "")
            else:
                return None, "Missing rules or default_action keys"
        except json.JSONDecodeError as e:
            return None, f"JSON Decode Error: {e}"
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
            torch.tensor(np.array([self.states[i] for i in idx]), dtype=torch.float32).to(device),
            torch.tensor([self.actions[i] for i in idx], dtype=torch.long).to(device),
            torch.tensor(np.array([self.next_states[i] for i in idx]), dtype=torch.float32).to(device),
            torch.tensor([self.rewards[i] for i in idx], dtype=torch.float32).to(device),
            torch.tensor([self.dones[i] for i in idx], dtype=torch.bool).to(device),
        )

    def __len__(self):
        return len(self.dones)

# ===============================
# DQN NETWORK
# ===============================
class DQN(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, x):
        return self.net(x)

# ===============================
# AGENT
# ===============================
class Agent:
    def __init__(self, obs_dim, act_dim, hp, seed):
        self.gamma = hp["discount"]
        self.eps_start = hp["epsilon_start"]
        self.eps_end = hp["epsilon_end"]
        self.eps_decay = hp["epsilon_decay"]
        self.tau = hp["tau"]

        self.eps = self.eps_start
        self.global_step = 0

        self.mem = ReplayMemory(hp["memory_capacity"])

        self.q = DQN(obs_dim, act_dim, hp["hidden_size"]).to(device)
        self.qt = DQN(obs_dim, act_dim, hp["hidden_size"]).to(device)
        self.qt.load_state_dict(self.q.state_dict())
        self.qt.eval()

        self.opt = optim.AdamW(self.q.parameters(), lr=hp["learning_rate"], amsgrad=True)
        self.loss_fn = nn.SmoothL1Loss()

        self.losses = []
        self._loss_acc = 0.0
        self._loss_count = 0
        
        self.suggested_policy_json = ""
        self.rule_executor = RulePolicyExecutor(action_n=act_dim, fallback="random")

    def act(self, s, action_space):
        # 1. Exploration
        if np.random.rand() < self.eps:
            return self.rule_executor.act(s, self.suggested_policy_json)

        # 2. Exploitation
        with torch.no_grad():
            s_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
            return self.q(s_tensor).argmax().item()

    def learn(self, batch, done):
        s, a, ns, r, d = self.mem.sample(batch)

        qsa = self.q(s).gather(1, a.unsqueeze(1))
        with torch.no_grad():
            max_next_q = self.qt(ns).max(1)[0].unsqueeze(1)
            target_q = r.unsqueeze(1) + (self.gamma * max_next_q * (~d).float().unsqueeze(1))

        loss = self.loss_fn(qsa, target_q)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q.parameters(), 100)
        self.opt.step()

        self._loss_acc += loss.item()
        self._loss_count += 1

        if done:
            self.losses.append(self._loss_acc / self._loss_count)
            self._loss_acc = 0.0
            self._loss_count = 0

    def update_target(self):
        target_net_state_dict = self.qt.state_dict()
        policy_net_state_dict = self.q.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.qt.load_state_dict(target_net_state_dict)

    def update_epsilon(self):
        self.global_step += 1
        self.eps = max(
            self.eps_end,
            self.eps_start - (self.eps_start - self.eps_end) * (self.global_step / self.eps_decay)
        )

# ===============================
# EVALUATION UTILS
# ===============================
def evaluate_policy(agent, env_fn, n, seed):
    env = env_fn()
    scores = []
    for i in range(n):
        s, _ = env.reset(seed=seed + 1000 + i)
        done = trunc = False
        ep_r = 0
        while not done and not trunc:
            with torch.no_grad():
                s_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
                a = agent.q(s_tensor).argmax().item()
            ns, r, done, trunc, _ = env.step(a)
            s = ns
            ep_r += r
        scores.append(ep_r)
    env.close()
    return np.mean(scores)
def evaluate_policy_with_coverage(
    agent,
    env_fn,
    n,
    seed,
    regions
):

    env = env_fn()

    scores = []

    region_counts = np.zeros(len(regions))

    total_states = 0

    for i in range(n):

        s, _ = env.reset(seed=seed + 1000 + i)

        done = trunc = False

        ep_r = 0

        while not done and not trunc:

            rid = get_region_id(s, regions)

            region_counts[rid] += 1

            total_states += 1

            with torch.no_grad():
                st = torch.FloatTensor(s).unsqueeze(0).to(device)

                a = agent.q(st).argmax().item()

            ns, r, done, trunc, _ = env.step(a)

            s = ns

            ep_r += r

        scores.append(ep_r)

    env.close()

    coverage_info = []

    for idx, region in enumerate(regions):

        pct = 100.0 * region_counts[idx] / max(total_states, 1)

        coverage_info.append({
            "region_name":
                region_to_text(region),
            "coverage":
                round(pct, 2)
        })

    return np.mean(scores), coverage_info
# ===============================
# TRAINER 
# ===============================
class Trainer:
    def __init__(self, hp, seed):
        set_seed(seed)
        env = gym.make(hp["env_id"])
        env.reset(seed=seed)
        env.action_space.seed(seed)
        
        self.save_path = f"dqn_cartpole_seed_{seed}.pth"
        self.llm_log_path = f"llm_logs_cartpole_seed_{seed}.jsonl"
        self.env = env
        self.n_states = env.observation_space.shape[0]
        self.seed = seed

        self.agent = Agent(
            obs_dim=self.n_states,
            act_dim=env.action_space.n,
            hp=hp,
            seed=seed
        )
        self.hp = hp
        
        self.past_policies = deque(maxlen=hp["past_policies_len"]) 

        self.current_regions = [] #current active LLM policy

        def make_env():
            def _make():
                e = gym.make(hp["env_id"])
                e.reset(seed=seed + 10_000)
                e.action_space.seed(seed + 10_000)
                return e
            return _make
            
        self.eval_env_fn = make_env()
        self.llm_advisor = LLMPolicyAdvisor()

    def train(self):
        rows = []
        threshold_ep = None
        
        with open(self.llm_log_path, "w") as f:
            f.write("")

        for ep in range(self.hp["max_episodes"]):
            s, _ = self.env.reset(seed=self.seed + ep)
            done = trunc = False
            ep_r = 0

            # LLM Interaction
            llm_response_text = "N/A"
            if ep > 30 and ep % self.hp["llm_call_frequency"] == 0:
                print(f"   [LLM] Calling GPT-4o for Episode {ep}...")
                
                # eval_r_for_llm = evaluate_policy(
                #     self.agent, self.eval_env_fn, 20, self.seed
                # )
                if len(self.current_regions) > 0:
                    eval_r_for_llm, coverage_info = \
                        evaluate_policy_with_coverage(
                            self.agent,
                            self.eval_env_fn,
                            50,
                            self.seed,
                            self.current_regions
                        )
                    print("\n[Coverage Statistics]")
                    for c in coverage_info:
                        print(
                            f"{c['region_name']} : "
                            f"{c['coverage']}%"
                        )
                    print()
                else:
                    eval_r_for_llm = evaluate_policy(
                        self.agent,
                        self.eval_env_fn,
                        50,
                        self.seed
                    )
                    coverage_info = []

                
                if self.agent.suggested_policy_json:
                    self.past_policies.append((self.agent.suggested_policy_json, eval_r_for_llm, coverage_info))
                
                llm_response_text, full_user_prompt = self.llm_advisor.get_policy(
                    episode=ep,
                    policy_history=list(self.past_policies),
                    seed_val=self.seed
                )
                
                parsed_policy, reasoning = self.llm_advisor.parse_response(llm_response_text)
                
                if parsed_policy:
                    self.agent.suggested_policy_json = parsed_policy

                    self.current_regions = \
                        extract_regions_from_policy(
                            parsed_policy
                        )
                    print(
                        f"[LLM] Extracted "
                        f"{len(self.current_regions)} regions"
                    )
                    print(f"   [LLM] Applied Policy!")
                    print(f"   [LLM] Notes: {reasoning}")
                else:
                    print(f"   [LLM] Failed to parse policy. Error: {reasoning}")

                log_entry = {
                    "episode": ep,
                    "user_prompt": full_user_prompt,
                    "raw_json_response": llm_response_text,
                    "parsed_policy": parsed_policy,
                    "reasoning": reasoning,
                    "current_eval_score": eval_r_for_llm,
                    "coverage_info": coverage_info,
                    "current_epsilon": self.agent.eps
                }
                with open(self.llm_log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")


            while not done and not trunc:
                a = self.agent.act(s, self.env.action_space)
                ns, r, done, trunc, _ = self.env.step(a)

                self.agent.mem.store(s, a, ns, r, done or trunc)

                if len(self.agent.mem) > self.hp["min_buffer_size"] and self.agent.global_step % self.hp["train_freq"] == 0:
                    self.agent.learn(self.hp["batch_size"], done or trunc)
                    
                self.agent.update_target()
                self.agent.update_epsilon()

                s = ns
                ep_r += r

            # Evaluation
            eval_r = evaluate_policy(self.agent, self.eval_env_fn, 50, self.seed)
            loss = self.agent.losses[-1] if self.agent.losses else np.nan

            rows.append([ep + 1, ep_r, eval_r, self.agent.eps, loss, llm_response_text])
            print(f"Seed {self.seed} | Ep {ep+1:3d} | TrainR {ep_r:.2f} | EvalR {eval_r:.2f} | Eps {self.agent.eps:.3f}")

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
            print(f">>> DID NOT CONVERGE within {self.hp['max_episodes']} episodes")
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
        # --- Standard RL Hyperparameters (CartPole Baseline) ---
        "env_id": "CartPole-v1",
        "learning_rate": 3e-4,
        "discount": 0.99,
        "batch_size": 128,
        "train_freq": 1,
        "max_episodes": 500,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 12000, # Steps
        "memory_capacity": 10_000,
        "min_buffer_size": 500,
        "tau": 0.005,
        "hidden_size": 128,
        "early_stop_threshold": 450.0,

        # --- NEW LLM Hyperparameters ---
        "llm_call_frequency": 20,      
        "past_policies_len": 10,       
    }

    SEEDS = [1,2,3,4,5,6,7,8,9,10]
    crossed = []
    
    for i, seed in enumerate(SEEDS):
        print(f"\n==== RUN {i+1}/{len(SEEDS)} | SEED {seed} ====")
        trainer = Trainer(HP, seed)
        rows, cross_ep = trainer.train()

        with open(f"run_cartpole_llm_seed_{seed}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "train_return", "eval_return", "epsilon", "loss", "llm_full_response"])
            writer.writerows(rows)

        if cross_ep is not None:
            crossed.append(cross_ep)
        else:
            crossed.append(HP["max_episodes"])

    with open("summary_cartpole_llm.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mean_cross_episode", np.mean(crossed) if crossed else "N/A"])
        writer.writerow(["std_cross_episode", np.std(crossed) if crossed else "N/A"])

    print("All runs complete. Logs saved.")
