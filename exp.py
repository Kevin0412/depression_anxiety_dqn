# -*- coding: utf-8 -*-
import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import csv
import time
import os
import json
from datetime import datetime

# Fix numpy compatibility issue
try:
    np.bool8 = np.bool_
except:
    pass

# Create results directory
os.makedirs("experiment_results", exist_ok=True)

env = gym.make("CartPole-v1")

# Set up matplotlib
plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Hyperparameters - Adjusted for better learning
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 800  # Faster decay for more exploitation
TAU = 0.005
LR = 1e-4

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size, model_type="normal"):
        """Sampling function - different strategies based on model type"""
        if len(self.memory) < batch_size:
            return random.sample(self.memory, len(self.memory))
            

        return random.sample(self.memory, batch_size)
        if model_type == "normal":
            # Normal DQN: uniform random sampling
            return random.sample(self.memory, batch_size)
        elif model_type == "depressed_mild":
            # Mildly depressed DQN: slightly less sensitive to high rewards
            return self._sample_depressed_mild(batch_size)
        else:
            return random.sample(self.memory, batch_size)
    
    def _sample_depressed_mild(self, batch_size):
        """Mild depression sampling: slightly reduce sensitivity to high rewards"""
        if len(self.memory) < batch_size:
            return random.sample(self.memory, batch_size)
        
        # Mild discount for high rewards (only 20% reduction)
        rewards = [trans.reward.item() for trans in self.memory]
        max_reward = max(rewards) if max(rewards) > 0 else 1.0
        
        # Reward weights: higher rewards have slightly lower weights
        reward_weights = [1.0 - (0.2 * (max(0, reward) / max_reward)) for reward in rewards]
        
        # Ensure all weights are positive
        reward_weights = [max(w, 0.3) for w in reward_weights]  # Higher minimum weight
        
        # Sample according to weights
        try:
            sampled_indices = random.choices(
                range(len(self.memory)), 
                weights=reward_weights, 
                k=batch_size
            )
        except:
            return random.sample(self.memory, batch_size)
        
        return [self.memory[i] for i in sampled_indices]

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DepressedDQN:
    """简化抑郁DQN模型 - 只使用greedy策略+Q值折扣"""
    
    def __init__(self, n_observations, n_actions, confidence_discount=0.01):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.confidence_discount = confidence_discount  # 自信折扣因子
        
        # 网络结构
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        
    def select_action(self, state, training=True):
        """使用标准greedy策略 - 和正常模型完全一样"""
        if not training:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    
    def optimize_model(self):
        """优化模型 - 核心抑郁机制：对高Q值更新进行折扣"""
        if len(self.memory) < BATCH_SIZE:
            return 0.0
            
        transitions = self.memory.sample(BATCH_SIZE, model_type="normal")  # 使用均匀采样
        batch = Transition(*zip(*transitions))

        # 处理可能为None的next_state
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], 
                                     device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # 计算当前Q值
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 计算目标Q值
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            if non_final_mask.sum() > 0:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # 核心抑郁机制：对高Q值进行折扣
        current_q_values = state_action_values.detach()
        if len(current_q_values) > 0:
            # 找到高Q值的经验（前30%）
            k = max(1, int(BATCH_SIZE * 0.3))  # 取前30%作为高自信经验
            high_confidence_indices = current_q_values.topk(k, dim=0).indices.squeeze()
            
            if high_confidence_indices.numel() > 0:
                # 对高自信经验应用折扣
                discount_factors = torch.ones_like(expected_state_action_values)
                discount_factors[high_confidence_indices] = 1.0 - self.confidence_discount
                expected_state_action_values = expected_state_action_values * discount_factors

        # 计算损失
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()


class NormalDQN:
    """Normal DQN model"""
    
    def __init__(self, n_observations, n_actions):
        self.n_observations = n_observations
        self.n_actions = n_actions
        
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        
        # Track confidence history
        self.confidence_history = deque(maxlen=100)
        
    def select_action(self, state, training=True):
        """Normal action selection"""
        if not training:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    
    def optimize_model(self):
        """Normal optimization"""
        if len(self.memory) < BATCH_SIZE:
            return 0.0
            
        transitions = self.memory.sample(BATCH_SIZE, model_type="normal")
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor([s is not None for s in batch.next_state], 
                                     device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            if non_final_mask.sum() > 0:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # Record average confidence level
        self.confidence_history.append(state_action_values.mean().item())
        
        return loss.item()


def run_single_experiment(model_type, num_episodes=600, run_id=0):
    """Run a single experiment with specified model type"""
    # Get environment info
    state, info = env.reset()
    n_actions = env.action_space.n
    n_observations = len(state)
    
    # Initialize model
    if model_type == "normal":
        model = NormalDQN(n_observations, n_actions)
        model_name = "normal"
    else:
        model = DepressedDQN(n_observations, n_actions)
        model_name = "depressed_mild"
    
    print(f"Running {model_name} - Run {run_id + 1}")
    
    # Track results
    results = {
        'durations': [],
        'rewards': [],
        'confidence_scores': [],
        'success_rate': [],  # Track success (reached 500 steps)
        'exploration_rates': [],
        'q_values_history': []
    }
    
    success_count = 0
    
    for i_episode in range(num_episodes):
        # Calculate current exploration rate
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * model.steps_done / EPS_DECAY)
        
        # Run episode
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        episode_confidence = []
        episode_q_values = []
        
        for t in count():
            if i_episode % 10 == 0:
                print(f"\r  Episode {i_episode}/{num_episodes}",end="")
            action = model.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Record confidence (max Q-value of current state)
            with torch.no_grad():
                q_values = model.policy_net(state)
                confidence = q_values.max().item()
                episode_confidence.append(confidence)
                episode_q_values.append(q_values.mean().item())

            model.memory.push(state, action, next_state, reward)
            state = next_state

            loss = model.optimize_model()

            # Soft update target network
            target_net_state_dict = model.target_net.state_dict()
            policy_net_state_dict = model.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            model.target_net.load_state_dict(target_net_state_dict)

            if done:
                duration = t + 1
                results['durations'].append(duration)
                results['rewards'].append(total_reward)
                results['confidence_scores'].append(np.mean(episode_confidence))
                results['exploration_rates'].append(eps_threshold)
                results['q_values_history'].append(np.mean(episode_q_values))
                
                # Check if successful (reached 500 steps)
                success = 1 if duration >= 500 else 0
                success_count += success
                results['success_rate'].append(success_count / (i_episode + 1))
                
                # Early stopping if consistently successful
                if i_episode > 100 and results['success_rate'][-1] > 0.95:
                    print(f"  Early stopping at episode {i_episode} - consistent success achieved")
                    # Pad remaining episodes with current values
                    remaining_episodes = num_episodes - i_episode - 1
                    if remaining_episodes > 0:
                        results['durations'].extend([duration] * remaining_episodes)
                        results['rewards'].extend([total_reward] * remaining_episodes)
                        results['confidence_scores'].extend([np.mean(episode_confidence)] * remaining_episodes)
                        results['exploration_rates'].extend([eps_threshold] * remaining_episodes)
                        results['q_values_history'].extend([np.mean(episode_q_values)] * remaining_episodes)
                        results['success_rate'].extend([results['success_rate'][-1]] * remaining_episodes)
                    break
                    
                break

    return {
        'model_type': model_type,
        'run_id': run_id,
        'results': results
    }


def run_multiple_experiments(num_runs=5, num_episodes=600):
    """Run multiple experiments for normal vs mild depression"""
    # Only two model types now
    model_configs = ["normal", "depressed_mild"]
    
    all_results = []
    
    for model_type in model_configs:
        model_results = []
        for run_id in range(num_runs):
            result = run_single_experiment(
                model_type=model_type,
                num_episodes=num_episodes,
                run_id=run_id
            )
            model_results.append(result)
        all_results.append(model_results)
    
    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_all_results(all_results, timestamp)
    
    return all_results


def save_all_results(all_results, timestamp):
    """Save all experiment results"""
    # Save detailed results for each run
    for model_results in all_results:
        if not model_results:
            continue
            
        model_type = model_results[0]['model_type']
        
        # Save individual run data
        for result in model_results:
            run_id = result['run_id']
            filename = f"experiment_results/{model_type}_run{run_id}_{timestamp}.csv"
            
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Episode', 'Duration', 'Reward', 'Confidence', 'Success_Rate', 'Exploration_Rate', 'Q_Value'])
                
                for i in range(len(result['results']['durations'])):
                    writer.writerow([
                        i + 1,
                        result['results']['durations'][i],
                        result['results']['rewards'][i],
                        result['results']['confidence_scores'][i],
                        result['results']['success_rate'][i],
                        result['results']['exploration_rates'][i],
                        result['results']['q_values_history'][i] if i < len(result['results']['q_values_history']) else 0
                    ])
    
    # Save summary statistics
    save_summary_statistics(all_results, timestamp)


def save_summary_statistics(all_results, timestamp):
    """Save summary statistics across all runs"""
    summary = {}
    
    for model_results in all_results:
        if not model_results:
            continue
            
        model_type = model_results[0]['model_type']
        model_name = model_type
        
        # Aggregate results across runs
        all_durations = []
        all_confidence = []
        all_success_rates = []
        all_final_success = []
        
        for result in model_results:
            results_data = result['results']
            all_durations.append(results_data['durations'])
            all_confidence.append(results_data['confidence_scores'])
            all_success_rates.append(results_data['success_rate'])
            # Final success rate (last episode)
            all_final_success.append(results_data['success_rate'][-1] if results_data['success_rate'] else 0)
        
        # Calculate statistics and convert numpy types to Python native types
        summary[model_name] = {
            'mean_duration': float(np.mean([np.mean(d) for d in all_durations])),
            'std_duration': float(np.std([np.mean(d) for d in all_durations])),
            'mean_confidence': float(np.mean([np.mean(c) for c in all_confidence])),
            'std_confidence': float(np.std([np.mean(c) for c in all_confidence])),
            'mean_final_success': float(np.mean(all_final_success)),
            'std_final_success': float(np.std(all_final_success)),
            'max_duration': int(np.max([np.max(d) for d in all_durations])),
            'num_runs': len(model_results),
            'first_success_episode': _calculate_first_success(all_durations)
        }
    
    # Save summary to JSON
    with open(f"experiment_results/summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save summary to CSV
    with open(f"experiment_results/summary_{timestamp}.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Mean_Duration', 'Std_Duration', 'Mean_Confidence', 
                        'Std_Confidence', 'Mean_Final_Success', 'Std_Final_Success', 
                        'Max_Duration', 'First_Success_Episode', 'Num_Runs'])
        
        for model_name, stats in summary.items():
            writer.writerow([
                model_name,
                stats['mean_duration'],
                stats['std_duration'],
                stats['mean_confidence'],
                stats['std_confidence'],
                stats['mean_final_success'],
                stats['std_final_success'],
                stats['max_duration'],
                stats['first_success_episode'],
                stats['num_runs']
            ])
    
    print(f"Summary statistics saved to experiment_results/summary_{timestamp}.csv")


def _calculate_first_success(all_durations):
    """Calculate the average first episode where success (500 steps) was achieved"""
    first_successes = []
    for durations in all_durations:
        for i, duration in enumerate(durations):
            if duration >= 500:
                first_successes.append(i + 1)
                break
        else:
            # If never succeeded, use the total number of episodes
            first_successes.append(len(durations))
    return float(np.mean(first_successes)) if first_successes else float(len(all_durations[0]))


if __name__ == "__main__":
    print("Starting multiple experiment runs...")
    print("This will run experiments for Normal DQN vs Mildly Depressed DQN")
    print("Total episodes per run: 1000")
    print("Number of runs per model: 1")
    print("This may take a while...")
    
    # Run experiments
    all_results = run_multiple_experiments(num_runs=1, num_episodes=1000)
    
    print("All experiments completed!")
    print("Results saved to experiment_results/ directory")
    print("You can now run the analysis script to generate plots and statistics.")