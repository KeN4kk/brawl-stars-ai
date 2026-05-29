import numpy as np
import torch
from typing import List, Dict
import json
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

class MultiAgentTrainer:
    """Trainer for multi-agent reinforcement learning"""
    
    def __init__(self, agents: List, env, config: Dict, device: str = 'cpu'):
        self.agents = agents
        self.env = env
        self.config = config
        self.device = device
        
        # Setup logging
        self.log_dir = Path(config['logging']['log_dir'])
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = self._setup_logger()
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.agent_stats = {i: {'wins': 0, 'kills': 0, 'total_reward': 0} 
                           for i in range(len(agents))}
        
        # Best models tracking
        self.best_reward = -float('inf')
        self.checkpoint_dir = Path('./checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def _setup_logger(self):
        logger = logging.getLogger('BrawlStarsAI')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.log_dir / 'training.log')
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def train_episode(self) -> Dict:
        """Train for one episode"""
        observations = self.env.reset()
        
        episode_rewards = np.zeros(len(self.agents))
        episode_length = 0
        
        while not self.env.game_over:
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(self.agents):
                action = agent.select_action(observations[i], training=True)
                actions.append(action)
            
            # Environment step
            next_observations, rewards, dones = self.env.step(np.array(actions))
            episode_rewards += rewards
            episode_length += 1
            
            # Store experiences
            for i, agent in enumerate(self.agents):
                agent.store_transition(observations[i], actions[i], 
                                      next_observations[i], rewards[i], dones[i])
            
            observations = next_observations
            
            # Training step
            if hasattr(agent, 'train_step'):
                for agent in self.agents:
                    if len(agent.replay_buffer) > self.config['training']['batch_size']:
                        agent.train_step(self.config['training']['batch_size'])
        
        # Update statistics
        self.episode_rewards.append(episode_rewards.mean())
        self.episode_lengths.append(episode_length)
        
        for i, player in enumerate(self.env.players):
            self.agent_stats[i]['total_reward'] += episode_rewards[i]
            self.agent_stats[i]['kills'] += player.kills
        
        # Update epsilon for DQN agents
        for agent in self.agents:
            if hasattr(agent, 'update_epsilon'):
                agent.update_epsilon()
        
        return {
            'mean_reward': episode_rewards.mean(),
            'episode_length': episode_length,
            'episode_rewards': episode_rewards
        }
    
    def train(self, num_episodes: int, save_interval: int = 100,
             eval_interval: int = 50):
        """Main training loop"""
        
        self.logger.info(f"Starting training for {num_episodes} episodes")
        self.logger.info(f"Number of agents: {len(self.agents)}")
        
        pbar = tqdm(range(num_episodes), desc="Training")
        
        for episode in pbar:
            episode_stats = self.train_episode()
            
            # Update progress bar
            pbar.set_postfix({
                'mean_reward': f"{episode_stats['mean_reward']:.2f}",
                'length': episode_stats['episode_length']
            })
            
            # Evaluation
            if (episode + 1) % eval_interval == 0:
                eval_stats = self.evaluate(num_eval_episodes=5)
                self.logger.info(f"Episode {episode + 1} - Eval Reward: {eval_stats['mean_reward']:.2f}")
            
            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                self._save_checkpoint(episode + 1)
            
            # Target network update for DQN
            if (episode + 1) % self.config['training']['target_update_frequency'] == 0:
                for agent in self.agents:
                    if hasattr(agent, 'update_target_network'):
                        agent.update_target_network()
        
        self.logger.info("Training completed!")
        self._save_final_models()
    
    def evaluate(self, num_eval_episodes: int = 10) -> Dict:
        """Evaluate agents"""
        
        eval_rewards = []
        
        for _ in range(num_eval_episodes):
            observations = self.env.reset()
            episode_reward = 0
            
            while not self.env.game_over:
                actions = []
                for i, agent in enumerate(self.agents):
                    action = agent.select_action(observations[i], training=False)
                    actions.append(action)
                
                next_observations, rewards, _ = self.env.step(np.array(actions))
                episode_reward += rewards.mean()
                observations = next_observations
            
            eval_rewards.append(episode_reward)
        
        mean_reward = np.mean(eval_rewards)
        
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            self._save_best_models()
        
        return {
            'mean_reward': mean_reward,
            'std_reward': np.std(eval_rewards),
            'eval_rewards': eval_rewards
        }
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode}"
        checkpoint_path.mkdir(exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            agent_path = checkpoint_path / f"agent_{i}.pt"
            if hasattr(agent, 'policy_net'):
                torch.save(agent.policy_net.state_dict(), agent_path)
            elif hasattr(agent, 'network'):
                torch.save(agent.network.state_dict(), agent_path)
        
        # Save stats
        stats_path = checkpoint_path / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump({
                'episode': episode,
                'mean_rewards': self.episode_rewards[-100:],
                'agent_stats': self.agent_stats
            }, f, indent=2)
        
        self.logger.info(f"Checkpoint saved at episode {episode}")
    
    def _save_best_models(self):
        """Save best models"""
        best_path = self.checkpoint_dir / "best_models"
        best_path.mkdir(exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            agent_path = best_path / f"agent_{i}.pt"
            if hasattr(agent, 'policy_net'):
                torch.save(agent.policy_net.state_dict(), agent_path)
            elif hasattr(agent, 'network'):
                torch.save(agent.network.state_dict(), agent_path)
        
        self.logger.info(f"Best models saved with reward: {self.best_reward:.2f}")
    
    def _save_final_models(self):
        """Save final models"""
        final_path = self.checkpoint_dir / "final_models"
        final_path.mkdir(exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            agent_path = final_path / f"agent_{i}.pt"
            if hasattr(agent, 'policy_net'):
                torch.save(agent.policy_net.state_dict(), agent_path)
            elif hasattr(agent, 'network'):
                torch.save(agent.network.state_dict(), agent_path)
        
        self.logger.info("Final models saved!")
    
    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        return {
            'total_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'best_reward': self.best_reward,
            'agent_stats': self.agent_stats
        }
