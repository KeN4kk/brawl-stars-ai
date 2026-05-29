#!/usr/bin/env python3

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
import logging

from game.environment import GameEnvironment
from agents.dqn_agent import DQNAgent, DuelingDQNAgent
from agents.ppo_agent import PPOAgent, A3CAgent
from training.trainer import MultiAgentTrainer
from utils.visualization import Visualizer
from utils.config import load_config

def create_agents(config: dict, device: str, num_agents: int = 8):
    """Create multi-agent system"""
    
    obs_space = config['agent']['observation_space']
    action_space = config['agent']['action_space']
    hidden_layers = config['network']['hidden_layers']
    
    agents = []
    agent_types = [
        "aggressive", "defensive", "healer", "sniper",
        "tank", "speedster", "balanced", "coordinator"
    ]
    
    # Mix different algorithms for diversity
    for i in range(num_agents):
        agent_type = agent_types[i]
        
        if i < 3:
            # DQN agents (aggressive learners)
            agent = DQNAgent(
                state_size=obs_space,
                action_size=action_space,
                hidden_layers=hidden_layers,
                learning_rate=config['training']['learning_rate'],
                gamma=config['training']['gamma'],
                epsilon_start=config['training']['epsilon_start'],
                epsilon_end=config['training']['epsilon_end'],
                epsilon_decay=config['training']['epsilon_decay'],
                buffer_size=config['training']['buffer_size'],
                device=device,
                agent_type=agent_type
            )
        elif i < 6:
            # Dueling DQN agents (balanced)
            agent = DuelingDQNAgent(
                state_size=obs_space,
                action_size=action_space,
                hidden_layers=hidden_layers,
                learning_rate=config['training']['learning_rate'],
                gamma=config['training']['gamma'],
                epsilon_start=config['training']['epsilon_start'],
                epsilon_end=config['training']['epsilon_end'],
                epsilon_decay=config['training']['epsilon_decay'],
                buffer_size=config['training']['buffer_size'],
                device=device,
                agent_type=agent_type
            )
        else:
            # PPO agents (policy gradient)
            agent = PPOAgent(
                state_size=obs_space,
                action_size=action_space,
                hidden_layers=hidden_layers,
                learning_rate=config['training']['learning_rate'],
                gamma=config['training']['gamma'],
                num_epochs=config['ppo']['num_epochs'],
                clip_ratio=config['ppo']['clip_ratio'],
                device=device,
                agent_type=agent_type
            )
        
        agents.append(agent)
    
    return agents

def main():
    parser = argparse.ArgumentParser(description='Brawl Stars AI Training')
    parser.add_argument('--train', action='store_true', help='Train the agents')
    parser.add_argument('--eval', action='store_true', help='Evaluate trained agents')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--visualize', action='store_true', help='Visualize training')
    
    args = parser.parse_args()
    
    # Load configuration
    print("📋 Loading configuration...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"🎮 Brawl Stars AI Bot")
    print(f"{'='*50}")
    print(f"Device: {args.device}")
    print(f"Map Size: {config['game']['map_width']}x{config['game']['map_height']}")
    print(f"Number of Agents: {config['game']['max_players']}")
    print(f"{'='*50}\n")
    
    # Create environment
    print("🌍 Creating game environment...")
    env = GameEnvironment(config, num_agents=config['game']['max_players'])
    
    # Create agents
    print("🤖 Creating AI agents...")
    agents = create_agents(config, args.device, config['game']['max_players'])
    for i, agent in enumerate(agents):
        print(f"   Agent {i}: {agent.agent_type} ({agent.__class__.__name__})")
    
    # Create trainer
    print("📚 Creating trainer...")
    trainer = MultiAgentTrainer(agents, env, config, device=args.device)
    
    # Training
    if args.train:
        print(f"\n🚀 Starting training for {args.episodes} episodes...\n")
        trainer.train(
            num_episodes=args.episodes,
            save_interval=100,
            eval_interval=50
        )
        
        stats = trainer.get_training_stats()
        print(f"\n📊 Training Statistics:")
        print(f"   Total Episodes: {stats['total_episodes']}")
        print(f"   Mean Reward (last 100): {stats['mean_reward']:.2f}")
        print(f"   Best Reward: {stats['best_reward']:.2f}")
    
    # Evaluation
    if args.eval:
        print("\n📊 Evaluating agents...\n")
        eval_stats = trainer.evaluate(num_eval_episodes=20)
        print(f"Evaluation Results:")
        print(f"   Mean Reward: {eval_stats['mean_reward']:.2f}")
        print(f"   Std Reward: {eval_stats['std_reward']:.2f}")
    
    # Visualization
    if args.visualize:
        print("\n🎨 Starting visualization...")
        visualizer = Visualizer(env, agents, config)
        visualizer.visualize_games(num_games=5)
    
    print("\n✅ Done!")

if __name__ == '__main__':
    main()
