import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Dict
import pygame
from pathlib import Path

class Visualizer:
    """Visualize game state and training progress"""
    
    def __init__(self, env, agents: List, config: Dict):
        self.env = env
        self.agents = agents
        self.config = config
        
        self.save_dir = Path(config['visualization']['plots_dir'])
        self.save_dir.mkdir(exist_ok=True)
    
    def visualize_games(self, num_games: int = 5, save: bool = True):
        """Visualize multiple games"""
        
        for game_num in range(num_games):
            print(f"Visualizing game {game_num + 1}/{num_games}...")
            self._visualize_game(game_num, save)
    
    def _visualize_game(self, game_num: int, save: bool = True):
        """Visualize a single game"""
        
        observations = self.env.reset()
        frames = []
        
        # Collect frames
        while not self.env.game_over:
            state = self.env.get_state()
            frames.append(state)
            
            # Get actions
            actions = []
            for i, agent in enumerate(self.agents):
                action = agent.select_action(observations[i], training=False)
                actions.append(action)
            
            # Step
            next_observations, _, _ = self.env.step(np.array(actions))
            observations = next_observations
        
        # Render frames
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Render map
        ax = axes[0, 0]
        self._render_frame(frames[-1], ax)
        ax.set_title('Final Game State')
        
        # Player stats
        ax = axes[0, 1]
        self._plot_player_stats(ax)
        ax.set_title('Player Statistics')
        
        # Reward over time
        ax = axes[1, 0]
        self._plot_rewards(ax)
        ax.set_title('Cumulative Rewards')
        
        # Health over time
        ax = axes[1, 1]
        self._plot_health(frames, ax)
        ax.set_title('Player Health')
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f"game_{game_num}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.close()
    
    def _render_frame(self, state: Dict, ax):
        """Render game state on matplotlib axis"""
        
        width = self.config['game']['map_width']
        height = self.config['game']['map_height']
        
        # Draw map
        rect = patches.Rectangle((0, 0), width, height, 
                                 linewidth=2, edgecolor='black', 
                                 facecolor='lightgreen', alpha=0.3)
        ax.add_patch(rect)
        
        # Draw players
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'brown']
        for player_id, x, y, health, alive in state['players']:
            color = colors[player_id % len(colors)]
            alpha = 0.8 if alive else 0.3
            circle = patches.Circle((x, y), 15, color=color, alpha=alpha)
            ax.add_patch(circle)
            ax.text(x, y, str(player_id), ha='center', va='center', fontsize=8)
        
        # Draw projectiles
        for x, y, owner_id in state['projectiles']:
            circle = patches.Circle((x, y), 5, color='red', alpha=0.7)
            ax.add_patch(circle)
        
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    def _plot_player_stats(self, ax):
        """Plot player statistics"""
        
        player_ids = list(range(len(self.agents)))
        kills = [self.env.players[i].kills for i in player_ids]
        health = [max(0, self.env.players[i].health) for i in player_ids]
        
        x = np.arange(len(player_ids))
        width = 0.35
        
        ax.bar(x - width/2, kills, width, label='Kills', alpha=0.8)
        ax.bar(x + width/2, health, width, label='Health', alpha=0.8)
        
        ax.set_xlabel('Player ID')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(player_ids)
        ax.legend()
    
    def _plot_rewards(self, ax):
        """Plot cumulative rewards"""
        
        # This would require tracking rewards over time
        # For now, just plot player kills
        kills = [self.env.players[i].kills for i in range(len(self.agents))]
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'brown']
        
        bars = ax.bar(range(len(self.agents)), kills, color=colors[:len(self.agents)], alpha=0.8)
        ax.set_xlabel('Agent ID')
        ax.set_ylabel('Kills')
    
    def _plot_health(self, frames: List[Dict], ax):
        """Plot health over time"""
        
        time_steps = len(frames)
        num_agents = len(self.agents)
        
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'brown']
        
        for agent_id in range(min(num_agents, 4)):  # Plot first 4 agents
            health_history = []
            
            for frame in frames:
                players_dict = {p[0]: p for p in frame['players']}
                if agent_id in players_dict:
                    health = players_dict[agent_id][3]
                    health_history.append(max(0, health))
                else:
                    health_history.append(0)
            
            ax.plot(health_history, label=f'Agent {agent_id}', 
                   color=colors[agent_id], linewidth=2)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Health')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_training_progress(self, episode_rewards: List[float], 
                              episode_lengths: List[int], save: bool = True):
        """Plot training progress"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Reward over episodes
        ax1.plot(episode_rewards, linewidth=1, alpha=0.7)
        if len(episode_rewards) > 100:
            ma = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
            ax1.plot(range(99, len(episode_rewards)), ma, linewidth=2, label='MA-100')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Mean Reward')
        ax1.set_title('Training Reward Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Episode length
        ax2.plot(episode_lengths, linewidth=1, alpha=0.7, color='orange')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length (steps)')
        ax2.set_title('Episode Duration')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / "training_progress.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.close()
