import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import math

@dataclass
class Vector2:
    x: float
    y: float
    
    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)
    
    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def normalize(self):
        dist = math.sqrt(self.x**2 + self.y**2)
        if dist == 0:
            return Vector2(0, 0)
        return Vector2(self.x / dist, self.y / dist)

class Projectile:
    def __init__(self, pos: Vector2, velocity: Vector2, damage: int, owner_id: int):
        self.pos = pos
        self.velocity = velocity
        self.damage = damage
        self.owner_id = owner_id
        self.lifetime = 5.0  # seconds
        self.active = True
    
    def update(self, dt: float):
        self.pos = self.pos + self.velocity * dt
        self.lifetime -= dt
        if self.lifetime <= 0:
            self.active = False
    
    def is_active(self):
        return self.active and self.lifetime > 0

class Player:
    def __init__(self, player_id: int, pos: Vector2, agent_type: str):
        self.id = player_id
        self.pos = pos
        self.velocity = Vector2(0, 0)
        self.agent_type = agent_type
        
        # Stats
        self.health = 100
        self.max_health = 100
        self.ammo = 12
        self.max_ammo = 12
        self.ammo_regen_time = 0
        self.ammo_regen_rate = 2
        
        # Combat
        self.damage = 10
        self.fire_rate = 0.5
        self.last_shot_time = 0
        self.speed = 150  # pixels per second
        self.attack_range = 300
        
        # State
        self.alive = True
        self.kills = 0
        self.damage_dealt = 0
        self.damage_taken = 0
    
    def update(self, dt: float):
        # Ammo regeneration
        self.ammo_regen_time += dt
        if self.ammo_regen_time >= 1.0 and self.ammo < self.max_ammo:
            self.ammo += self.ammo_regen_rate
            self.ammo_regen_time = 0
        
        self.ammo = min(self.ammo, self.max_ammo)
        self.last_shot_time += dt
    
    def take_damage(self, damage: int):
        self.health -= damage
        self.damage_taken += damage
        if self.health <= 0:
            self.alive = False
            self.health = 0
    
    def heal(self, amount: int):
        self.health = min(self.health + amount, self.max_health)
    
    def can_shoot(self):
        return self.ammo > 0 and self.last_shot_time >= self.fire_rate

class GameEnvironment:
    def __init__(self, config: Dict, num_agents: int = 8):
        self.config = config
        self.num_agents = num_agents
        
        self.width = config['game']['map_width']
        self.height = config['game']['map_height']
        self.duration = config['game']['game_duration']
        
        self.players: List[Player] = []
        self.projectiles: List[Projectile] = []
        self.time_elapsed = 0
        self.game_over = False
        
        self._init_players()
    
    def _init_players(self):
        """Initialize all players"""
        agent_types = [
            "aggressive", "defensive", "healer", "sniper",
            "tank", "speedster", "balanced", "coordinator"
        ]
        
        for i in range(self.num_agents):
            x = np.random.randint(50, self.width - 50)
            y = np.random.randint(50, self.height - 50)
            pos = Vector2(float(x), float(y))
            player = Player(i, pos, agent_types[i])
            self.players.append(player)
    
    def reset(self):
        """Reset game environment"""
        self.players = []
        self.projectiles = []
        self.time_elapsed = 0
        self.game_over = False
        self._init_players()
        return self._get_observations()
    
    def step(self, actions: np.ndarray, dt: float = 0.016) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute one step of the environment
        
        Actions: [move_x, move_y, shoot, ability, target_x, target_y]
        Returns: observations, rewards, dones
        """
        self.time_elapsed += dt
        
        # Check if game is over
        alive_players = sum(1 for p in self.players if p.alive)
        if self.time_elapsed >= self.duration or alive_players <= 1:
            self.game_over = True
        
        # Update all players
        for i, player in enumerate(self.players):
            if not player.alive:
                continue
            
            action = actions[i]
            
            # Movement
            move_x = np.clip(action[0], -1, 1)
            move_y = np.clip(action[1], -1, 1)
            
            velocity = Vector2(move_x, move_y).normalize()
            player.velocity = velocity * player.speed
            player.pos = player.pos + player.velocity * dt
            
            # Boundary collision
            player.pos.x = np.clip(player.pos.x, 20, self.width - 20)
            player.pos.y = np.clip(player.pos.y, 20, self.height - 20)
            
            # Shooting
            if action[2] > 0.5 and player.can_shoot():
                self._shoot(player, Vector2(action[4], action[5]))
            
            # Ability use
            if action[3] > 0.5:
                self._use_ability(player)
            
            player.update(dt)
        
        # Update projectiles and check collisions
        self._update_projectiles(dt)
        self._check_collisions()
        
        # Get observations and rewards
        observations = self._get_observations()
        rewards = self._calculate_rewards()
        dones = np.array([not p.alive or self.game_over for p in self.players], dtype=bool)
        
        return observations, rewards, dones
    
    def _shoot(self, player: Player, target_dir: Vector2):
        """Fire a projectile"""
        if player.ammo <= 0:
            return
        
        direction = target_dir.normalize()
        velocity = direction * 400  # projectile speed
        projectile = Projectile(player.pos, velocity, player.damage, player.id)
        self.projectiles.append(projectile)
        
        player.ammo -= 1
        player.last_shot_time = 0
    
    def _use_ability(self, player: Player):
        """Use special ability based on agent type"""
        if player.agent_type == "healer":
            # Heal nearby allies
            for other in self.players:
                if other.id != player.id and other.pos.distance_to(player.pos) < 150:
                    other.heal(20)
        
        elif player.agent_type == "tank":
            # Temporary shield
            player.max_health += 50
            player.health = player.max_health
        
        elif player.agent_type == "speedster":
            # Speed boost
            player.speed *= 1.5
    
    def _update_projectiles(self, dt: float):
        """Update all projectiles"""
        for projectile in self.projectiles[:]:
            projectile.update(dt)
            if not projectile.is_active():
                self.projectiles.remove(projectile)
    
    def _check_collisions(self):
        """Check projectile-player collisions"""
        for projectile in self.projectiles[:]:
            for player in self.players:
                if player.id == projectile.owner_id or not player.alive:
                    continue
                
                if player.pos.distance_to(projectile.pos) < 15:
                    player.take_damage(projectile.damage)
                    
                    # Award kill credit
                    for p in self.players:
                        if p.id == projectile.owner_id and p.alive:
                            p.damage_dealt += projectile.damage
                            if not player.alive:
                                p.kills += 1
                    
                    if projectile in self.projectiles:
                        self.projectiles.remove(projectile)
                    break
    
    def _get_observations(self) -> np.ndarray:
        """Get observations for all agents"""
        observations = []
        
        for i, player in enumerate(self.players):
            obs = []
            
            # Self state (7)
            obs.append(player.pos.x / self.width)
            obs.append(player.pos.y / self.height)
            obs.append(player.health / player.max_health)
            obs.append(player.ammo / player.max_ammo)
            obs.append(player.velocity.x / player.speed)
            obs.append(player.velocity.y / player.speed)
            obs.append(float(player.alive))
            
            # Relative positions to 5 nearest enemies (30 values)
            distances = []
            for j, other in enumerate(self.players):
                if i != j and other.alive:
                    dist = player.pos.distance_to(other.pos)
                    distances.append((dist, j))
            
            distances.sort()
            distances = distances[:5]
            
            for dist, j in distances:
                other = self.players[j]
                obs.append((other.pos.x - player.pos.x) / self.width)
                obs.append((other.pos.y - player.pos.y) / self.height)
                obs.append(other.health / other.max_health)
                obs.append(other.ammo / other.max_ammo)
                obs.append(dist / max(self.width, self.height))
                obs.append(1.0)
            
            # Pad with zeros if less than 5 enemies
            while len(distances) < 5:
                obs.extend([0, 0, 0, 0, 0, 0])
                distances.append(None)
            
            # Game state (5)
            obs.append(self.time_elapsed / self.duration)
            obs.append(sum(1 for p in self.players if p.alive) / self.num_agents)
            obs.append(player.kills / max(1, self.num_agents))
            obs.append(player.damage_dealt / 1000.0)
            obs.append(player.damage_taken / 1000.0)
            
            observations.append(np.array(obs, dtype=np.float32))
        
        return np.array(observations)
    
    def _calculate_rewards(self) -> np.ndarray:
        """Calculate rewards for each agent"""
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        
        for i, player in enumerate(self.players):
            if not player.alive:
                rewards[i] = -10  # Death penalty
            else:
                # Reward for staying alive
                rewards[i] += 0.1
                
                # Reward for kills
                rewards[i] += player.kills * 5
                
                # Reward for damage
                rewards[i] += min(player.damage_dealt / 100, 5)
                
                # Penalty for taking damage
                rewards[i] -= min(player.damage_taken / 100, 5)
        
        return rewards
    
    def get_state(self) -> Dict:
        """Get current game state"""
        return {
            'players': [(p.id, p.pos.x, p.pos.y, p.health, p.alive) for p in self.players],
            'projectiles': [(pr.pos.x, pr.pos.y, pr.owner_id) for pr in self.projectiles],
            'time_elapsed': self.time_elapsed,
            'game_over': self.game_over
        }
