import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
import pickle
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    n_users: int = 5
    demos_per_user: int = 10
    grid_size: int = 10
    maze_size: int = 12
    beta_temperature: float = 5.0
    lambda_constraint: float = 1.0
    learning_rate: float = 1e-3
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    mcmc_iterations: int = 10000
    mcmc_burnin: int = 5000
    mcmc_chains: int = 4
    prior_variance: float = 1.0

class GridWorld:
    """Gridworld environment with obstacles"""
    
    def __init__(self, size: int = 10, obstacles: List[Tuple[int, int]] = None):
        self.size = size
        self.obstacles = obstacles if obstacles else [(5, 2), (5, 7)]
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        self.action_names = ['right', 'left', 'down', 'up']
        
    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        """Check if state is within bounds and not an obstacle"""
        x, y = state
        return (0 <= x < self.size and 0 <= y < self.size and 
                state not in self.obstacles)
    
    def get_next_state(self, state: Tuple[int, int], action_idx: int) -> Tuple[int, int]:
        """Get next state from current state and action"""
        x, y = state
        dx, dy = self.actions[action_idx]
        next_state = (x + dx, y + dy)
        
        # If next state is invalid, stay in current state (collision)
        if not self.is_valid_state(next_state):
            return state
        return next_state
    
    def get_features(self, state: Tuple[int, int], action_idx: int) -> np.ndarray:
        """Extract features for constraint function"""
        x, y = state
        dx, dy = self.actions[action_idx]
        next_state = (x + dx, y + dy)
        
        features = []
        
        # Feature 1: Collision indicator (hard constraint violation)
        collision = not self.is_valid_state(next_state)
        features.append(float(collision))
        
        # Feature 2: Distance to nearest obstacle
        min_dist = float('inf')
        for ox, oy in self.obstacles:
            dist = abs(x - ox) + abs(y - oy)  # Manhattan distance
            min_dist = min(min_dist, dist)
        features.append(1.0 / (1.0 + min_dist))  # Normalized inverse distance
        
        # Feature 3: Moving toward obstacle indicator
        next_x, next_y = next_state
        current_min_dist = min_dist
        next_min_dist = float('inf')
        for ox, oy in self.obstacles:
            dist = abs(next_x - ox) + abs(next_y - oy)
            next_min_dist = min(next_min_dist, dist)
        
        moving_toward = float(next_min_dist < current_min_dist and not collision)
        features.append(moving_toward)
        
        # Feature 4: Boundary proximity
        boundary_dist = min(x, y, self.size - 1 - x, self.size - 1 - y)
        features.append(1.0 / (1.0 + boundary_dist))
        
        # Feature 5: Action type (for different movement patterns)
        action_features = [0.0] * 4
        action_features[action_idx] = 1.0
        features.extend(action_features)
        
        return np.array(features)

class StochasticMaze:
    """Stochastic maze environment with slip dynamics"""
    
    def __init__(self, size: int = 12, slip_prob: float = 0.2):
        self.size = size
        self.slip_prob = slip_prob
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.action_names = ['right', 'left', 'down', 'up']
        
        # Create maze walls (simplified for demo)
        self.walls = set()
        # Vertical walls
        for y in range(2, 8):
            self.walls.add((4, y))
            self.walls.add((7, y))
        # Horizontal walls  
        for x in range(1, 4):
            self.walls.add((x, 4))
        for x in range(8, 11):
            self.walls.add((x, 4))
            
    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        """Check if state is valid"""
        x, y = state
        return (0 <= x < self.size and 0 <= y < self.size and 
                state not in self.walls)
    
    def get_next_state(self, state: Tuple[int, int], action_idx: int) -> Tuple[int, int]:
        """Get next state with stochastic dynamics"""
        x, y = state
        
        # With slip_prob, take random orthogonal action
        if np.random.random() < self.slip_prob:
            # Get orthogonal actions
            if action_idx in [0, 1]:  # horizontal movement
                action_idx = np.random.choice([2, 3])  # vertical
            else:  # vertical movement
                action_idx = np.random.choice([0, 1])  # horizontal
        
        dx, dy = self.actions[action_idx]
        next_state = (x + dx, y + dy)
        
        if not self.is_valid_state(next_state):
            return state
        return next_state
    
    def get_features(self, state: Tuple[int, int], action_idx: int) -> np.ndarray:
        """Extract features for constraint function"""
        x, y = state
        dx, dy = self.actions[action_idx]
        next_state = (x + dx, y + dy)
        
        features = []
        
        # Feature 1: Wall collision
        collision = not self.is_valid_state(next_state)
        features.append(float(collision))
        
        # Feature 2: Distance to nearest wall
        min_dist = float('inf')
        for wx, wy in self.walls:
            dist = abs(x - wx) + abs(y - wy)
            min_dist = min(min_dist, dist)
        features.append(1.0 / (1.0 + min_dist))
        
        # Feature 3: Moving toward wall
        next_x, next_y = next_state
        current_min_dist = min_dist
        next_min_dist = float('inf')
        for wx, wy in self.walls:
            dist = abs(next_x - wx) + abs(next_y - wy)
            next_min_dist = min(next_min_dist, dist)
        
        moving_toward = float(next_min_dist < current_min_dist and not collision)
        features.append(moving_toward)
        
        # Feature 4: Boundary proximity
        boundary_dist = min(x, y, self.size - 1 - x, self.size - 1 - y)
        features.append(1.0 / (1.0 + boundary_dist))
        
        # Action type features
        action_features = [0.0] * 4
        action_features[action_idx] = 1.0
        features.extend(action_features)
        
        return np.array(features)

class ConstraintLearner:
    """Multi-user Bayesian constraint learning"""
    
    def __init__(self, env, config: ExperimentConfig):
        self.env = env
        self.config = config
        self.feature_dim = None
        self.theta_true = None
        self.theta_map = None
        self.theta_samples = None
        
    def set_true_constraints(self):
        """Set ground truth constraint parameters"""
        # Get feature dimension from a sample
        sample_features = self.env.get_features((0, 0), 0)
        self.feature_dim = len(sample_features)
        
        # True constraint: high cost for collision, medium for approaching danger
        self.theta_true = np.zeros(self.feature_dim)
        self.theta_true[0] = 5.0  # Collision feature
        self.theta_true[2] = 1.0  # Moving toward obstacle feature
        
    def constraint_cost(self, state: Tuple[int, int], action_idx: int, theta: np.ndarray) -> float:
        """Compute constraint cost for state-action pair"""
        features = self.env.get_features(state, action_idx)
        return np.dot(theta, features)
    
    def compute_q_values(self, reward_func, theta: np.ndarray) -> np.ndarray:
        """Compute Q-values using value iteration with constraints"""
        states = [(x, y) for x in range(self.env.size) for y in range(self.env.size)
                 if self.env.is_valid_state((x, y))]
        state_to_idx = {s: i for i, s in enumerate(states)}
        n_states = len(states)
        n_actions = len(self.env.actions)
        
        # Initialize Q-values
        Q = np.zeros((n_states, n_actions))
        
        # Value iteration
        for iteration in range(1000):
            Q_old = Q.copy()
            
            for i, state in enumerate(states):
                for a in range(n_actions):
                    # Get immediate reward and constraint cost
                    reward = reward_func(state, a)
                    constraint = self.constraint_cost(state, a, theta)
                    immediate = reward - self.config.lambda_constraint * constraint
                    
                    # Get next state and its value
                    next_state = self.env.get_next_state(state, a)
                    if next_state in state_to_idx:
                        next_value = np.max(Q[state_to_idx[next_state], :])
                    else:
                        next_value = 0
                    
                    Q[i, a] = immediate + 0.9 * next_value
            
            # Check convergence
            if np.max(np.abs(Q - Q_old)) < self.config.convergence_threshold:
                break
                
        return Q, states, state_to_idx
    
    def generate_user_demonstrations(self, user_id: int, reward_func, theta: np.ndarray, 
                                   n_demos: int) -> List[List[Tuple]]:
        """Generate demonstrations for a user following Boltzmann policy"""
        Q, states, state_to_idx = self.compute_q_values(reward_func, theta)
        
        demonstrations = []
        
        for demo in range(n_demos):
            trajectory = []
            state = (0, 0)  # Start state
            
            for step in range(50):  # Maximum trajectory length
                if state not in state_to_idx:
                    break
                    
                state_idx = state_to_idx[state]
                q_vals = Q[state_idx, :]
                
                # Boltzmann policy
                exp_q = np.exp(self.config.beta_temperature * q_vals)
                probs = exp_q / np.sum(exp_q)
                
                # Sample action
                action = np.random.choice(len(self.env.actions), p=probs)
                
                # Record transition
                trajectory.append((state, action))
                
                # Move to next state
                next_state = self.env.get_next_state(state, action)
                
                # Check if reached goal (user-specific)
                if self.is_user_goal(user_id, next_state):
                    break
                    
                state = next_state
            
            demonstrations.append(trajectory)
        
        return demonstrations
    
    def is_user_goal(self, user_id: int, state: Tuple[int, int]) -> bool:
        """Check if state is goal for specific user"""
        # Different users have different goals
        goal_locations = [
            (9, 9), (9, 0), (0, 9), (7, 7), (2, 8),
            (8, 2), (3, 6), (6, 3), (1, 7), (7, 1)
        ]
        if user_id < len(goal_locations):
            return state == goal_locations[user_id]
        return state == (9, 9)  # Default goal
    
    def log_likelihood(self, theta: np.ndarray, demonstrations: Dict[int, List]) -> float:
        """Compute log-likelihood across all users"""
        total_ll = 0.0
        
        for user_id, demos in demonstrations.items():
            # Get user's reward function
            reward_func = self.get_user_reward_function(user_id)
            
            # Compute Q-values for this user
            Q, states, state_to_idx = self.compute_q_values(reward_func, theta)
            
            # Compute likelihood for this user's demonstrations
            for trajectory in demos:
                for state, action in trajectory:
                    if state in state_to_idx:
                        state_idx = state_to_idx[state]
                        q_vals = Q[state_idx, :]
                        
                        # Boltzmann probability
                        exp_q = np.exp(self.config.beta_temperature * q_vals)
                        probs = exp_q / np.sum(exp_q)
                        
                        # Add log probability
                        if probs[action] > 1e-12:  # Avoid log(0)
                            total_ll += np.log(probs[action])
                        else:
                            total_ll += -50  # Large penalty for impossible actions
        
        return total_ll
    
    def get_user_reward_function(self, user_id: int):
        """Get reward function for specific user"""
        def reward_func(state: Tuple[int, int], action_idx: int) -> float:
            next_state = self.env.get_next_state(state, action_idx)
            
            # Goal reward (user-specific)
            if self.is_user_goal(user_id, next_state):
                return 10.0
            
            # Step penalty for valid moves
            if next_state == state:  # Collision occurred
                return -10.0
            else:
                return -1.0  # Small step cost
        
        return reward_func
    
    def map_estimation(self, demonstrations: Dict[int, List]) -> np.ndarray:
        """Maximum a posteriori estimation of constraint parameters"""
        
        def objective(theta):
            # Negative log posterior
            ll = self.log_likelihood(theta, demonstrations)
            prior = -0.5 * np.sum(theta**2) / self.config.prior_variance
            return -(ll + prior)
        
        # Multiple random initializations to avoid local optima
        best_theta = None
        best_obj = float('inf')
        
        for init in range(5):
            theta_init = np.random.normal(0, 0.1, self.feature_dim)
            
            try:
                result = minimize(objective, theta_init, method='BFGS',
                                options={'maxiter': self.config.max_iterations})
                
                if result.success and result.fun < best_obj:
                    best_obj = result.fun
                    best_theta = result.x
            except:
                continue
        
        if best_theta is None:
            # Fallback initialization
            best_theta = np.zeros(self.feature_dim)
            
        self.theta_map = best_theta
        return best_theta
    
    def mcmc_sampling(self, demonstrations: Dict[int, List], n_samples: int = 1000) -> np.ndarray:
        """MCMC sampling from posterior (simplified Metropolis-Hastings)"""
        
        # Initialize chain
        theta_current = self.theta_map.copy() if self.theta_map is not None else np.zeros(self.feature_dim)
        
        samples = []
        n_accepted = 0
        proposal_std = 0.1
        
        for i in tqdm(range(n_samples + self.config.mcmc_burnin), desc="MCMC Sampling"):
            # Propose new state
            theta_proposed = theta_current + np.random.normal(0, proposal_std, self.feature_dim)
            
            # Compute acceptance probability
            try:
                ll_current = self.log_likelihood(theta_current, demonstrations)
                ll_proposed = self.log_likelihood(theta_proposed, demonstrations)
                
                # Prior terms
                prior_current = -0.5 * np.sum(theta_current**2) / self.config.prior_variance
                prior_proposed = -0.5 * np.sum(theta_proposed**2) / self.config.prior_variance
                
                log_alpha = (ll_proposed + prior_proposed) - (ll_current + prior_current)
                alpha = min(1.0, np.exp(log_alpha))
                
                # Accept or reject
                if np.random.random() < alpha:
                    theta_current = theta_proposed
                    n_accepted += 1
                    
            except:
                # Reject on numerical errors
                pass
            
            # Store samples after burn-in
            if i >= self.config.mcmc_burnin:
                samples.append(theta_current.copy())
        
        print(f"MCMC acceptance rate: {n_accepted / (n_samples + self.config.mcmc_burnin):.3f}")
        
        self.theta_samples = np.array(samples)
        return self.theta_samples

class ExperimentRunner:
    """Run all experiments from the paper"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        
    def run_multi_user_experiment(self) -> Dict:
        """Run multi-user constraint learning experiment"""
        print("Running multi-user experiment...")
        
        # Test different numbers of users
        user_counts = [1, 2, 3, 5, 8]
        total_demos = 50
        results = []
        
        for n_users in user_counts:
            print(f"\nTesting {n_users} users...")
            
            # Run multiple seeds
            seed_results = []
            for seed in range(5):  # Reduced for demo
                np.random.seed(seed)
                
                # Create environment
                env = GridWorld()
                learner = ConstraintLearner(env, self.config)
                learner.set_true_constraints()
                
                # Generate demonstrations
                demos_per_user = total_demos // n_users
                demonstrations = {}
                
                for user_id in range(n_users):
                    reward_func = learner.get_user_reward_function(user_id)
                    user_demos = learner.generate_user_demonstrations(
                        user_id, reward_func, learner.theta_true, demos_per_user)
                    demonstrations[user_id] = user_demos
                
                # Learn constraints
                theta_learned = learner.map_estimation(demonstrations)
                
                # Evaluate
                mse = self.compute_constraint_mse(env, learner.theta_true, theta_learned)
                violation_rate = self.compute_violation_rate(env, theta_learned, demonstrations)
                
                seed_results.append({
                    'mse': mse,
                    'violation_rate': violation_rate,
                    'n_users': n_users,
                    'demos_per_user': demos_per_user
                })
            
            # Aggregate results
            mean_mse = np.mean([r['mse'] for r in seed_results])
            std_mse = np.std([r['mse'] for r in seed_results])
            mean_vr = np.mean([r['violation_rate'] for r in seed_results])
            std_vr = np.std([r['violation_rate'] for r in seed_results])
            
            results.append({
                'n_users': n_users,
                'demos_per_user': demos_per_user,
                'mse_mean': mean_mse,
                'mse_std': std_mse,
                'violation_mean': mean_vr,
                'violation_std': std_vr
            })
            
            print(f"  MSE: {mean_mse:.3f} ± {std_mse:.3f}")
            print(f"  Violation Rate: {mean_vr:.3f} ± {std_vr:.3f}")
        
        return results
    
    def run_baseline_comparison(self) -> Dict:
        """Compare against baseline methods"""
        print("\nRunning baseline comparison...")
        
        methods = ['single_user_best', 'single_user_avg', 'pooled_data', 'our_method']
        environments = ['gridworld', 'stochastic_maze']
        
        results = {}
        
        for env_name in environments:
            print(f"\nTesting {env_name}...")
            
            if env_name == 'gridworld':
                env = GridWorld()
            else:
                env = StochasticMaze()
            
            env_results = {}
            
            for method in methods:
                print(f"  Method: {method}")
                
                seed_results = []
                for seed in range(3):  # Reduced for demo
                    np.random.seed(seed)
                    
                    learner = ConstraintLearner(env, self.config)
                    learner.set_true_constraints()
                    
                    if method == 'our_method':
                        # Multi-user learning
                        demonstrations = {}
                        for user_id in range(self.config.n_users):
                            reward_func = learner.get_user_reward_function(user_id)
                            user_demos = learner.generate_user_demonstrations(
                                user_id, reward_func, learner.theta_true, 
                                self.config.demos_per_user)
                            demonstrations[user_id] = user_demos
                        
                        theta_learned = learner.map_estimation(demonstrations)
                        
                    elif method == 'single_user_best':
                        # Single user with most demonstrations
                        best_mse = float('inf')
                        best_theta = None
                        
                        for user_id in range(self.config.n_users):
                            reward_func = learner.get_user_reward_function(user_id)
                            user_demos = learner.generate_user_demonstrations(
                                user_id, reward_func, learner.theta_true, 
                                self.config.demos_per_user)
                            
                            theta_single = learner.map_estimation({user_id: user_demos})
                            mse = self.compute_constraint_mse(env, learner.theta_true, theta_single)
                            
                            if mse < best_mse:
                                best_mse = mse
                                best_theta = theta_single
                        
                        theta_learned = best_theta
                        
                    elif method == 'single_user_avg':
                        # Average of single-user solutions
                        user_thetas = []
                        
                        for user_id in range(self.config.n_users):
                            reward_func = learner.get_user_reward_function(user_id)
                            user_demos = learner.generate_user_demonstrations(
                                user_id, reward_func, learner.theta_true, 
                                self.config.demos_per_user)
                            
                            theta_single = learner.map_estimation({user_id: user_demos})
                            user_thetas.append(theta_single)
                        
                        theta_learned = np.mean(user_thetas, axis=0)
                        
                    elif method == 'pooled_data':
                        # Pool all demonstrations and treat as single user
                        all_demos = []
                        for user_id in range(self.config.n_users):
                            reward_func = learner.get_user_reward_function(user_id)
                            user_demos = learner.generate_user_demonstrations(
                                user_id, reward_func, learner.theta_true, 
                                self.config.demos_per_user)
                            all_demos.extend(user_demos)
                        
                        # Use first user's reward function for pooled approach
                        theta_learned = learner.map_estimation({0: all_demos})
                    
                    # Evaluate
                    mse = self.compute_constraint_mse(env, learner.theta_true, theta_learned)
                    violation_rate = self.compute_violation_rate(env, theta_learned, 
                                                              {0: all_demos} if method == 'pooled_data' 
                                                              else demonstrations)
                    
                    seed_results.append({
                        'mse': mse,
                        'violation_rate': violation_rate
                    })
                
                # Aggregate
                mean_mse = np.mean([r['mse'] for r in seed_results])
                std_mse = np.std([r['mse'] for r in seed_results])
                mean_vr = np.mean([r['violation_rate'] for r in seed_results])
                std_vr = np.std([r['violation_rate'] for r in seed_results])
                
                env_results[method] = {
                    'mse_mean': mean_mse,
                    'mse_std': std_mse,
                    'violation_mean': mean_vr,
                    'violation_std': std_vr
                }
                
                print(f"    MSE: {mean_mse:.3f} ± {std_mse:.3f}")
                print(f"    Violation Rate: {mean_vr:.3f} ± {std_vr:.3f}")
            
            results[env_name] = env_results
        
        return results
    
    def run_safety_compositionality(self) -> Dict:
        """Test safety against malicious users"""
        print("\nRunning safety compositionality experiment...")
        
        env = GridWorld()
        learner = ConstraintLearner(env, self.config)
        learner.set_true_constraints()
        
        # Learn constraints from trusted users
        trusted_demonstrations = {}
        for user_id in range(5):  # 5 trusted users
            reward_func = learner.get_user_reward_function(user_id)
            user_demos = learner.generate_user_demonstrations(
                user_id, reward_func, learner.theta_true, self.config.demos_per_user)
            trusted_demonstrations[user_id] = user_demos
        
        theta_learned = learner.map_estimation(trusted_demonstrations)
        
        # Create malicious user with adversarial reward
        def malicious_reward_func(state: Tuple[int, int], action_idx: int) -> float:
            """Reward function that encourages dangerous behavior"""
            next_state = env.get_next_state(state, action_idx)
            
            # Reward collisions and dangerous moves
            if next_state == state:  # Collision
                return 10.0  # High reward for collisions!
            
            # Reward moving toward obstacles
            features = env.get_features(state, action_idx)
            if features[2] > 0:  # Moving toward obstacle
                return 5.0
            
            return -1.0  # Penalty for safe moves
        
        # Test different constraint weights
        constraint_weights = [0.0, 1.0, 1.5]  # 0.0 = vanilla IRL
        results = {}
        
        for lambda_val in constraint_weights:
            original_lambda = self.config.lambda_constraint
            self.config.lambda_constraint = lambda_val
            
            # Generate malicious demonstrations
            if lambda_val == 0.0:
                # Vanilla IRL: no constraints
                malicious_demos = learner.generate_user_demonstrations(
                    999, malicious_reward_func, np.zeros(learner.feature_dim), 20)
            else:
                # Our method: use learned constraints
                malicious_demos = learner.generate_user_demonstrations(
                    999, malicious_reward_func, theta_learned, 20)
            
            # Compute violation rate
            violation_rate = self.compute_violation_rate(env, theta_learned, {999: malicious_demos})
            
            results[f'lambda_{lambda_val}'] = violation_rate
            print(f"  λ={lambda_val}: Violation rate = {violation_rate:.3f}")
            
            # Restore original lambda
            self.config.lambda_constraint = original_lambda
        
        return results
    
    def compute_constraint_mse(self, env, theta_true: np.ndarray, theta_learned: np.ndarray) -> float:
        """Compute MSE between true and learned constraint functions"""
        mse = 0.0
        count = 0
        
        for x in range(env.size):
            for y in range(env.size):
                if env.is_valid_state((x, y)):
                    for action_idx in range(len(env.actions)):
                        features = env.get_features((x, y), action_idx)
                        true_cost = np.dot(theta_true, features)
                        learned_cost = np.dot(theta_learned, features)
                        mse += (true_cost - learned_cost)**2
                        count += 1
