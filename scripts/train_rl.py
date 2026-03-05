"""
HydroTwin OS — Plane 3: RL Training Script

Trains a Soft Actor-Critic (SAC) agent on the DataCenterEnv and saves the model.
"""

import os
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from hydrotwin.env.datacenter_env import DataCenterEnv
from hydrotwin.env.scenarios import NormalOps

def main(total_timesteps: int, alpha: float, beta: float, gamma: float, delta: float, model_path: str):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print(f"Training SAC Agent with weights: α={alpha}, β={beta}, γ={gamma}, δ={delta}")

    # Set up environment with custom reward weights
    # We specify weights via the facility_config dict, which gets passed to the environment
    # Actually, we can update the env's reward function directly or via config.
    # PareReward is created inside DataCenterEnv if not passed. Let's pass it via facility_config if supported.
    from hydrotwin.reward.pareto_reward import ParetoReward, RewardWeights, DynamicAdjustmentConfig
    
    # We want a fixed base weight for training maybe, but dynamic config is on by default.
    # To isolate weights, we will disable dynamic config for this specific run if needed.
    reward_fn = ParetoReward(
        base_weights=RewardWeights(alpha=alpha, beta=beta, gamma=gamma, delta=delta),
        dynamic_config=DynamicAdjustmentConfig(enabled=True)
    )

    def make_env():
        env = DataCenterEnv(scenario=NormalOps(), max_episode_steps=1440)
        env.reward_fn = reward_fn  # override default reward_fn
        return Monitor(env)

    env = make_env()
    eval_env = make_env()

    # Callbacks
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=os.path.dirname(model_path),
        log_path=os.path.dirname(model_path), 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )

    model = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4, batch_size=256, buffer_size=100000)
    
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    
    # Save the final model
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="models/sac_nexus_agent")
    
    args = parser.parse_args()
    main(args.timesteps, args.alpha, args.beta, args.gamma, args.delta, args.output)
