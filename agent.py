import OOS_env
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
import os


def train(input_dir):
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    env = OOS_env.OOSenv(input_dir)

    """
    learning_rate = 1e-3
    gamma = 0.99
    clip_range = 0.2
    ent_coef = 1e-7
    vf_coef = 0.5
    
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=1
    )
    """

    model = MaskablePPO("MultiInputPolicy", env, verbose=1)

    TIMESTEPS = 50_000
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)  # Train the model
        model.save(f"{model_dir}/maskPPO_{TIMESTEPS*iters}")  # Save the model every TIMESTEPS


def test(model_timesteps, input_dir, render=True):
    env = gym.make('OOS-maintenance-v0', input_directory=input_dir, render_mode=None)

    # Load Model
    model = MaskablePPO.load(f"models/maskPPO_{model_timesteps}", env=env)
    rewards = 0

    # Run a test
    obs, info = env.reset()
    terminated = False
    env.render()

    while True:
        action_masks = get_action_masks(env)
        action, _ = model.predict(observation=obs, deterministic=True, action_masks=action_masks)  # Turn on deterministic so predict always returns the same behavior
        obs, reward, terminated, truncated, info = env.step(action)
        rewards += reward
        env.render()

        if terminated or truncated:
            print(info)
            break

    print(f"---   Total rewards : {rewards}   ---")


if __name__ == "__main__":

    case_study = "30d_study_case"
    input_directory = (
        "/Users/nathanclaret/Desktop/thesis/code/data_studycase/"
        + case_study
    )

    train(input_directory)
    #test(10000000, input_directory)

    """
    env = gym.make('OOS-maintenance-v0', input_directory=input_directory, render_mode=None)
    env.reset()
    env.current_time = 1
    print("\n")
    print(env.current_time)
    print(env.A_Rt[env.current_time])
    print(env.unwrapped.get_refuel_positions())
    #print(env.maint_params)
    print("\n")
    """
