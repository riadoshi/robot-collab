#!/usr/bin/env python3
"""
Clean demo generation script for the sweeping task.
Generates demonstrations with fixed cube order (red, green, blue) but randomized cube positions.
"""

import os
# --- Headless EGL setup ---
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


import sys
import numpy as np
import argparse
from pathlib import Path
import imageio
from tqdm import tqdm
import importlib.util

# Import parser module directly to avoid OpenAI key check in prompting/__init__.py
parser_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompting", "parser.py")
spec = importlib.util.spec_from_file_location("parser_module", parser_path)
parser_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser_module)
LLMResponseParser = parser_module.LLMResponseParser

from rocobench.envs.task_sweep import SweepTask
from rocobench.policy import PlannedPathPolicy


def generate_action_string(alice_action, bob_action):
    """Generate properly formatted action string for parser."""
    return f"""EXECUTE
NAME Alice ACTION {alice_action}
NAME Bob ACTION {bob_action}
"""


def execute_action_plan(env, parser, obs, alice_action, bob_action, timeout=1500):
    """Execute a single action plan for both robots."""
    action_str = generate_action_string(alice_action, bob_action)
    
    # Parse the action
    success, reason, path_plans = parser.parse(obs, action_str)
    if not success:
        print(f"Parse failed: {reason}")
        return False, None
    
    # Execute each path plan
    all_frames = []
    for path_plan in path_plans:
        policy = PlannedPathPolicy(
            physics=env.physics,
            robots=env.get_sim_robots(),
            path_plan=path_plan,
            control_freq=20,
            graspable_object_names=env.get_graspable_objects(),
            allowed_collision_pairs=env.get_allowed_collision_pairs(),
            timeout=timeout,
            skip_smooth_path=False,
        )
        
        # Plan the motion
        plan_success, plan_reason = policy.plan(env)
        if not plan_success:
            print(f"Planning failed: {plan_reason}")
            return False, None, obs
        
        # Execute the plan
        while not policy.plan_exhausted:
            action = policy.act(obs, env.physics)
            obs = env.step(action)
            all_frames.append(env.physics.render(camera_id='video', height=480, width=640))
    
    return True, all_frames, obs


def generate_demo(env, parser, demo_idx, save_dir, timeout=1500):
    """Generate a single demo with fixed cube order (red, green, blue)."""
    print(f"\n{'='*60}")
    print(f"Generating demo {demo_idx}")
    print(f"{'='*60}")
    
    # Reset environment (this randomizes cube positions via sample_initial_scene)
    obs = env.reset()
    all_frames = []
    demo_success = True
    
    # Use fixed cube order: red, green, blue
    cube_names = ['red_cube', 'green_cube', 'blue_cube']
    
    print(f"Cube order: {cube_names}")
    
    # Helper function to save video on failure
    def save_video_on_failure():
        if len(all_frames) > 0:
            video_path = save_dir / f"demo_{demo_idx:03d}_FAILED.mp4"
            imageio.mimsave(video_path, all_frames, fps=30)
            print(f"Saved partial video to {video_path}")
    
    try:
        # Sweep each cube
        for i, cube_name in enumerate(cube_names):
            print(f"\n--- Round {i+1}/{len(cube_names)}: Sweeping {cube_name} ---")
            
            # Step 1: Alice MOVEs to cube (positions dustpan), Bob WAITs
            print(f"  Alice positioning dustpan for {cube_name}...")
            success, frames, obs = execute_action_plan(
                env, parser, obs, 
                f"MOVE {cube_name}", 
                "WAIT",
                timeout=timeout
            )
            
        #     if not success:
        #         print(f"  Failed to position dustpan for {cube_name}")
        #         save_video_on_failure()
        #         return False
            
            all_frames.extend(frames)
            
            obs = env.get_obs()
            
        #     # Step 2: Bob MOVEs to cube, Alice WAITs
        #     print(f"  Bob moving to {cube_name}...")
        #     success, frames = execute_action_plan(
        #         env, parser, obs,
        #         "WAIT",
        #         f"MOVE {cube_name}",
        #         timeout=timeout
        #     )
            
        #     if not success:
        #         print(f"  Failed to move Bob to {cube_name}")
        #         save_video_on_failure()
        #         return False
            
        #     all_frames.extend(frames)
        #     obs = env.get_obs()
            
        #     # Step 3: Alice WAITs, Bob SWEEPs
        #     print(f"  Bob sweeping {cube_name}...")
        #     success, frames = execute_action_plan(
        #         env, parser, obs,
        #         "WAIT",
        #         f"SWEEP {cube_name}",
        #         timeout=timeout
        #     )
            
        #     if not success:
        #         print(f"  Failed to sweep {cube_name}")
        #         save_video_on_failure()
        #         return False
            
        #     all_frames.extend(frames)
        #     obs = env.get_obs()
            
        #     # Check if cube is in dustpan
        #     cube_state = obs.objects[cube_name]
        #     if 'dustpan_bottom' in cube_state.contacts:
        #         print(f"  ✓ {cube_name} successfully swept into dustpan")
        #     else:
        #         print(f"  ⚠ {cube_name} may not be fully in dustpan")
        
        # # Step 3: Alice DUMPs, Bob WAITs
        # print(f"\n--- Final Step: Dumping into trash bin ---")
        # success, frames = execute_action_plan(
        #     env, parser, obs,
        #     "DUMP",
        #     "WAIT",
        #     timeout=timeout
        # )
        
        # if not success:
        #     print(f"  Failed to dump")
        #     save_video_on_failure()
        #     return False
        
        # all_frames.extend(frames)
        # obs = env.get_obs()
    
    except Exception as e:
        print(f"Exception occurred: {e}")
        save_video_on_failure()
        raise
    
    # Check reward
    reward, done = env.get_reward_done(obs)
    print(f"\n{'='*60}")
    print(f"Demo {demo_idx} complete!")
    print(f"  Success: {done}")
    print(f"  Reward: {reward}")
    print(f"  Total frames: {len(all_frames)}")
    print(f"{'='*60}")
    
    # Save video (success or partial success)
    if len(all_frames) > 0:
        if done:
            video_path = save_dir / f"demo_{demo_idx:03d}.mp4"
        else:
            video_path = save_dir / f"demo_{demo_idx:03d}_INCOMPLETE.mp4"
        imageio.mimsave(video_path, all_frames, fps=30)
        print(f"Saved video to {video_path}")
    
    return done


def main():
    parser_args = argparse.ArgumentParser()
    parser_args.add_argument("--num_demos", "--n", type=int, default=10, help="Number of demos to generate")
    parser_args.add_argument("--save_dir", type=str, default="demos_sweep", help="Directory to save demos")
    parser_args.add_argument("--seed_start", type=int, default=0, help="Starting seed")
    parser_args.add_argument("--timeout", type=int, default=1500, help="RRT timeout per action")
    args = parser_args.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Generating {args.num_demos} demos")
    print(f"Save directory: {save_dir}")
    print(f"Cube order: red, green, blue (fixed)")
    print(f"Cube positions: randomized")
    
    success_count = 0
    
    for demo_idx in range(args.num_demos):
        seed = args.seed_start + demo_idx
        
        # Initialize environment
        env = SweepTask(
            filepath="rocobench/envs/task_sweep.xml",
            np_seed=seed,
        )
        
        # Initialize parser
        parser = LLMResponseParser(
            env=env,
            llm_output_mode="action",
            robot_agent_names={
                "ur5e_robotiq": "Alice",
                "panda": "Bob",
            },
            response_keywords=["NAME", "ACTION"],
            direct_waypoints=3,
        )
        
        # Generate demo
        try:
            success = generate_demo(
                env, 
                parser, 
                demo_idx, 
                save_dir,
                timeout=args.timeout,
            )
            if success:
                success_count += 1
        except Exception as e:
            print(f"Demo {demo_idx} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"  Successful demos: {success_count}/{args.num_demos}")
    print(f"  Success rate: {success_count/args.num_demos*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

