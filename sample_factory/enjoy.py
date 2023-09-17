import os
import time
from collections import deque
from typing import Dict, Tuple

import joblib
import gym
import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.algo.utils.hierarchical_utils import MetaController
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.huggingface.huggingface_utils import generate_model_card, generate_replay_video, push_to_hf
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import debug_log_every_n, experiment_dir, log


def visualize_policy_inputs(normalized_obs: Dict[str, Tensor]) -> None:
    """
    Display actual policy inputs after all wrappers and normalizations using OpenCV imshow.
    """
    import cv2

    if "obs" not in normalized_obs.keys():
        return

    obs = normalized_obs["obs"]
    # visualize obs only for the 1st agent
    obs = obs[0]
    if obs.dim() != 3:
        # this function is only for RGB images
        return

    # convert to HWC
    obs = obs.permute(1, 2, 0)
    # convert to numpy
    obs = obs.cpu().numpy()
    # convert to uint8
    obs = cv2.normalize(
        obs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1
    )  # this will be different frame-by-frame but probably good enough to give us an idea?
    scale = 5
    obs = cv2.resize(obs, (obs.shape[1] * scale, obs.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("policy inputs", obs)
    cv2.waitKey(delay=1)


def render_frame(cfg, env, video_frames, num_episodes, last_render_start) -> float:
    render_start = time.time()

    if cfg.save_video:
        need_video_frame = len(video_frames) < cfg.video_frames or cfg.video_frames < 0 and num_episodes == 0
        if need_video_frame:
            frame = env.render()
            if frame is not None:
                video_frames.append(frame.copy())
    else:
        if not cfg.no_render:
            target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
            current_delay = render_start - last_render_start
            time_wait = target_delay - current_delay

            if time_wait > 0:
                # log.info("Wait time %.3f", time_wait)
                time.sleep(time_wait)

            try:
                env.render()
            except (gym.error.Error, TypeError) as ex:
                debug_log_every_n(1000, f"Exception when calling env.render() {str(ex)}")

    return render_start


def recursive_copy(data):
    if isinstance(data, np.ndarray):
        return data.copy()
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().copy()
    elif type(data) is list:
        return [recursive_copy(x) for x in data]
    elif type(data) is dict:
        return {key: recursive_copy(value) for key, value in data.items()}
    else:
        return data
    

def get_agent_obs(obs, agent_i):
    if type(obs) is dict:
        return { key: value[agent_i] for key, value in obs.items() }
    else:
        return obs[agent_i]


def get_stacked(inpt, key):
    return torch.stack([_inpt[key] for _inpt in inpt], 0)


def enjoy(cfg: Config) -> Tuple[StatusCode, float]:
    verbose = False

    cfg = load_from_checkpoint(cfg)

    hierarchical = cfg.hierarchical_rl

    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")

    cfg.num_envs = 1

    render_mode = "human"
    if cfg.save_video:
        render_mode = "rgb_array"
    elif cfg.no_render:
        render_mode = None

    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
    )
    env_info = extract_env_info(env, cfg)

    if hierarchical:
        meta_controller = MetaController(cfg, env_info)
    else:
        meta_controller = None

    if hasattr(env.unwrapped, "reset_on_init"):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")

    actor_critics = []

    for agent_i in range(env.num_agents):
        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space, env.num_rewards)
        actor_critic.eval()
        actor_critic.model_to_device(device)
        name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, agent_i), f"{name_prefix}_*")
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict["model"])
        actor_critics.append(actor_critic)

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    true_objectives = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0

    episode_datas = [[] for _ in range(env.num_agents)]
    episode_data = [dict(obs=[], normalized_obs=[], actions=[], values=[], original_actions=[],
                         denormed_values=[], rewards=[], terminated=[], truncated=[], 
                         infos=[], actual_acting=[]) for _ in range(env.num_agents)]

    last_render_start = time.time()

    def max_frames_reached(frames):
        return cfg.max_num_frames is not None and frames > cfg.max_num_frames

    reward_list = []

    obs, infos = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = None
    finished_episode = [False for _ in range(env.num_agents)]

    for i in range(env.num_agents):
        episode_data[i]["obs"].append(recursive_copy(get_agent_obs(obs, i)))
        episode_data[i]["infos"].append(recursive_copy(infos[i]))

    video_frames = []
    num_episodes = 0

    with torch.no_grad():
        while not max_frames_reached(num_frames):
            normalized_obss = [prepare_and_normalize_obs(actor_critic, obs) for i, actor_critic in enumerate(actor_critics)]
            normalized_obss = [{key: value[i: i + 1] for key, value in normalized_obs.items()} for normalized_obs in normalized_obss]

            if not cfg.no_render:
                visualize_policy_inputs(normalized_obss[0])

            policy_outputs = [actor_critic(normalized_obs, rnn_states) for actor_critic, normalized_obs in zip(actor_critics, normalized_obss)]
            for agent_i in range(env.num_agents):
                policy_outputs[agent_i] = { key: value[0] for key, value in policy_outputs[agent_i].items() }

            if hierarchical:
                for agent_i in range(env.num_agents):
                    meta_controller.set_policy_outputs(agent_i, policy_outputs[agent_i])

            # sample actions from the distribution by default
            actions = get_stacked(policy_outputs, "actions")
            values = get_stacked(policy_outputs, "values")
            denormed_values = get_stacked(policy_outputs, "denormalized_values")

            if cfg.eval_deterministic:
                action_distributions = [actor_critic.action_distribution() for actor_critic in actor_critics]
                actions = [argmax_actions(action_distribution) for action_distribution in action_distributions]
                actions = np.stack(actions, 0)

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            if hierarchical:
                for i in range(env.num_agents):
                    episode_data[i]["original_actions"].append(recursive_copy(actions[i]))
                actions = meta_controller.process_actions(actions)

            rnn_states = get_stacked(policy_outputs, "new_rnn_states")

            for _ in range(render_action_repeat):
                last_render_start = render_frame(cfg, env, video_frames, num_episodes, last_render_start)

                for i in range(env.num_agents):
                    episode_data[i]["normalized_obs"].append(recursive_copy(normalized_obss[i]))
                    episode_data[i]["actions"].append(recursive_copy(actions[i]))
                    episode_data[i]["values"].append(recursive_copy(values[i]))
                    episode_data[i]["denormed_values"].append(recursive_copy(denormed_values[i]))
                    episode_data[i]['actual_acting'].append(meta_controller.policy_idx == i)

                obs, rew, terminated, truncated, infos = env.step(actions)
                dones = make_dones(terminated, truncated)
                infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

                for i in range(env.num_agents):
                    episode_data[i]["obs"].append(recursive_copy(get_agent_obs(obs, i)))
                    episode_data[i]["rewards"].append(recursive_copy(rew[i]))
                    episode_data[i]["terminated"].append(recursive_copy(terminated[i]))
                    episode_data[i]["truncated"].append(recursive_copy(truncated[i]))
                    episode_data[i]["infos"].append(recursive_copy(infos[i]))

                if episode_reward is None:
                    episode_reward = rew.float().clone()
                else:
                    episode_reward += rew.float()

                num_frames += 1
                if num_frames % 100 == 0:
                    log.debug(f"Num frames {num_frames}...")

                dones = dones.cpu().numpy()
                for agent_i, done_flag in enumerate(dones):
                    if done_flag:
                        finished_episode[agent_i] = True
                        rew = episode_reward[agent_i].cpu().numpy()
                        episode_rewards[agent_i].append(rew.copy())

                        episode_datas[agent_i].append(recursive_copy(episode_data[agent_i]))
                        for key in episode_data[agent_i].keys():
                            episode_data[agent_i][key] = []

                        true_objective = rew
                        if isinstance(infos, (list, tuple)):
                            true_objective = infos[agent_i].get("true_objective", rew)
                        true_objectives[agent_i].append(true_objective.copy())

                        if verbose:
                            log.info(
                                "Episode finished for agent %d at %d frames. Reward: %.3f, true_objective: %.3f",
                                agent_i,
                                num_frames,
                                episode_reward[agent_i],
                                true_objectives[agent_i][-1],
                            )
                        rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                        episode_reward[agent_i] = 0

                        if cfg.use_record_episode_statistics:
                            # we want the scores from the full episode not a single agent death (due to EpisodicLifeEnv wrapper)
                            if "episode" in infos[agent_i].keys():
                                num_episodes += 1
                                reward_list.append(infos[agent_i]["episode"]["r"])
                        else:
                            num_episodes += 1
                            reward_list.append(true_objective)

                # if episode terminated synchronously for all agents, pause a bit before starting a new one
                if all(dones):
                    render_frame(cfg, env, video_frames, num_episodes, last_render_start)
                    time.sleep(0.05)

                if all(finished_episode):
                    finished_episode = [False] * env.num_agents
                    avg_episode_rewards_str, avg_true_objective_str = "", ""
                    for agent_i in range(env.num_agents):
                        avg_rew = np.mean(episode_rewards[agent_i], 0)
                        avg_true_obj = np.mean(true_objectives[agent_i], 0)

                        if not any(np.isnan(avg_rew)):
                            if avg_episode_rewards_str:
                                avg_episode_rewards_str += ", "
                            avg_episode_rewards_str += f"#{agent_i}: {avg_rew}"
                        if not any(np.isnan(avg_true_obj)):
                            if avg_true_objective_str:
                                avg_true_objective_str += ", "
                            avg_true_objective_str += f"#{agent_i}: {avg_true_obj}"

                    log.info(
                        "Avg episode rewards: %s, true rewards: %s", avg_episode_rewards_str, avg_true_objective_str
                    )
                    # log.info(
                    #     "Avg episode reward: %s, avg true_objective: %s",
                    #     f"{np.mean([np.mean(episode_rewards[i], 0) for i in range(env.num_agents)])}",
                    #     f"{np.mean([np.mean(true_objectives[i], 0) for i in range(env.num_agents)])}",
                    # )

                # VizDoom multiplayer stuff
                # for player in [1, 2, 3, 4, 5, 6, 7, 8]:
                #     key = f'PLAYER{player}_FRAGCOUNT'
                #     if key in infos[0]:
                #         log.debug('Score for player %d: %r', player, infos[0][key])

            if num_episodes >= cfg.max_num_episodes:
                break

    env.close()

    for agent_i in range(env.num_agents):
        acting_freqs = []
        episode_lens = []
        for episode_i in range(len(episode_datas[agent_i])):
            episode_len = len(episode_datas[agent_i][episode_i]['actual_acting'])
            episode_lens.append(episode_len)
            if episode_len > 0:
                total_actings = sum(episode_datas[agent_i][episode_i]['actual_acting'])
                acting_freqs.append(total_actings / episode_len)
        log.info(f"agent #{agent_i} acting frequency: {np.mean(acting_freqs)} episode lens: {np.mean(episode_lens)}")
        print(acting_freqs)

    for agent_i in range(env.num_agents):
        for episode_i in range(len(episode_rewards[agent_i])):
            log.info(f"agent #{agent_i} episode #{episode_i}: {episode_rewards[agent_i][episode_i]}")

    episode_data_path = os.path.join(experiment_dir(cfg=cfg), "episode_data.joblib")
    joblib.dump(episode_datas, episode_data_path)

    if cfg.save_video:
        if cfg.fps > 0:
            fps = cfg.fps
        else:
            fps = 30
        generate_replay_video(experiment_dir(cfg=cfg), video_frames, fps, cfg)

    if cfg.push_to_hub:
        generate_model_card(
            experiment_dir(cfg=cfg),
            cfg.algo,
            cfg.env,
            cfg.hf_repository,
            reward_list,
            cfg.enjoy_script,
            cfg.train_script,
        )
        push_to_hf(experiment_dir(cfg=cfg), cfg.hf_repository)

    return ExperimentStatus.SUCCESS, sum([sum(episode_rewards[i]) for i in range(env.num_agents)]) / sum(
        [len(episode_rewards[i]) for i in range(env.num_agents)]
    )
