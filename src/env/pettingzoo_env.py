from functools import lru_cache
import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils import wrappers
# from pettingzoo.utils.conversions import from_parallel_wrapper
from pettingzoo.utils.env import ParallelEnv

from .commons_env import HarvestCommonsEnv


def parallel_env(**ssd_args):
    return _parallel_env(**ssd_args)


class ssd_parallel_env(ParallelEnv):
    render_mode = "human"
    def __init__(self, env, ep_length, penalty):
        self.ssd_env = env
        self.ep_length = ep_length
        self.penalty=penalty
        self.possible_agents = list(self.ssd_env.agents.keys())
        self.ssd_env.reset()
        self.observation_space = lru_cache(maxsize=None)(lambda agent_id: env.observation_space)
        self.observation_spaces = {agent: env.observation_space for agent in self.possible_agents}
        self.action_space = lru_cache(maxsize=None)(lambda agent_id: env.action_space)
        self.action_spaces = {agent: env.action_space for agent in self.possible_agents}

    def reset(self, seed=None, **kwargs):
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        self.dones = {agent: False for agent in self.agents}
        return self.ssd_env.reset()

    def seed(self, seed=None):
        return self.ssd_env.seed(seed)

    def render(self, render_mode="human"):
        return self.ssd_env.render(mod=render_mode)

    def close(self):
        self.ssd_env.close()

    def step(self, actions):
        obss, rews, self.dones, infos = self.ssd_env.step(actions)
        del self.dones["__all__"]
        self.num_cycles += 1
        
        if self.penalty: # simple punishment for fire action
            rews = {agent_id: -1 if infos[agent_id]['fire'] else r for agent_id, r in rews.items()}
            
        # if rewards metric is not 0, zero all rewards for later insert desired rewards
        if self.ssd_env.metric != 'Efficiency':
            for k in rews.keys():
                rews[k] = 0
        if self.num_cycles >= self.ep_length:
            self.dones = {agent: True for agent in self.agents}
            self.ssd_env.compute_social_metrics()
            for k in infos.keys():
                infos[k]['metrics'] = self.ssd_env.get_social_metrics()
                
            # inser desired rewards at tghe end if episode
            if self.ssd_env.metric == 'Efficiency*Peace':  # eff * global peace
                for k in rews.keys():
                    rews[k] = infos[k]['metrics']['efficiency'] * infos[k]['metrics']['peace']
            elif self.ssd_env.metric == 'Efficiency*Peace*Equality':  # eff * eq * global peace
                for k in rews.keys():
                    rews[k] = infos[k]['metrics']['efficiency'] * infos[k]['metrics']['peace'] * infos[k]['metrics']['equality']
            self.agents = [agent for agent in self.agents if not self.dones[agent]]
        return obss, rews, self.dones, self.dones, infos

    def get_images(self):
         self.ssd_env.full_map_to_colors()
    
    def get_social_metrics(self):
        return self.ssd_env.get_social_metrics()

    
class _parallel_env(ssd_parallel_env, EzPickle):
    metadata = {"render.modes": ["human", "rgb_array"],
                "name": "custom_environment_v0"
                }

    def __init__(self, ep_length=600, penalty=False, **ssd_args):
        EzPickle.__init__(self, ep_length, penalty, **ssd_args)
        env = HarvestCommonsEnv(**ssd_args)
        super().__init__(env, ep_length, penalty)