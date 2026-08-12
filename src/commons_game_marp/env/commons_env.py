import numpy as np
import gymnasium
# from social_dilemmas.envs
from .commons_agent import HarvestCommonsAgent, HARVEST_DEFAULT_VIEW_SIZE
from .map_env import MapEnv, ACTIONS
from .maps import SMALL_HARVEST_MAP, MEDIUM_HARVEST_MAP, HARVEST_MAP_LARGER, HARVEST_MAP
APPLE_RADIUS = 2

# Add custom actions to the agent
ACTIONS['FIRE'] = 7  # length of firing range

SPAWN_PROB_SLOW = [0, 0.005, 0.02, 0.05] #github
SPAWN_PROB_FAST = [0, 0.01, 0.05, 0.1] # paper
OUTCAST_POSITION = -99




MAP = {"small": SMALL_HARVEST_MAP,
       "medium": MEDIUM_HARVEST_MAP}


class HarvestCommonsEnv(MapEnv):

    def __init__(self, ascii_map=HARVEST_MAP, num_agents=1, render=False, agent_view_range=HARVEST_DEFAULT_VIEW_SIZE,
                 color_map=None, ep_length=600, spawn_speed='slow', metric="Efficiency", penalty=False):
        self.ep_length = ep_length
        self.apple_points = []
        self.agent_view_range = agent_view_range
        self.spawn_speed = SPAWN_PROB_SLOW if spawn_speed=="slow" else SPAWN_PROB_FAST
        self.metric=metric
        self.penalty = penalty

        super().__init__(ascii_map, num_agents, render, color_map=color_map)

        self.rewards_record = {}
        self.timeout_record = {}

        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'A':
                    self.apple_points.append([row, col])

        self.rewards_record = {}
        self.timeout_record = {}
        self.fire_counter = 0
        self.fire_sucsses = 0
        self.metrics = {"efficiency": 0,
                "equality": 0,
                "sustainability": 0,
                "peace": 0,
                "fire_attempts": 0,
                "fire_sucsses": 0}
    @property
    def action_space(self):
        agents = list(self.agents.values())
        return agents[0].action_space

    @property
    def observation_space(self):
        agents = list(self.agents.values())
        
        obs_space = {
            "curr_obs": gymnasium.spaces.Box(
                low=0,
                high=255,
                shape = agents[0].observation_space.shape,
                dtype=np.uint8,
            )
        }
        return obs_space

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)

        # Social metrics must reflect true environment rewards, so record them
        # before the FIRE penalty is applied. Applying the penalty first would
        # corrupt efficiency and equality -- the same quantities the reward
        # model is trained to predict.
        env_rewards = dict(rewards)

        for agent_id, _ in self.agents.items():
            # infos[agent_id]['r'] is always the true environment reward; the
            # returned `rewards` dict may instead carry the -1 FIRE penalty
            # applied below. Consumers that need the unpenalized reward (e.g.
            # social metrics, reward-model preference data) must read `r`.
            infos[agent_id]['r'] = env_rewards[agent_id]
            infos[agent_id]['fire'] = action[agent_id] == 7
            self.fire_counter += int(action[agent_id] == 7)

        self.update_social_metrics(env_rewards)

        if self.penalty:
            for agent_id, _ in self.agents.items():
                if action[agent_id] == 7:
                    rewards[agent_id] = -1

        return observations, rewards, dones, infos

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = HarvestCommonsAgent(agent_id, spawn_point, rotation, grid, lateral_view_range=self.agent_view_range,
                                        frontal_view_range=self.agent_view_range)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        # reset social metrics
        self.metrics = {"efficiency": 0,
                        "equality": 0,
                        "sustainability": 0,
                        "peace": 0,
                        "fire_attempts": 0,
                        "fire_sucsses": 0}
        
        for apple_point in self.apple_points:
            self.world_map[apple_point[0], apple_point[1]] = 'A'
   
    def custom_action(self, agent, action):
        agent.fire_beam('F')
        updates = self.update_map_fire(agent.get_pos().tolist(),
                                       agent.get_orientation(),
                                       ACTIONS['FIRE'], fire_char='F')
        return updates

    def custom_map_update(self):
        "See parent class"
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)

        # Outcast timed-out agents
        for agent_id, agent in self.agents.items():
            if agent.remaining_timeout > 0:
                agent.remaining_timeout -= 1
                # print("Agent %s its on timeout for %d n_steps" % (agent_id, agent.remaining_timeout))
                if not np.any(agent.pos == OUTCAST_POSITION):
                    self.update_map([[agent.pos[0], agent.pos[1], ' ']])
                    agent.pos = np.array([OUTCAST_POSITION, OUTCAST_POSITION])
            # Return agent to environment
            if agent.remaining_timeout == 0 and np.any(agent.pos == OUTCAST_POSITION):
                # print("%s has finished timeout" % agent_id)
                spawn_point = self.spawn_point()
                spawn_rotation = self.spawn_rotation()
                agent.update_agent_pos(spawn_point)
                agent.update_agent_rot(spawn_rotation)

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in self.agent_pos and self.world_map[row, col] != 'A':
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j ** 2 + k ** 2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if 0 <= x + j < self.world_map.shape[0] and \
                                    self.world_map.shape[1] > y + k >= 0:
                                symbol = self.world_map[x + j, y + k]
                                if symbol == 'A':
                                    num_apples += 1

                spawn_prob = self.spawn_speed[min(num_apples, 3)]
                rand_num = np.random.rand(1)[0]
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, 'A'))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get('A', 0)
        return num_apples

    def update_social_metrics(self, rewards):
        # Save a record of rewards by agent as they are needed for the social metrics computation
        for agent_id, reward in rewards.items():
            if agent_id in self.rewards_record.keys():
                self.rewards_record[agent_id].append(reward)
            else:
                self.rewards_record[agent_id] = [reward]

            is_agent_in_timeout = True if self.agents[agent_id].remaining_timeout > 0 else False
            self.fire_sucsses += int(self.agents[agent_id].remaining_timeout==self.agents[agent_id].TIMEOUT_TIME-1)
            if agent_id in self.timeout_record.keys():
                self.timeout_record[agent_id].append(is_agent_in_timeout)
            else:
                self.timeout_record[agent_id] = [is_agent_in_timeout]

    def compute_social_metrics(self):
        if len(self.rewards_record) < 1:
            return None

        # Compute sum of rewards
        sum_of_rewards = dict(zip(self.agents.keys(), [0] * self.num_agents))
        for agent_id, rewards in self.rewards_record.items():
            sum_of_rewards[agent_id] = np.sum(rewards)

        agents_sum_rewards = np.sum(list(sum_of_rewards.values()))

        # Compute efficiency/sustainability
        efficiency = agents_sum_rewards / self.num_agents

        # Compute Equality (Gini Coefficient)
        sum_of_diff = 0
        for agent_id_a, rewards_sum_a in sum_of_rewards.items():
            for agent_id_b, rewards_sum_b in sum_of_rewards.items():
                sum_of_diff += np.abs(rewards_sum_a - rewards_sum_b)

        agents_sum_rewards = agents_sum_rewards if agents_sum_rewards != 0 else 1
        equality = 1 - sum_of_diff / (2 * self.num_agents * (agents_sum_rewards))

        # Compute sustainability metric (Average time of at which rewards were collected)
        avg_time = 0
        for agent_id, rewards in self.rewards_record.items():
            pos_reward_time_steps = np.argwhere(np.array(rewards) > 0)
            if pos_reward_time_steps.size != 0:
                avg_time += np.mean(pos_reward_time_steps)

        sustainability = avg_time / (self.num_agents * self.ep_length)

        # Compute peace metric
        timeout_steps = 0
        for agent_id, peace_record in self.timeout_record.items():
            timeout_steps += np.sum(peace_record)
        peace = (self.num_agents * self.ep_length - timeout_steps) / (self.num_agents * self.ep_length)
        metrics = {"efficiency": efficiency,
                   "equality": equality,
                   "sustainability": sustainability,
                   "peace": peace,
                   "fire_attempts": self.fire_counter,
                   "fire_sucsses": self.fire_sucsses}
        self.metrics = metrics
        self.timeout_record = {}
        self.rewards_record = {}
        self.fire_counter = 0
        self.fire_sucsses = 0
        
    def get_social_metrics(self):
        return self.metrics
    
