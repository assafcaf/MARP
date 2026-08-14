"""Base map class that defines the rendering process
"""

import random
import gymnasium
import matplotlib.pyplot as plt
import numpy as np
from .constants import ACTIONS, ORIENTATIONS, DIFFERENT_COLORMAP, SAME_COLORMAP


class MapEnv(gymnasium.Env):

    def __init__(self, ascii_map, num_agents=1, render=True, color_map=None, same_color=False,
                 include_state_in_info=False):
        """

        Parameters
        ----------
        ascii_map: list of strings
            Specify what the map should look like. Look at constant.py for
            further explanation
        num_agents: int
            Number of agents to have in the system.
        render: bool
            Whether to render the environment
        color_map: dict
            Specifies how to convert between ascii chars and colors
        include_state_in_info: bool
            Put the full-map RGB render on every agent's info dict as `state`.

            Off by default, and that default is worth defending. The render is a
            whole-map `map_to_colors` pass *per agent per step* -- 1938 bytes on
            the medium map against 675 for the observation the agent actually
            sees -- and it measured at ~48% of total step time with nothing in
            the repo reading it. It also has to cross a process boundary once
            the environments run in workers, at ~15 MB per iteration.

            Turn it on only for an external consumer that needs global state.
        """# rather to use effiency or effiency*peace
        self.include_state_in_info = include_state_in_info
        # Per-environment RNGs. Every source of randomness in here used to draw
        # from the process-global numpy/random streams, which made an episode
        # depend on how many other environments shared the process and in what
        # order they stepped -- so results changed with `num_workers`, and two
        # copies in one process were correlated through the shared stream.
        #
        # The seeds are drawn from the global stream at construction, so a run
        # is still governed by `Trainer._seed_rngs`; after that each copy is
        # self-contained.
        self.np_random = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        self.py_random = random.Random(np.random.randint(0, 2**31 - 1))
        self.num_agents = num_agents
        self.base_map = self.ascii_to_numpy(ascii_map)
        # map without agents or beams
        self.world_map = np.full((len(self.base_map), len(self.base_map[0])), ' ')
        self.beam_pos = []

        self.agents = {}

        # returns the agent at a desired position if there is one
        self.pos_dict = {}
        self.color_map = SAME_COLORMAP.copy() if same_color else DIFFERENT_COLORMAP.copy()

        self.spawn_points = []  # where agents can appear

        self.wall_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'P':
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == '@':
                    self.wall_points.append([row, col])
        self.setup_agents()

    def custom_reset(self):
        """Reset custom elements of the map. For example, spawn apples and build walls"""
        pass

    def custom_action(self, agent, action):
        """Execute any custom actions that may be defined, like fire or clean

        Parameters
        ----------
        agent: agent that is taking the action
        action: key of the action to be taken

        Returns
        -------
        updates: list(list(row, col, char))
            List of cells to place onto the map
        """
        pass

    def custom_map_update(self):
        """Custom map updates that don't have to do with agent actions"""
        pass

    def setup_agents(self):
        """Construct all the agents for the environment"""
        raise NotImplementedError

    # FIXME(ev) move this to a utils eventually
    def ascii_to_numpy(self, ascii_list):
        """converts a list of strings into a numpy array


        Parameters
        ----------
        ascii_list: list of strings
            List describing what the map should look like
        Returns
        -------
        arr: np.ndarray
            numpy array describing the map with ' ' indicating an empty space
        """

        arr = np.full((len(ascii_list), len(ascii_list[0])), ' ')
        for row in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                arr[row, col] = ascii_list[row][col]
        return arr

    def step(self, actions):
        """Takes in a dict of actions and converts them to a map update

        Parameters
        ----------
        actions: dict {agent-id: int}
            dict of actions, keyed by agent-id that are passed to the agent. The agent
            interprets the int and converts it to a command

        Returns
        -------
        observations: dict of arrays representing agent observations
        rewards: dict of rewards for each agent
        dones: dict indicating whether each agent is done
        info: dict to pass extra info to gymnasium
        """

        self.beam_pos = []
        agent_actions = {}
        for agent_id, action in actions.items():
            agent_action = self.agents[agent_id].action_map(action)
            agent_actions[agent_id] = agent_action

        # move
        self.update_moves(agent_actions)

        for agent in self.agents.values():
            pos = agent.get_pos()
            if pos[0] > 0 and pos[1] > 0:
                new_char = agent.consume(self.world_map[pos[0], pos[1]])
                self.world_map[pos[0], pos[1]] = new_char

        # execute custom moves like firing
        self.update_custom_moves(agent_actions)

        # execute spawning events
        self.custom_map_update()

        map_with_agents = self.get_map_with_agents()

        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        for agent in self.agents.values():
            agent.grid = map_with_agents
            rgb_arr = self.map_to_colors(agent.get_state(), self.color_map, full_map=False)
            rgb_arr = self.rotate_view(agent.orientation, rgb_arr)
            # observations[agent.agent_id] = rgb_arr
            observations[agent.agent_id] = {"curr_obs": rgb_arr}
            rewards[agent.agent_id] = agent.compute_reward()
            dones[agent.agent_id] = agent.get_done()
            infos[agent.agent_id] = (
                {"state": self.state} if self.include_state_in_info else {}
            )
        dones["__all__"] = np.any(list(dones.values()))
        return observations, rewards, dones, infos

    def reset(self, seed=0):
        """Reset the environment.

        This method is performed in between rollouts. It resets the state of
        the environment.

        Parameters
        ----------
        seed: int or None
            Random seed for reproducibility. If None, uses current RNG state.
            If provided, seeds numpy.random and random modules.

        Returns
        -------
        observation: dict of numpy ndarray
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """
        if seed is not None:
            # Re-seed this environment only. Seeding the global streams here
            # reset the policy's exploration RNG once per episode.
            self.np_random = np.random.default_rng(seed)
            self.py_random = random.Random(seed)
        self.beam_pos = []
        self.agents = {}
        self.setup_agents()
        self.reset_map()
        self.custom_map_update()

        map_with_agents = self.get_map_with_agents()

        observations = {}
        infos = {}
        for agent in self.agents.values():
            agent.grid = map_with_agents
            # agent.grid = util.return_view(map_with_agents, agent.pos,
            #                               agent.row_size, agent.col_size)
            rgb_arr = self.map_to_colors(agent.get_state(), self.color_map, full_map=False)
            # observations[agent.agent_id] = rgb_arr
            observations[agent.agent_id] = {"curr_obs": rgb_arr}
            infos[agent.agent_id] = (
                {"state": self.state} if self.include_state_in_info else {}
            )
        return observations, infos

    @property
    def agent_pos(self):
        return [agent.get_pos().tolist() for agent in self.agents.values()]

    # This method is just used for testing
    # FIXME(ev) move into the testing class
    @property
    def test_map(self):
        """Gets a version of the environment map where generic
        'P' characters have been replaced with specific agent IDs.

        Returns:
            2D array of strings representing the map.
        """
        grid = np.copy(self.world_map)

        for agent_id, agent in self.agents.items():
            # If agent is not within map, skip.
            if not (agent.pos[0] >= 0 and agent.pos[0] < grid.shape[0] and
                    agent.pos[1] >= 0 and agent.pos[1] < grid.shape[1]):
                continue

            grid[agent.pos[0], agent.pos[1]] = 'P'

        for beam_pos in self.beam_pos:
            grid[beam_pos[0], beam_pos[1]] = beam_pos[2]

        return grid

    def get_map_with_agents(self):
        """Gets a version of the environment map where generic
        'P' characters have been replaced with specific agent IDs.

        Returns:
            2D array of strings representing the map.
        """
        grid = np.copy(self.world_map)

        for agent_id, agent in self.agents.items():
            char_id = str(int(agent_id[-1]) + 1)

            # If agent is not within map, skip.
            if not(agent.pos[0] >= 0 and agent.pos[0] < grid.shape[0] and
                   agent.pos[1] >= 0 and agent.pos[1] < grid.shape[1]):
                continue

            grid[agent.pos[0], agent.pos[1]] = char_id

        for beam_pos in self.beam_pos:
            grid[beam_pos[0], beam_pos[1]] = beam_pos[2]

        return grid

    def check_agent_map(self, agent_map):
        """Checks the map to make sure agents aren't duplicated"""
        unique, counts = np.unique(agent_map, return_counts=True)
        count_dict = dict(zip(unique, counts))

        # check for multiple agents
        for i in range(self.num_agents):
            if count_dict[str(i+1)] != 1:
                print('Error! Wrong number of agent', i, 'in map!')
                return False
        return True

    def map_to_colors(self, map=None, color_map=None, full_map=False):
        """Converts a map to an array of RGB values.
        Parameters
        ----------
        map: np.ndarray
            map to convert to colors
        color_map: dict
            mapping between array elements and desired colors
        Returns
        -------
        arr: np.ndarray
            3-dim numpy array consisting of color map
        """
        if map is None:
            map = self.get_map_with_agents()
        if color_map is None:
            color_map = self.color_map

        if not full_map:
            map[map.shape[0]//2, map.shape[1]//2] = 'S'

        rgb_arr = np.zeros((map.shape[0], map.shape[1], 3), dtype=np.uint8)
        for row_elem in range(map.shape[0]):
            for col_elem in range(map.shape[1]):
                rgb_arr[row_elem, col_elem, :] = color_map[map[row_elem, col_elem]]

        return rgb_arr

    @property
    def state(self):
        return self.map_to_colors(self.get_map_with_agents(), full_map=True).astype(np.uint8)
    

    def render(self, filename=None, mod=None):
        """ Creates an image of the map to plot or save.

        Args:
            path: If a string is passed, will save the image
                to disk at this location.
        """
        map_with_agents = self.get_map_with_agents()

        rgb_arr = self.map_to_colors(map_with_agents, full_map=True)
        if mod == "human":
            plt.cla()
            plt.imshow(rgb_arr, interpolation="nearest")
            if filename is None:
                plt.show(block=False)
            else:
                plt.savefig(filename)
            return None
        return rgb_arr.astype(np.uint8)

    def update_moves(self, agent_actions):
        """Converts agent action tuples into a new map and new agent positions.
        Also resolves conflicts over multiple agents wanting a cell.

        This method works by finding all conflicts over a cell and randomly assigning them
       to one of the agents that desires the slot. It then sets all of the other agents
       that wanted the cell to have a move of staying. For moves that do not directly
       conflict with another agent for a cell, but may not be temporarily resolvable
       due to an agent currently being in the desired cell, we continually loop through
       the actions until all moves have been satisfied or deemed impossible.
       For example, agent 1 may want to move from [1,2] to [2,2] but agent 2 is in [2,2].
       Agent 2, however, is moving into [3,2]. Agent-1's action is first in the order so at the
       first pass it is skipped but agent-2 moves to [3,2]. In the second pass, agent-1 will
       then be able to move into [2,2].

        Parameters
        ----------
        agent_actions: dict
            dict with agent_id as key and action as value
        """

        reserved_slots = []
        for agent_id, action in agent_actions.items():
            agent = self.agents[agent_id]
            selected_action = ACTIONS[action]
            # TODO(ev) these two parts of the actions
            if 'MOVE' in action or 'STAY' in action:
                # rotate the selected action appropriately
                rot_action = self.rotate_action(selected_action, agent.get_orientation())
                new_pos = agent.get_pos() + rot_action
                # allow the agents to confirm what position they can move to
                new_pos = agent.return_valid_pos(new_pos)
                reserved_slots.append((*new_pos, 'P', agent_id))
            elif 'TURN' in action:
                new_rot = self.update_rotation(action, agent.get_orientation())
                agent.update_agent_rot(new_rot)

        # now do the conflict resolution part of the process

        # helpful for finding the agent in the conflicting slot
        agent_by_pos = {tuple(agent.get_pos()): agent.agent_id for agent in self.agents.values()}

        # agent moves keyed by ids
        agent_moves = {}

        # lists of moves and their corresponding agents
        move_slots = []
        agent_to_slot = []

        for slot in reserved_slots:
            row, col = slot[0], slot[1]
            if slot[2] == 'P':
                agent_id = slot[3]
                agent_moves[agent_id] = [row, col]
                move_slots.append([row, col])
                agent_to_slot.append(agent_id)

        # cut short the computation if there are no moves
        if len(agent_to_slot) > 0:

            # first we will resolve all slots over which multiple agents
            # want the slot

            # shuffle so that a random agent has slot priority
            shuffle_list = list(zip(agent_to_slot, move_slots))
            self.np_random.shuffle(shuffle_list)
            agent_to_slot, move_slots = zip(*shuffle_list)
            unique_move, indices, return_count = np.unique(move_slots, return_index=True,
                                                           return_counts=True, axis=0)
            search_list = np.array(move_slots)

            # first go through and remove moves that can't possible happen. Three types
            # 1. Trying to move into an agent that has been issued a stay command
            # 2. Trying to move into the spot of an agent that doesn't have a move
            # 3. Two agents trying to walk through one another

            # Resolve all conflicts over a space
            if np.any(return_count > 1):
                for move, index, count in zip(unique_move, indices, return_count):
                    if count > 1:
                        # check that the cell you are fighting over doesn't currently
                        # contain an agent that isn't going to move for one of the agents
                        # If it does, all the agents commands should become STAY
                        # since no moving will be possible
                        conflict_indices = np.where((search_list == move).all(axis=1))[0]
                        all_agents_id = [agent_to_slot[i] for i in conflict_indices]
                        # all other agents now stay in place so update their moves
                        # to reflect this
                        conflict_cell_free = True
                        for agent_id in all_agents_id:
                            moves_copy = agent_moves.copy()
                            # TODO(ev) code duplication, simplify
                            if move.tolist() in self.agent_pos:
                                # find the agent that is currently at that spot and make sure
                                # that the move is possible. If it won't be, remove it.
                                conflicting_agent_id = agent_by_pos[tuple(move)]
                                curr_pos = self.agents[agent_id].get_pos().tolist()
                                curr_conflict_pos = self.agents[conflicting_agent_id]. \
                                    get_pos().tolist()
                                conflict_move = agent_moves.get(conflicting_agent_id,
                                                                curr_conflict_pos)
                                # Condition (1):
                                # a STAY command has been issued
                                if agent_id == conflicting_agent_id:
                                    conflict_cell_free = False
                                # Condition (2)
                                # its command is to stay
                                # or you are trying to move into an agent that hasn't
                                # received a command
                                elif conflicting_agent_id not in moves_copy.keys() or \
                                        curr_conflict_pos == conflict_move:
                                    conflict_cell_free = False

                                # Condition (3)
                                # It is trying to move into you and you are moving into it
                                elif conflicting_agent_id in moves_copy.keys():
                                    if agent_moves[conflicting_agent_id] == curr_pos and \
                                            move.tolist() == self.agents[conflicting_agent_id] \
                                            .get_pos().tolist():
                                        conflict_cell_free = False

                        # if the conflict cell is open, let one of the conflicting agents
                        # move into it
                        if conflict_cell_free:
                            self.agents[agent_to_slot[index]].update_agent_pos(move)
                            agent_by_pos = {tuple(agent.get_pos()):
                                            agent.agent_id for agent in self.agents.values()}
                        # ------------------------------------
                        # remove all the other moves that would have conflicted
                        remove_indices = np.where((search_list == move).all(axis=1))[0]
                        all_agents_id = [agent_to_slot[i] for i in remove_indices]
                        # all other agents now stay in place so update their moves
                        # to stay in place
                        for agent_id in all_agents_id:
                            agent_moves[agent_id] = self.agents[agent_id].get_pos().tolist()

            # make the remaining un-conflicted moves
            while len(agent_moves.items()) > 0:
                agent_by_pos = {tuple(agent.get_pos()):
                                agent.agent_id for agent in self.agents.values()}
                num_moves = len(agent_moves.items())
                moves_copy = agent_moves.copy()
                del_keys = []
                for agent_id, move in moves_copy.items():
                    if agent_id in del_keys:
                        continue
                    if move in self.agent_pos:
                        # find the agent that is currently at that spot and make sure
                        # that the move is possible. If it won't be, remove it.
                        conflicting_agent_id = agent_by_pos[tuple(move)]
                        curr_pos = self.agents[agent_id].get_pos().tolist()
                        curr_conflict_pos = self.agents[conflicting_agent_id].get_pos().tolist()
                        conflict_move = agent_moves.get(conflicting_agent_id, curr_conflict_pos)
                        # Condition (1):
                        # a STAY command has been issued
                        if agent_id == conflicting_agent_id:
                            del agent_moves[agent_id]
                            del_keys.append(agent_id)
                        # Condition (2)
                        # its command is to stay
                        # or you are trying to move into an agent that hasn't received a command
                        elif conflicting_agent_id not in moves_copy.keys() or \
                                curr_conflict_pos == conflict_move:
                            del agent_moves[agent_id]
                            del_keys.append(agent_id)
                        # Condition (3)
                        # It is trying to move into you and you are moving into it
                        elif conflicting_agent_id in moves_copy.keys():
                            if agent_moves[conflicting_agent_id] == curr_pos and \
                                    move == self.agents[conflicting_agent_id].get_pos().tolist():
                                del agent_moves[conflicting_agent_id]
                                del agent_moves[agent_id]
                                del_keys.append(agent_id)
                                del_keys.append(conflicting_agent_id)
                    # this move is unconflicted so go ahead and move
                    else:
                        self.agents[agent_id].update_agent_pos(move)
                        del agent_moves[agent_id]
                        del_keys.append(agent_id)

                # no agent is able to move freely, so just move them all
                # no updates to hidden cells are needed since all the
                # same cells will be covered
                if len(agent_moves) == num_moves:
                    for agent_id, move in agent_moves.items():
                        self.agents[agent_id].update_agent_pos(move)
                    break

    def update_custom_moves(self, agent_actions):
        for agent_id, action in agent_actions.items():
            # check its not a move based action
            if 'MOVE' not in action and 'STAY' not in action and 'TURN' not in action:
                agent = self.agents[agent_id]
                updates = self.custom_action(agent, action)
                if len(updates) > 0:
                    self.update_map(updates)

    def update_map(self, new_points):
        """For points in new_points, place desired char on the map"""
        for i in range(len(new_points)):
            row, col, char = new_points[i]
            self.world_map[row, col] = char

    def reset_map(self):
        """Resets the map to be empty as well as a custom reset set by subclasses"""
        self.world_map = np.full((len(self.base_map), len(self.base_map[0])), ' ')
        self.build_walls()
        self.custom_reset()

    def update_map_fire(self, firing_pos, firing_orientation, fire_len, fire_char, cell_types=[],
                        update_char=[], blocking_cells='P'):
        """From a firing position, fire a beam that may clean or hit agents

        Notes:
            (1) Beams are blocked by agents
            (2) A beam travels along until it hits a blocking cell at which beam the beam
                covers that cell and stops
            (3) If a beam hits a cell whose character is in cell_types, it replaces it with
                the corresponding index in update_char
            (4) As per the rules, the beams fire from in front of the agent and on its
                sides so the beam that starts in front of the agent travels out one
                cell further than it does along the sides.
            (5) This method updates the beam_pos, an internal representation of how
                which cells need to be rendered with fire_char in the agent view

        Parameters
        ----------
        firing_pos: (list)
            the row, col from which the beam is fired
        firing_orientation: (list)
            the direction the beam is to be fired in
        fire_len: (int)
            the number of cells forward to fire
        fire_char: (str)
            the cell that should be placed where the beam goes
        cell_types: (list of str)
            the cells that are affected by the beam
        update_char: (list of str)
            the character that should replace the affected cells.
        blocking_cells: (list of str)
            cells that block the firing beam
        Returns
        -------
        updates: (tuple (row, col, char))
            the cells that have been hit by the beam and what char will be placed there
        """
        agent_by_pos = {tuple(agent.get_pos()): agent_id for agent_id, agent in self.agents.items()}
        start_pos = np.asarray(firing_pos)
        firing_direction = ORIENTATIONS[firing_orientation]
        # compute the other two starting positions
        right_shift = self.rotate_right(firing_direction)
        firing_pos = [start_pos, start_pos + right_shift - firing_direction,
                      start_pos - right_shift - firing_direction]
        firing_points = []
        updates = []
        for pos in firing_pos:
            next_cell = pos + firing_direction
            for i in range(fire_len):
                if self.test_if_in_bounds(next_cell) and \
                        self.world_map[next_cell[0], next_cell[1]] != '@':

                    # FIXME(ev) code duplication
                    # agents absorb beams
                    # activate the agents hit function if needed
                    if [next_cell[0], next_cell[1]] in self.agent_pos:
                        agent_id = agent_by_pos[(next_cell[0], next_cell[1])]
                        self.agents[agent_id].hit(fire_char)
                        firing_points.append((next_cell[0], next_cell[1], fire_char))
                        if self.world_map[next_cell[0], next_cell[1]] in cell_types:
                            type_index = cell_types.index(self.world_map[next_cell[0],
                                                                         next_cell[1]])
                            updates.append((next_cell[0], next_cell[1], update_char[type_index]))
                        break

                    # update the cell if needed
                    if self.world_map[next_cell[0], next_cell[1]] in cell_types:
                        type_index = cell_types.index(self.world_map[next_cell[0], next_cell[1]])
                        updates.append((next_cell[0], next_cell[1], update_char[type_index]))

                    firing_points.append((next_cell[0], next_cell[1], fire_char))

                    # check if the cell blocks beams. For example, waste blocks beams.
                    if self.world_map[next_cell[0], next_cell[1]] in blocking_cells:
                        break

                    # increment the beam position
                    next_cell += firing_direction

                else:
                    break

        self.beam_pos += firing_points
        return updates

    def spawn_point(self):
        """Returns a randomly selected free spawn point.

        Two bugs used to live in these six lines, and between them they meant
        `reset(seed=...)` did not determine a run:

        1. `random.shuffle(self.spawn_points)` shuffled the instance's own list
           *in place*. Seeding fixes the shuffle operation, but it was applied
           to whatever order the previous shuffles had left behind, so the same
           seed produced different agent layouts on every reset. Shuffling an
           index permutation instead leaves `self.spawn_points` in its original
           order, so the seed alone determines the result.
        2. The loop had no `break`, so it kept overwriting `spawn_index` and
           returned the *last* free spawn point rather than a chosen one. The
           in-place shuffle was the only thing making spawning random at all.
        """
        curr_agent_pos = [agent.get_pos().tolist() for agent in self.agents.values()]
        order = list(range(len(self.spawn_points)))
        self.py_random.shuffle(order)
        for index in order:
            spawn_point = self.spawn_points[index]
            if [spawn_point[0], spawn_point[1]] not in curr_agent_pos:
                return np.array(spawn_point)
        raise AssertionError('There are not enough spawn points! Check your map?')

    def spawn_rotation(self):
        """Return a randomly selected initial rotation for an agent"""
        rand_int = int(self.np_random.integers(len(ORIENTATIONS.keys())))
        return list(ORIENTATIONS.keys())[rand_int]

    def rotate_view(self, orientation, view):
        """Takes a view of the map and rotates it the agent orientation
        Parameters
        ----------
        orientation: str
            str in {'UP', 'LEFT', 'DOWN', 'RIGHT'}
        view: np.ndarray (row, column, channel)
        Returns
        -------
        a rotated view
        """
        if orientation == 'UP':
            return view
        elif orientation == 'LEFT':
            return np.rot90(view, k=1, axes=(0, 1))
        elif orientation == 'DOWN':
            return np.rot90(view, k=2, axes=(0, 1))
        elif orientation == 'RIGHT':
            return np.rot90(view, k=3, axes=(0, 1))
        else:
            raise ValueError('Orientation {} is not valid'.format(orientation))

    def build_walls(self):
        for i in range(len(self.wall_points)):
            row, col = self.wall_points[i]
            self.world_map[row, col] = '@'

    ########################################
    # Utility methods, move these eventually
    ########################################

    # TODO(ev) this can be a general property of map_env or a util
    def rotate_action(self, action_vec, orientation):
        # WARNING: Note, we adopt the physics convention that \theta=0 is in the +y direction
        if orientation == 'UP':
            return action_vec
        elif orientation == 'LEFT':
            return self.rotate_left(action_vec)
        elif orientation == 'RIGHT':
            return self.rotate_right(action_vec)
        else:
            return self.rotate_left(self.rotate_left(action_vec))

    def rotate_left(self, action_vec):
        return np.dot(ACTIONS['TURN_COUNTERCLOCKWISE'], action_vec)

    def rotate_right(self, action_vec):
        return np.dot(ACTIONS['TURN_CLOCKWISE'], action_vec)

    # TODO(ev) this should be an agent property
    def update_rotation(self, action, curr_orientation):
        if action == 'TURN_COUNTERCLOCKWISE':
            if curr_orientation == 'LEFT':
                return 'DOWN'
            elif curr_orientation == 'DOWN':
                return 'RIGHT'
            elif curr_orientation == 'RIGHT':
                return 'UP'
            else:
                return 'LEFT'
        else:
            if curr_orientation == 'LEFT':
                return 'UP'
            elif curr_orientation == 'UP':
                return 'RIGHT'
            elif curr_orientation == 'RIGHT':
                return 'DOWN'
            else:
                return 'LEFT'

    # TODO(ev) this definitely should go into utils or the general agent class
    def test_if_in_bounds(self, pos):
        """Checks if a selected cell is outside the range of the map"""
        if pos[0] < 0 or pos[0] >= self.world_map.shape[0]:
            return False
        elif pos[1] < 0 or pos[1] >= self.world_map.shape[1]:
            return False
        else:
            return True
