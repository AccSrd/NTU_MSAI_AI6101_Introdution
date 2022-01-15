import sys
import time
import copy
import math
import random

import numpy as np

from types import SimpleNamespace as SN
from typing import List, Union, Any, Set, Dict, Sequence


# agents: -1. box: -2. goal: -3. cliff: -4. Others: -5
OBS_AGENT = -1
OBS_BOX = -2
OBS_GOAL = -3
OBS_CLIFF = -4
OBS_GRID = -5

AGENT_0 = 'A'
BOX = 'B'
GOAL = 'G'
DANGER = 'x'
GRID = '_'
DEFAULT_DANGER_REGION = {
    'A': [4, 3],
    'B': [4, 10],
    'C': [5, 2],
    'D': [5, 11],
}

AGENTS_POSIITON = np.array([[5, 0]])
BOX_POSITION = np.array([4, 1])

class CliffBoxPushingBase:
    """
    Cliff Box Pushing
    """
    # all possible actions
    ACTION_NO_OP = 0  # always not available
    ACTION_UP = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_RIGHT = 4
    ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
    FORCES = {
        ACTION_UP:    np.array([-1, 0]),
        ACTION_DOWN:  np.array([1,  0]),
        ACTION_LEFT:  np.array([0, -1]),
        ACTION_RIGHT: np.array([0,  1]),
    }
    def __init__(self,
                 seed=None,
                 map_name='cliffboxpushing_v0',
                 steps=50,
                 n_agents=2,
                 world_width=13,
                 world_height=6,
                 reward_offcliff=-200,
                 obs_size=3,
                 reward_box_moving=True,
                 danger_region=DEFAULT_DANGER_REGION,
                 state_last_action=True):
        """
        Agent has an obervation of 3x3.

             _______________________________________________________
         ___|_0_|_1_|_2_|_3_|_4_|_5_|_6_|_7_|_8_|_9_|_10|_11|_12|_13|
        |_0_|___|___|___|___|___|___|___|___|___|___|___|___|___|___|
        |_1_|___|___|___|___|___|___|___|___|___|___|___|___|___|___|
        |_2_|___|___|___|___|___|___|___|___|___|___|___|___|___|___|
        |_3_|___|___|___|___|___|___|___|___|___|___|___|___|___|___|
        |_4_|___|_B_|___|_x_|_x_|_x_|_x_|_x_|_x_|_x_|_x_|_x_|___|_G_|
        |_5_|_A_|___|_x_|_x_|_x_|_x_|_x_|_x_|_x_|_x_|_x_|_x_|_x_|___|
        """
        self.world_width = world_width
        self.world_height = world_height
        self.reward_offcliff = reward_offcliff
        self.danger_region = SN(**danger_region)
        self.obs_size = obs_size

        self.episode_limit = steps
        self.box_attached = False
        self.reward_box_moving = True

        self.start = BOX_POSITION
        self.goal = np.array([4, self.world_width-1])

        self.world = np.chararray((world_height, world_width))
        self.world[:] = GRID
        self.world[self.danger_region.A[0], self.danger_region.A[1]:self.danger_region.B[1]+1] = DANGER
        self.world[self.danger_region.C[0], self.danger_region.C[1]:self.danger_region.D[1]+1] = DANGER

        self.n_agents = n_agents
        self._seed = seed
        self._agent_ids = list(range(self.n_agents))
        self.curr_t = 0
        self.agent_curr_positions = copy.deepcopy(AGENTS_POSIITON)
        self.agent_pre_positions = copy.deepcopy(AGENTS_POSIITON)
        self.box_curr_position = copy.deepcopy(BOX_POSITION)
        self.box_pre_position = copy.deepcopy(BOX_POSITION)
        agent_0 = self.agent_curr_positions[0]
        
        self.world[agent_0[0], agent_0[1]] = AGENT_0
        self.world[4, 1] = BOX
        self.world[4, self.world_width-1] = GOAL

        # states and observations
        self._state, self._obs = None, None
        self._available_actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]
        self._init_obs_states()

        self.episode_actions = []

    def _init_obs_states(self):
        self._init_states()
        self._init_obs()

    def _init_states(self):
        # states
        self._state = np.zeros((self.world_height, self.world_width)) + OBS_GRID
        agent_0 = self.agent_curr_positions[0]
        pos = self.box_curr_position
        self._state[agent_0[0], agent_0[1]] = OBS_AGENT
        self._state[pos[0], pos[1]] = OBS_BOX
        self._state[self.goal[0], self.goal[1]] = OBS_AGENT
        self._state[self.danger_region.A[0], self.danger_region.A[1]:self.danger_region.B[1]+1] = OBS_CLIFF
        self._state[self.danger_region.C[0], self.danger_region.C[1]:self.danger_region.D[1]+1] = OBS_CLIFF
        self._state = self._state.flatten().tolist()

    def _init_obs(self):
        agent_0 = self.agent_curr_positions[0]
        # observation of agent 0
        obs_0 = np.zeros((self.obs_size, self.obs_size)) + OBS_GRID  # init value
        obs_0[self.obs_size//2, self.obs_size//2] = OBS_AGENT  # agent
        obs_0 = np.concatenate((obs_0.flatten(), agent_0))  # position of itself
        obs_0 = np.concatenate((obs_0, np.array([self._agent_box_distance(0)])))  # add box distance

        self._obs = obs_0

    def _update_states(self):
        self._init_states()

    def _update_agent_obs(self, agent_id):
        agent_pos = self.agent_curr_positions[agent_id]
        # observation of agent 0: basic
        obs = np.zeros((self.obs_size, self.obs_size)) + OBS_GRID  # init value
        obs[self.obs_size//2, self.obs_size//2] = OBS_AGENT  # agent

        # get box info
        if abs(agent_pos[0] - self.box_curr_position[0]) <= 1 and abs(agent_pos[1] - self.box_curr_position[1]) <= 1:
            x_diff = self.box_curr_position[0] - agent_pos[0]
            y_diff = self.box_curr_position[1] - agent_pos[1]
            obs[x_diff+1, y_diff+1] = OBS_BOX

        # get goal info
        if abs(agent_pos[0] - self.goal[0]) <= 1 and abs(agent_pos[1] - self.goal[1]) <= 1:
            x_diff = agent_pos[0] - self.goal[0]
            y_diff = agent_pos[1] - self.goal[1]
            obs[x_diff+1, y_diff+1] = OBS_GOAL

        # get cliff info. Check the position of it
        for col in [-1, 0, 1]:
            for row in [-1, 0, 1]:
                pos = [agent_pos[0] + row, agent_pos[1] + col]
                if (pos[0] == self.danger_region.A[0] and self.danger_region.A[1] <= pos[1] <= self.danger_region.B[1]) or \
                    (pos[0] == self.danger_region.C[0] and self.danger_region.C[1] <= pos[1] <= self.danger_region.D[1]):
                    obs[row+1, col+1] = OBS_CLIFF

        obs = np.concatenate((obs.flatten(), agent_pos))  # position of itself
        obs = np.concatenate((obs, np.array([self._agent_box_distance(0)])))  # add box distance
        return obs

    def _update_obs(self):
        """
        Init basic data. And put other agent, box, cliff and even goal into the observation
        """
        obs_0 = self._update_agent_obs(0)
        self._obs = [obs_0]

    def _agent_box_distance(self, agent_id):
        return np.sum(np.abs(self.agent_curr_positions[agent_id] - self.box_curr_position))

    def reset(self):
        self.world = np.chararray((self.world_height, self.world_width))
        self.world[:] = GRID
        self.world[self.danger_region.A[0], self.danger_region.A[1]:self.danger_region.B[1]+1] = DANGER
        self.world[self.danger_region.C[0], self.danger_region.C[1]:self.danger_region.D[1]+1] = DANGER

        self.curr_t = 0
        self.agent_curr_positions = copy.deepcopy(AGENTS_POSIITON)
        self.agent_pre_positions = copy.deepcopy(AGENTS_POSIITON)
        self.box_curr_position = copy.deepcopy(self.start)
        self.box_pre_position = copy.deepcopy(self.start)
        agent_0 = self.agent_curr_positions[0]
        self.world[agent_0[0], agent_0[1]] = AGENT_0
        self.world[4, 1] = BOX
        self.world[4, self.world_width-1] = GOAL
        self.box_attached = False

        # states and observations
        self._available_actions = [1, 1, 1, 1]
        self._init_obs_states()
        self.episode_actions = []

    def step(self, actions):
        """ 
        Returns reward, terminated, info
        """
        info = {}

        int_actions = [int(a) for a in actions]

        self.episode_actions.append(int_actions)

        self.check_box_attached()
        # Phase 1. Change the state
        # 1. is both the agent attached to the box?
        if self.box_attached:
            agent_0_force = self.FORCES[int_actions[0]]
            # check if the force is on the box
            if self.can_move_box(agent_forces=agent_0_force):
                # get the joint force
                joint_force = np.clip(agent_0_force, a_min=-1, a_max=1)
                # the box can be moved
                # check out of boundary
                box_position = self.box_curr_position + joint_force
                box_position, box_stuck = \
                    self._check_pos_boundary(pos=box_position, box_hard_boundary=True)

                self.box_curr_position = box_position
                self.update_agent_pos(agent_id=0, force=self.FORCES[int_actions[0]])
            # else move the agent
            else:
                # check agent 0, 1
                self.update_agent_pos(agent_id=0, force=self.FORCES[int_actions[0]])
                # the box is not attached to the agents now
                # check self.box_attached
                self.check_box_attached()
        # 2. else the box cannot move
        else:
            # check agent 0, 1
            self.update_agent_pos(agent_id=0, force=self.FORCES[int_actions[0]])
            # check self.box_attached
            self.check_box_attached()

        # Phase 2. calculate the rewards
        reward, teminated = self.calculate_rewards()

        # Phase 3. Update the world
        self.update_world()

        # Phase 4. Update the observations and states
        self._update_states()
        self._update_obs()
        self.curr_t += 1
        return reward, teminated, info

    def update_world(self):
        agent_0 = self.agent_curr_positions[0]
        box_pos = self.box_curr_position

        # if position changed
        if not all(self.agent_pre_positions[0] == self.agent_curr_positions[0]):
            pos = self.agent_pre_positions[0]
            if (pos[0] != self.goal[0]) or (pos[1] != self.goal[1]):
                self.world[pos[0], pos[1]] = GRID

        if not all(self.box_pre_position == self.box_curr_position):
            pos = self.box_pre_position
            if self.world[pos[0], pos[1]].decode('UTF-8') not in {AGENT_0}:
                self.world[pos[0], pos[1]] = GRID

        if (agent_0[0] != self.goal[0]) or (agent_0[1] != self.goal[1]):
            self.world[agent_0[0], agent_0[1]] = AGENT_0
        self.world[box_pos[0], box_pos[1]] = BOX

        self.box_pre_position = copy.deepcopy(self.box_curr_position)
        self.agent_pre_positions = copy.deepcopy(self.agent_curr_positions)

    def print_world(self):
        if len(self.episode_actions) > 0:
            print(f'Action: {self.episode_actions[-1]}')
        print(self.world)

    def _check_pos_boundary(self, pos, box_hard_boundary=False):
        stuck = [False, False]
        
        if pos[0] < 0:
            pos[0] = 0
            stuck[0] = True
        if pos[0] > self.world_height-1:
            pos[0] = self.world_height-1
            stuck[0] = True
        
        if pos[1] < 0:
            pos[1] = 0
            stuck[1] = True
        if pos[1] > self.world_width-1:
            pos[1] = self.world_width-1
            stuck[1] = True
        
        if box_hard_boundary:
            if pos[0] == 0:
                pos[0] += 1
                stuck[0] = True
            elif pos[0] == self.world_height-1:
                pos[0] = self.world_height-2
                stuck[0] = True

            if pos[1] == 0:
                pos[1] += 1
                stuck[1] = True
                    
        return pos, stuck

    def calculate_rewards(self):
        teminated = self.curr_t == self.episode_limit - 1
        # the distance between agents and box
        dist_0 = np.sum(np.abs(self.agent_curr_positions[0] - self.box_curr_position))
        reward = -1  # -1 for each step
        if dist_0 == 1:
            reward += -dist_0
        else:
            reward += -dist_0
        # if agents or box is off the cliff
        if self.check_agent_off_cliff(agent_id=0) or \
                self.check_box_off_cliff():
            reward += self.reward_offcliff
            teminated = True
        
        if all(self.box_curr_position == self.goal):
            teminated = True
        
        if self.reward_box_moving:
            reward += -np.sum(np.abs(self.box_curr_position - self.goal))

        return reward, teminated

    def check_agent_off_cliff(self, agent_id):
        pos = self.agent_curr_positions[agent_id]
        return self._check_off_cliff(pos)

    def check_box_off_cliff(self):
        pos = self.box_curr_position
        return self._check_off_cliff(pos)

    def _check_off_cliff(self, pos):
        if (pos[0] == self.danger_region.A[0] and self.danger_region.A[1] <= pos[1] <= self.danger_region.B[1]) or \
            (pos[0] == self.danger_region.C[0] and self.danger_region.C[1] <= pos[1] <= self.danger_region.D[1]):
            return True
        return False

    def check_box_attached(self):
        dist_0 = np.sum(np.abs(self.agent_curr_positions[0] - self.box_curr_position))
        if dist_0 == 1:
            self.box_attached = True

    def can_move_box(self, agent_forces):
        agent_0_force = agent_forces
        if all(self.agent_curr_positions[0] + agent_0_force == self.box_curr_position):
               return True
        return False

    def update_agent_pos(self, agent_id: int, force: List[int]) -> None:
        """ Update agent's position """
        agent_position = self.agent_curr_positions[agent_id] + force
        agent_position, _ = self._check_pos_boundary(agent_position)
        if all(agent_position == self.box_curr_position):
            agent_position = self.agent_curr_positions[agent_id]
        self.agent_curr_positions[agent_id] = agent_position

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id=0):
        """ Returns observation for agent_id """
        return self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self._obs[0])

    def get_state(self):
        """ Get current state: current position of both the box and agent """
        return (tuple(self.agent_curr_positions[0].tolist()), tuple(self.box_curr_position.tolist()))

    def get_state_size(self):
        """ Returns the shape of the state"""
        return 4

    def get_avail_actions(self):
        _available_actions = [1, 1, 1, 1]
        return _available_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return self._available_actions

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO maybe there is a problem
        return len(self._available_actions)

    def render(self):
        self.print_world()

    def close(self):
        pass

    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info