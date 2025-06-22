""" Board
"""

import random
from collections import OrderedDict
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from enum import Enum


class Direction(Enum):
    """ Direction on grid of Board.
    """
    NORTH = 1   # -i
    SOUTH = 2   # +i
    EAST = 3    # +j
    WEST = 4    # -j


class Ship():
    """ Ship object

        Attributes
        - name : str  -- name of ship
        - size : int  -- size of ship (length)
        - orient : Direction(Enum) -- direction ship body is facing on grid
        - head_loc : (int, int)  -- location of boat's head on grid
        - life_points : int -- modifiable parameter for health

        Behaviors
        - ouch()  --  decrement life_points
    """

    def __init__(self, name='destroyer', size=2, head_loc=(0, 0),
                 orient=Direction.NORTH):

        """ Initalizes a Ship object.
            Note that this object does not do validity checks with the Board
                (i.e. does not verify is head_loc is a valid spot)
        """

        self.name = name
        self.size = size
        self.orient = orient
        self.head_loc = head_loc

        self.life_points = size


    def ouch(self):
        """
        Got torpedo'd.  Decrement life
        """
        self.life_points -= 1
        return


    def __repr__(self):
        return f"Ship({self.name}, size: {self.size}, head: {self.head_loc}, orientation: {self.orient}, lp: {self.life_points})"


    def __str__(self):
        return self.__repr__()



class Board():
    """ Board object.  Runs most of the battleship game.

        Attributes:
        - dim : int  -- dimension of square board
        - grid : nparray -- 2D array of int32 representing game board
                         -- 0 for ocean, +Z for ship locations, -Z for hit ships
        - ships : list(Ship)  -- information about battleships on the board
        - torpedos_used : int  -- number of torpedos fired so far
        - vis : bool  -- flag for visualizations

        Behaviors:
        - torpedo(coordinates : (int, int))  -- launch torpedo @ coordinate
        - score() -- return self.torpedos_used
        - check_gameover() -- Check gameover condition (all boats sunken)
        - ships_afloat_count() --  Returns number of ships current not sunk
        - print()  -- Prints self.grid as np.array.  Not fancy

        If using custom ship configuration, import list of ships as the
        following into the second parameter of __init__():
          ship_config = [('destroyer', 2), ('submarine', 3), ('cruiser', 3),
                         ('battleship', 4), ('carrier', 5)]
    """

    def __init__(self, dim: int = 10, ship_config='default', vis=True, playmode=True):
        """ Initializes game board for Battleship
              dim - dimension of square grid
              ship_config - 'default' for standard ship configuration and random placement.
                  Also accepts a list of Ship objects to subvert default placement.
              vis - flag for visualization.  Set to False to run headless
              playmode - if vis is True, this flag will toggle showing ship locations
                  False will display ship locations on the visualization.
        """

        self.dim = dim
        self.grid = np.zeros(shape=[dim, dim], dtype=np.int32)
        self.ships = list()

        self.torpedos_used = 0

        self._spawn_ships(presets=ship_config)

        self.vis = vis
        if vis:
            self._init_vis(playmode)
            # Event hook for mouse clicks
            self.fig.canvas.mpl_connect('button_press_event', self._button_press)

            ##################

    # Public behaviors

    def torpedo(self, coordinates):
        """ Launch attack on coordinate (i, j).  Coordinate is tuple of ints

            Return 0 for miss.  1 for hit.
            Hitting a location already torpedo'd is a miss.
            Hits will flip the sign of the .grid integer
        """

        i, j = coordinates
        if not self._is_position_valid(coordinates):
            return ValueError("Provided coordinates are invalid")

        self.torpedos_used += 1

        # ocean hit or repeated hit
        if self.grid[i, j] == 0:
            if self.vis:
                self._draw_torpedo(coordinates, hit=False)

            return 0

        # repeated hit, counts as miss
        elif self.grid[i, j] < 0:
            return 0

        # success case
        else:
            # update ships list
            idx = int(self.grid[i, j] - 1)
            hit_ship = self.ships[idx]
            hit_ship.ouch()
            # update grid
            self.grid[i, j] *= -1

            if self.vis:
                self._draw_torpedo(coordinates, hit=True)

            if self.vis and hit_ship.life_points == 0:
                self._draw_ship(idx)

            return 1

    def score(self):
        """ Returns int of torpedos used.  Essentially a score on the game
        """
        return self.torpedos_used

    def check_gameover(self):
        """ Check gameover condition (all ships sunk).
        """
        total_life = sum(ship.life_points for ship in self.ships)
        if total_life == 0:
            return True

        else:
            return False

    def ships_afloat_count(self):
        """ Returns number of ships remaining on the board and total lives left
        """
        cnt = 0
        total = 0

        for ship in self.ships:
            if ship.life_points > 0:
                cnt += 1
                total += ship.life_points

        return cnt, total

    def print(self):
        """ Simple print of grid
        """
        return self.grid

        ##################

    # Private behaviors

    def _spawn_ships(self, presets='default'):
        """ Spawn ships on the grid

            presets defines ship configuration.
            placement and orientation is uniform random across the grid.
        """

        if presets == 'default':
            ships = [('destroyer', 2), ('submarine', 3), ('cruiser', 3),
                     ('battleship', 4), ('carrier', 5)]

            for ship in ships:
                name, size = ship
                self._spawn_ship_random(name, size)
            return

        elif isinstance(presets[0], Ship):
            ships = presets

            for ship in ships:
                self._spawn_ship_specified(ship)
            return

        else:
            ValueError("Input presets into _spawn_ships() is not valid")

    def _spawn_ship_specified(self, ship):

        self._place_ship_on_grid(ship)
        self.ships.append(ship)

    def _spawn_ship_random(self, name, size):
        """ Spawns a ship.  Random position and orientation
        """

        done = False
        while (not done):

            # find random valid head position
            if not self._is_position_valid_ocean(head_pos := np.random.randint(0, self.dim, size=2)):
                continue

                # find random valid orientation
            dirs_rand = [Direction.NORTH, Direction.SOUTH, \
                         Direction.EAST, Direction.WEST]
            random.shuffle(dirs_rand)

            for orient in dirs_rand:
                if self._is_valid_ship_placement(head_pos, orient, size):
                    done = True
                    break

        ship = Ship(name=name, size=size, head_loc=head_pos, orient=orient)
        self._place_ship_on_grid(ship)
        self.ships.append(ship)

    def _place_ship_on_grid(self, ship):
        """ Updates self.grid with ship position
        """
        ship_idx = len(self.ships) + 1

        i, j = ship.head_loc
        self.grid[i, j] = ship_idx

        pos = ship.head_loc
        for _ in range(ship.size - 1):
            pos = self._step_in_dir(pos, ship.orient)
            i, j = pos
            self.grid[i, j] = ship_idx

    def _is_valid_ship_placement(self, head_pos, orient, size):
        """ Checks if provided ship configration on grid is valid
        """

        next_pos = head_pos
        for idx in range(size - 1):
            next_pos = self._step_in_dir(next_pos, orient)

            if not self._is_position_valid_ocean(next_pos):
                return False

        return True

    def _step_in_dir(self, pos, orient):
        """ Steps coordinate along a provided direction.  Can be confusing.
        """

        pos = np.array(pos)
        if orient == Direction.NORTH:
            return pos + (-1, 0)

        elif orient == Direction.SOUTH:
            return pos + (1, 0)

        elif orient == Direction.EAST:
            return pos + (0, 1)

        elif orient == Direction.WEST:
            return pos + (0, -1)

        else:
            raise ValueError("Invalid orientation provided.")

    def _is_position_valid(self, pos):
        """ Checks if grid[i, j] is occupied where pos = (i, j)
        """
        i, j = pos
        if (i < 0) or (j < 0) or (i >= self.dim) or (j >= self.dim):
            return False
        else:
            return True

    def _is_position_valid_ocean(self, pos):
        """ Checks if grid[i, j] is occupied where pos = (i, j)
        """
        i, j = pos

        return self._is_position_valid(pos) and self.grid[i, j] == 0

        ## Graphics

    def _init_vis(self, playmode):
        """ Initializes visualization objects
            .fig
            .ax
            .grid_display
        """
        self.fig = plt.figure(figsize=((self.dim + 2) / 3., (self.dim + 2) / 3.))
        self.fig.suptitle(f'Torpedos used: {self.torpedos_used}', fontsize=10)
        self.ax = self.fig.add_axes((0.05, 0.05, 0.9, 0.9),
                                    aspect='equal', frameon=False,
                                    xlim=(-0.05, self.dim + 0.05),
                                    ylim=(-0.05, self.dim + 0.05))
        for axis in (self.ax.xaxis, self.ax.yaxis):
            axis.set_major_formatter(plt.NullFormatter())
            axis.set_major_locator(plt.NullLocator())

        self.grid_display = np.array([[RegularPolygon((i + 0.5, j + 0.5),
                                                      numVertices=4,
                                                      radius=0.5 * np.sqrt(2),
                                                      orientation=np.pi / 4,
                                                      ec='black',
                                                      fc='cornflowerblue')
                                       for j in range(self.dim)]
                                      for i in range(self.dim)])

        [self.ax.add_patch(sq) for sq in self.grid_display.flat]

        if not playmode:
            self._draw_ships()

    def _draw_ship(self, idx_of_ship):

        ship = self.ships[idx_of_ship]

        i, j = ship.head_loc

        while self.grid[i, j] == int(-1 * (idx_of_ship + 1)):
            self.grid_display[i, j].set_facecolor('slategray')

            position = self._step_in_dir(np.array((i, j)), ship.orient)
            i, j = position

            if i < 0 or j < 0 or i >= self.dim or j >= self.dim:
                break

    def _draw_ships(self):

        for i in range(self.dim):
            for j in range(self.dim):
                if self.grid[i, j] > 0:
                    self.grid_display[i, j].set_facecolor('slategray')

        self.fig.canvas.draw()

    def _draw_torpedo(self, coordinates, hit):

        i, j = coordinates
        if hit:
            self.ax.add_patch(plt.Circle((i + 0.5, j + 0.5), radius=0.25,
                                         ec='red', fc='orangered'))

        else:
            self.ax.add_patch(plt.Circle((i + 0.5, j + 0.5), radius=0.25,
                                         ec='red', fc='black'))

        self.fig.suptitle(f'Torpedos used: {self.torpedos_used}', fontsize=10)
        self.fig.canvas.draw()

    def _button_press(self, event):
        """ Event hook for catching mouse clicks
            Pipes left and right click actions to user_select() and user_flag()
        """

        # Get coordinates of cell clicked on
        i, j = map(int, (event.xdata, event.ydata))
        if (i < 0 or j < 0 or i >= self.dim or j >= self.dim):
            return

        # Left Mouse Click.  button == 1
        #  Pipe to user_select
        if event.button == 1:
            self.torpedo((i, j))

        # Redraw canvas
        self.fig.canvas.draw()


BOARD_DIM = 10

# reward function parameters
PERSISTENCE_PENALTY = 0
HIT_REWARD = .5
REPEATED_PENALTY = -0.2
RADIUS = 2
PROXIMAL_REWARD = 0.2
SCORE_REWARD = 100


class BattleshipEnvClass(gym.Env):
    """ BattleshipEnvClass v7 (inherits from v6).

        action_space is a index representing grid coordinate to probe next
        obs_space is the entire (n x n) grid with discrete values (-1, 0, 1)

    """

    def __init__(self):

        self.board_dim = BOARD_DIM

        # Action space is index of action for grid.flatten()
        #   get i, j with i, j = (action % BOARD_DIM, action // BOARD_DIM)
        self.action_space = gym.spaces.Discrete(BOARD_DIM * BOARD_DIM)

        # Observation space is an integer array that summarizes knowledge of each
        #    grid block according to:   {0: unknown, 1: hit, -1: miss}
        self.observation_space = gym.spaces.Box(low=-1, high=1,
                                                shape=(BOARD_DIM, BOARD_DIM), dtype=np.int32)

        self.reset()

    def step(self, action):

        state_prev = np.copy(self.state)
        action = (action % BOARD_DIM, action // BOARD_DIM)

        if state_prev[action[0], action[1]] != 0:
            # - reward if torpedoing an already torpedo'd grid space (repeated penalty)
            return self.state, REPEATED_PENALTY, self.done, {}

            ####################
        # ADVANCE ENVIRONMENT -- Produce next state, check done condition
        hit = self.board.torpedo(action)

        if hit == 0:
            self.state[action[0], action[1]] = -1
        elif hit == 1:
            self.state[action[0], action[1]] = 1
        else:
            raise ValueError("Invalid return from board.torpedo(), f{hit}")

        self.done = self.board.check_gameover()

        ####################
        # REWARD CALCULATION
        reward = PERSISTENCE_PENALTY

        if hit:
            reward += HIT_REWARD

        # + reward if next torpedo is near a known hit grid space (proximal reward)
        neighbors = self._neighbors(action[0], action[1], RADIUS, self.board_dim)
        for neigh in neighbors:
            if self.state[neigh[0], neigh[1]] == 1:
                reward += PROXIMAL_REWARD

        if self.done:
            # score = self.board.score()
            reward += SCORE_REWARD

        info = {}
        return self.state, reward, self.done, info

    def render(self):
        pass

    def reset(self):

        self.board = Board(dim=self.board_dim, ship_config='default', vis=False, playmode=False)
        self.state = np.zeros((self.board_dim, self.board_dim), dtype=np.int32)
        self.done = False

        return self.state

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _neighbors(self, i, j, radius, dim):

        neighbors = list()
        for idx in range(radius):
            rad = idx + 1
            neighbors.extend([(i + rad, j),
                              (i, j + rad),
                              (i - rad, j),
                              (i, j - rad),
                              (i + rad, j + rad),
                              (i + rad, j - rad),
                              (i - rad, j + rad),
                              (i - rad, j - rad),
                              ])

        out = list()
        for neighbor in neighbors:
            if (0 <= neighbor[0] < dim) and (0 <= neighbor[1] < dim):
                out.append(neighbor)
            else:
                pass
        return out

    def _overwrite_board(self, board):
        self.board = board
