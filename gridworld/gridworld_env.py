"""
Gridworld environment with three modes: static, player, random.

Adapted from DeepReinforcementLearningInAction/Chapter 3/Gridworld.py

Modes
-----
- static : all pieces fixed (Player at (0,3), Goal at (0,0), Pit at (0,1), Wall at (1,1))
- player : only player position is randomised; goal/pit/wall stay fixed
- random : all four pieces placed randomly each episode
"""

import numpy as np
from .grid_board import GridBoard, randPair, addTuple

# Action mapping: integer → direction offset (row, col)
ACTION_MAP = {
    0: (-1, 0),   # up
    1: (1, 0),    # down
    2: (0, -1),   # left
    3: (0, 1),    # right
}
ACTION_NAMES = ['up', 'down', 'left', 'right']


class Gridworld:
    """
    4×4 GridWorld with Player (P), Goal (+), Pit (-), and Wall (W).

    Parameters
    ----------
    size : int
        Board side length (minimum 4).
    mode : str
        One of 'static', 'player', 'random'.
    """

    def __init__(self, size=4, mode='static'):
        self.size = max(size, 4)
        self.mode = mode
        self.board = GridBoard(size=self.size)

        # Add pieces (positions updated by init methods)
        self.board.addPiece('Player', 'P', (0, 0))
        self.board.addPiece('Goal', '+', (1, 0))
        self.board.addPiece('Pit', '-', (2, 0))
        self.board.addPiece('Wall', 'W', (3, 0))

        self.reset()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _init_static(self):
        self.board.components['Player'].pos = (0, 3)
        self.board.components['Goal'].pos = (0, 0)
        self.board.components['Pit'].pos = (0, 1)
        self.board.components['Wall'].pos = (1, 1)

    def _init_player(self):
        """Random player start; goal/pit/wall stay at static positions."""
        self._init_static()
        self.board.components['Player'].pos = randPair(0, self.size)
        if not self._validate_board():
            self._init_player()

    def _init_random(self):
        """All pieces placed randomly."""
        self.board.components['Player'].pos = randPair(0, self.size)
        self.board.components['Goal'].pos = randPair(0, self.size)
        self.board.components['Pit'].pos = randPair(0, self.size)
        self.board.components['Wall'].pos = randPair(0, self.size)
        if not self._validate_board():
            self._init_random()

    def _validate_board(self):
        """Ensure no overlapping pieces and the board is winnable."""
        positions = [
            self.board.components['Player'].pos,
            self.board.components['Goal'].pos,
            self.board.components['Pit'].pos,
            self.board.components['Wall'].pos,
        ]
        # No overlaps
        if len(set(positions)) < 4:
            return False

        # Player must not start on goal or pit
        if positions[0] == positions[1] or positions[0] == positions[2]:
            return False

        # Check corners are not trapped
        corners = [
            (0, 0), (0, self.size - 1),
            (self.size - 1, 0), (self.size - 1, self.size - 1),
        ]
        player_pos = self.board.components['Player'].pos
        goal_pos = self.board.components['Goal'].pos
        if player_pos in corners or goal_pos in corners:
            moves = [(0, 1), (1, 0), (-1, 0), (0, -1)]
            val_pl = [self._validate_move('Player', m) for m in moves]
            val_go = [self._validate_move('Goal', m) for m in moves]
            if 0 not in val_pl or 0 not in val_go:
                return False

        return True

    # ------------------------------------------------------------------
    # Move validation & execution
    # ------------------------------------------------------------------
    def _validate_move(self, piece, addpos):
        """
        0 = valid move, 1 = blocked (wall / out-of-bounds), 2 = pit (game lost).
        """
        new_pos = addTuple(self.board.components[piece].pos, addpos)
        wall = self.board.components['Wall'].pos
        pit = self.board.components['Pit'].pos

        if new_pos == wall:
            return 1
        if max(new_pos) > self.size - 1 or min(new_pos) < 0:
            return 1
        if new_pos == pit:
            return 2
        return 0

    def step(self, action):
        """
        Take an action (int 0–3) and return (next_state, reward, done).

        Actions: 0=up, 1=down, 2=left, 3=right
        """
        addpos = ACTION_MAP[action]
        result = self._validate_move('Player', addpos)
        if result in (0, 2):  # valid or pit (still move there)
            new_pos = addTuple(self.board.components['Player'].pos, addpos)
            self.board.movePiece('Player', new_pos)

        reward = self._reward()
        done = self.is_done()
        return self.get_state(), reward, done

    def _reward(self):
        player = self.board.components['Player'].pos
        goal = self.board.components['Goal'].pos
        pit = self.board.components['Pit'].pos
        if player == pit:
            return -10
        elif player == goal:
            return 10
        else:
            return -1

    def is_done(self):
        player = self.board.components['Player'].pos
        goal = self.board.components['Goal'].pos
        pit = self.board.components['Pit'].pos
        return player == goal or player == pit

    # ------------------------------------------------------------------
    # State representation
    # ------------------------------------------------------------------
    def get_state(self):
        """Return flattened (64,) float32 array — 4 channels × 4 × 4."""
        return self.board.render_np().flatten()

    def reset(self):
        """Re-initialise the board and return the initial state."""
        if self.mode == 'static':
            self._init_static()
        elif self.mode == 'player':
            self._init_player()
        else:
            self._init_random()
        return self.get_state()

    def display(self):
        """Pretty-print the board."""
        return self.board.render()
