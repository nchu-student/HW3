"""
GridBoard module — low-level board representation for the 4×4 GridWorld.

Adapted from DeepReinforcementLearningInAction/Chapter 3/GridBoard.py
"""

import numpy as np
import random


def randPair(s, e):
    """Return a random (row, col) pair in [s, e)."""
    return np.random.randint(s, e), np.random.randint(s, e)


def addTuple(a, b):
    """Element-wise addition of two tuples."""
    return tuple(x + y for x, y in zip(a, b))


class BoardPiece:
    """A single piece on the grid board."""

    def __init__(self, name, code, pos):
        self.name = name   # e.g. 'Player'
        self.code = code   # ASCII char for display, e.g. 'P'
        self.pos = pos      # (row, col) tuple


class GridBoard:
    """
    A size×size grid that holds named pieces.

    Supports:
    - render()    → human-readable 2D char array
    - render_np() → (num_pieces, size, size) one-hot numpy array
    """

    def __init__(self, size=4):
        self.size = size
        self.components = {}   # name → BoardPiece

    def addPiece(self, name, code, pos=(0, 0)):
        self.components[name] = BoardPiece(name, code, pos)

    def movePiece(self, name, pos):
        self.components[name].pos = pos

    def render(self):
        """Return a (size, size) unicode array for display."""
        dtype = '<U2'
        board = np.zeros((self.size, self.size), dtype=dtype)
        board[:] = ' '
        for name, piece in self.components.items():
            board[piece.pos] = piece.code
        return board

    def render_np(self):
        """
        Return a (num_pieces, size, size) numpy array.
        Each channel is a binary mask for one piece.
        """
        num_pieces = len(self.components)
        board = np.zeros((num_pieces, self.size, self.size), dtype=np.float32)
        for layer, (name, piece) in enumerate(self.components.items()):
            board[layer, piece.pos[0], piece.pos[1]] = 1.0
        return board
