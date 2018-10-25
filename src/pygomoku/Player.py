# coding=utf-8
from __future__ import print_function
import abc
import six
import sys
import numpy as np
from pygomoku.Board import Board
from pygomoku.mcts.MCTS import MCTS, MCTSWithDNN
from pygomoku.mcts.policy_fn import rollout_policy_fn, MCTS_expand_policy_fn
from pygomoku.mcts.Networks import SimpleCNN


@six.add_metaclass(abc.ABCMeta)
class Player(object):
    """
    The abstract class of player
    """
    @abc.abstractmethod
    def getAction(self, board):
        """
        Get the player's action
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        """
        Print some information about current player.
        """
        pass


class HumanPlayer(Player):
    """
    Human player
    """
    kDefaultName = "John Doe(Human Player)"
    kFromStdin = 0
    kFromGUI = 1

    def __init__(self, color, name=None, input_mode="stdin"):
        """
        @param color: The player's color.
        @param name: Name of player
        @param input_mode: "stdin" or "gui", how to get next move. Default is "stdin"
        """
        self.__color = color
        self.__name = name
        if self.__name is None:
            self.__name = HumanPlayer.kDefaultName
        if input_mode == "gui":
            self.__input_mode = HumanPlayer.kFromGUI
        else:
            self.__input_mode = HumanPlayer.kFromStdin

    def getActionViaGui(self, board):
        pass

    def getActionViaStdin(self, board):
        while True:
            if sys.version_info.major == 2:
                next_location = raw_input(
                    "Your movement(format:[vertical_index, horizontal_index], start from 0): ").split(',')
            else:
                next_location = input(
                    "Your movement(format:[vertical_index, horizontal_index], start from 0): ").split(',')
            next_location = list(map(int, next_location))

            next_move = board.locationToMove(next_location)
            if board.isValidMove(next_move):
                return next_move
            else:
                print("Invalid movement! Please try again.")

    def getAction(self, board):
        # check color
        if board.current_player != self.__color:
            raise RuntimeError("The current player's color in board is"
                "not equal to the color of current player.")

        if self.__input_mode == HumanPlayer.kFromGUI:
            return self.getActionViaGui(board)
        elif self.__input_mode == HumanPlayer.kFromStdin:
            return self.getActionViaStdin(board)

    def __str__(self):
        if self.__color == Board.kPlayerBlack:
            color = "Black[@]"
        elif self.__color == Board.kPlayerWhite:
            color = "White[O]"
        else:
            color = "None[+]"

        if self.__input_mode == HumanPlayer.kFromGUI:
            input_mode = "gui"
        else:
            input_mode = "stdin"
        return "[--Player Info--]\nHuman Player\nName: %s\nColor: %s\nInput Mode: %s" % (
            self.__name, color, input_mode
        )

    @property
    def color(self):
        return self.__color

    @color.setter
    def color(self, given_color):
        if given_color not in [Board.kPlayerBlack, Board.kPlayerWhite, Board.kEmpty]:
            return
        self.__color = given_color

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, given_name):
        if not isinstance(given_name, str):
            return
        self.__name = given_name

    __repr__ = __str__


class PureMCTSPlayer(Player):
    """
    Pure MCTS player
    """
    def __init__(self, color, name="Pure MCTS player", weight_c=5, compute_budget=10000, silent=False):
        self._search_tree = MCTS(MCTS_expand_policy_fn, rollout_policy_fn,
            weight_c=weight_c, compute_budget=compute_budget, silent=silent)
        self.__color = color
        self.__name = name
        self.__silent = silent
    
    def reset(self):
        self._search_tree.reset()
    
    def __str__(self):
        if self.__color == Board.kPlayerBlack:
            color = "Black[@]"
        elif self.__color == Board.kPlayerWhite:
            color = "White[O]"
        else:
            color = "None[+]"
        
        return "[--Player info--]\nPure MCTS Player\nName: {}\nColor: {}\nProperty: {}".format(
            self.__name, color, self._search_tree.__str__()
        )

    __repr__ = __str__
    
    def getAction(self, board):
        # check color
        if board.current_player != self.__color:
            raise RuntimeError("The current player's color in board is"
                "not equal to the color of current player.")

        # update the MCT with last move
        self._search_tree.updateWithMove(board.last_move)
        
        # get next move
        next_move = self._search_tree.getMove(board)
        self._search_tree.updateWithMove(next_move)
        return next_move
    
    def gaussNext(self, board, careless_level=100):
        """Gauss next move of opponent.
        """
        if board.current_player == self.__color:
            raise RuntimeError("A AI player cannot gauss move of itself")
        return self._search_tree.think(board, careless_level)

    @property
    def color(self):
        return self.__color
    @color.setter
    def color(self, given_color):
        if given_color not in [Board.kPlayerBlack, Board.kPlayerWhite, Board.kEmpty]:
            return
        self.__color = given_color

    @property
    def name(self):
        return self.__name    
    @name.setter
    def name(self, given_name):
        if not isinstance(given_name, str):
            return
        self.__name = given_name
    
    @property
    def silent(self):
        return self.__silent
    @silent.setter
    def silent(self, given_value):
        if isinstance(given_value, bool):
            self.__silent = given_value
            self._search_tree.silent = given_value

class DNNMCTSPlayer(Player):
    """
    MCTS player with DNN.

    Attributes:
        color: The stone color of current player
        name: Player's name.
        network: An instance of NeuralNetwork which will be used by player.
        search_tree: A Monte Corlo Search Tree.
        exploration_level: temperature parameter in (0, 1] controls 
                the level of exploration.
        self_play: If True, use self_play mode.
    """
    def __init__(self, color, network, name="DNN MCTS Player",
                 weight_c=5, compute_budget=10000, exploration_level=1e-4,
                 self_play=False, silent=False):
        self._color = color
        self._name = name
        self.network = network
        self._search_tree = MCTSWithDNN(network.policyValueFunc, weight_c, compute_budget, silent=silent)
        self._silent = silent
        self.exploration_level = exploration_level
        self._self_play = self_play
    
    def reset(self):
        self._search_tree.reset()

    def getAction(self, board, return_policy_vec=False):
        # check color
        if board.current_player != self._color:
            raise RuntimeError("The current player's color in board is"
                "not equal to the color of current player.")
        # check board size
        if (board.width != self.network.width or 
            board.height != self.network.height):
            raise ValueError("The size of network ({},{}) is not equal to the size of board({},{})".format(
                             self.network.height, self.network.width, board.height, board.width))
        # get next move
        actions, probs = self._search_tree.getMove(board, self.exploration_level)
        if self._self_play:
            # Add Dirichlet prior noise for training.
            move = np.random.choice(
                actions,
                p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
            )
            # update only once because the opponent(itself) will update too.
            self._search_tree.updateWithMove(move)
        else:   # play with true opponent
            self._search_tree.updateWithMove(board.last_move)
            move = np.random.choice(actions, p=probs)
            self._search_tree.updateWithMove(move)
        
        if return_policy_vec:
            policy_vec = np.zeros(board.width * board.height)
            policy_vec[list(actions)] = probs
            return move, policy_vec
        else:
            return move

    def __str__(self):
        if self._color == Board.kPlayerBlack:
            color = "Black[@]"
        elif self._color == Board.kPlayerWhite:
            color = "White[O]"
        else:
            color = "None[+]"
        
        return "[--Player info--]\nDNN MCTS Player\nName: {}\nColor: {}\nProperty: {}".format(
            self._name, color, self._search_tree.__str__()
        )

    __repr__ = __str__

    def gaussNext(self):
        pass
    
    @property
    def color(self):
        return self._color
    @color.setter
    def color(self, given_color):
        if given_color in [Board.kPlayerBlack, Board.kPlayerWhite]:
            self._color = given_color
    
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, given_name):
        if isinstance(given_name, str):
            self._name = given_name
    
    @property
    def silent(self):
        return self._silent
    @silent.setter
    def silent(self, given_value):
        if isinstance(given_value, bool):
            self._silent = given_value
            self._search_tree.silent = given_value
    
    @property
    def self_play(self):
        return self._self_play
    @self_play.setter
    def self_play(self, given_value):
        if isinstance(given_value, bool):
            self._self_play = given_value

    
