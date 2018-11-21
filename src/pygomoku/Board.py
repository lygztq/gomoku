# coding=utf-8
from __future__ import print_function
import numpy as np


class Board(object):
    """Board class for training.
    
    Attributes:
        width: The width of board.
        height: The height of board.
        numberToWin: How many stones need on a line to win
    """
    kPlayerWhite = 0
    kPlayerBlack = 1
    kEmpty = -1

    kStoneChar = {
        1: '@',
        0: 'O',
        -1: '+'
    }

    def __init__(self, **kwargs):
        """
        @param kwargs: The dictionary of args
            width:          The width of board
            height:         The height of board
            numberToWin:    How many stones need on a line to win
        """
        self.__width = int(kwargs.get('width', 15))
        self.__height = int(kwargs.get('height', 15))
        self.numberToWin = int(kwargs.get('numberToWin', 5))
        # states: board states stored as dictionary
        # key: moves as location on the board
        # value: player as pieces type
        self.initBoard()

    def initBoard(self, start_player=None):
        # Do some check
        if start_player == None:
            start_player = Board.kPlayerBlack
        if self.__width < self.numberToWin or self.__height < self.numberToWin:
            raise Exception("Board width({}) or height({}) can not be less than {}".format(
                self.__width, self.__height, self.numberToWin))

        self.__current_player = start_player
        self.availables = list(
            range(self.__width * self.__height))  # Valid moves
        self.moved = []  # Moves that already have stone on it
        self.states = {}
        self.__last_move = None  # Last position

    def isValidMove(self, move):
        if not isinstance(move, int):
            return False

        if move < 0 or move >= self.__width * self.__height:
            return False

        if move in self.moved:
            return False

        return True

    def moveToLocation(self, move):
        """
        Two different types of index: move and location
        move = h_index * width + w_index
        location = [h_index, w_index]

        NOTE: If input move is None or invalid, this method will return None
        """
        if move is None:
            return None
        if move < 0 or move >= self.__width * self.__height:
            return None
        h_index = move // self.__width
        w_index = move % self.__width
        return [h_index, w_index]

    def locationToMove(self, location):
        if location is None:
            return None
        if len(location) != 2:
            return None
        h_index, w_index = location
        move = h_index * self.__width + w_index

        if move < 0 or move >= self.__width * self.__height:
            return None

        return move

    def __changePlayer(self):
        self.__current_player = 1 - self.__current_player

    def play(self, move):
        if move in self.moved:
            return False
        self.states[move] = self.__current_player
        self.availables.remove(move)
        self.moved.append(move)
        self.__changePlayer()
        self.__last_move = move
        return True

    def undo(self):
        if not self.moved:
            return False
        del self.states[self.__last_move]
        self.availables.append(self.__last_move)
        self.moved = self.moved[:-1]
        self.__changePlayer()
        if self.moved:
            self.__last_move = self.moved[-1]
        else:
            self.__last_move = None
        return True

    def currentState(self):
        """
        Return the board state from the perspective of the current player.
        state shape: 4 * height * width

        board_state[0]: current board state with only current player's stone
        board_state[1]: current board state with only opponent's stones
        board_state[2]: only one stone, indicate the last move(opponent made this move).
        board_state[3]: indicate the player to play, 0 for white, 1 for black
        """
        board_state = np.zeros((4, self.__height, self.__width))

        if self.states:  # if self.states is not empty
            moves, players = np.array(list(zip(*self.states.items())))
            curr_moves = moves[players == self.__current_player]
            oppo_moves = moves[players != self.__current_player]
            board_state[0][curr_moves // self.__width,
                           curr_moves % self.__width] = 1
            board_state[1][oppo_moves // self.__width,
                           oppo_moves % self.__width] = 1
            board_state[2][self.__last_move // self.__width,
                           self.__last_move % self.__width] = 1

        if self.__current_player == Board.kPlayerBlack:
            board_state[3] += 1
        
        return board_state

    def fastGetWinner(self):
        """
        If the game is plain sailing, i.e. the only operation is play stone and remove stone from board,
        then the last move will end the game, and only the last move can determine the winner.
        """
        if len(self.moved) < 2*self.numberToWin-1:  # No player has put numberToWin stones on the board
            return None

        # Horizontal
        for m in self.moved[::-1][:2]:
            last_player = self.states[m]
            # Here we try to find a interval whose elements are have same color with last_player, the interval is [)
            left_bd = self.__last_move
            right_bd = self.__last_move + 1
            while self.states.get(left_bd-1, Board.kEmpty) == last_player and left_bd % self.__width != 0:
                left_bd -= 1
            while self.states.get(right_bd, Board.kEmpty) == last_player and right_bd % self.__width != 0:
                right_bd += 1
            if (right_bd - left_bd) >= self.numberToWin:
                return last_player

            # Vertical
            left_bd = self.__last_move
            right_bd = self.__last_move + self.__width
            count = 1
            while self.states.get(left_bd - self.__width, Board.kEmpty) == last_player:
                left_bd -= self.__width
                count += 1
            while self.states.get(right_bd, Board.kEmpty) == last_player:
                right_bd += self.__width
                count += 1
            if count >= self.numberToWin:
                return last_player

            # main diagonal
            left_bd = self.__last_move
            right_bd = self.__last_move + self.__width + 1
            count = 1
            while self.states.get(left_bd - 1 - self.__width, Board.kEmpty) == last_player and left_bd % self.__width != 0:
                left_bd -= (self.__width + 1)
                count += 1
            while self.states.get(right_bd, Board.kEmpty) == last_player and right_bd % self.__width != 0:
                right_bd += (self.__width + 1)
                count += 1
            if count >= self.numberToWin:
                return last_player

            # deputy diagonal
            left_bd = self.__last_move
            right_bd = self.__last_move + self.__width - 1
            count = 1
            while self.states.get(left_bd + 1 - self.__width, Board.kEmpty) == last_player and left_bd % self.__width != self.__width - 1:
                left_bd -= (self.__width - 1)
                count += 1
            while self.states.get(right_bd, Board.kEmpty) == last_player and right_bd % self.__width != self.__width - 1:
                right_bd += (self.__width - 1)
                count += 1
            if count >= self.numberToWin:
                return last_player

        return None

    def getWinner(self):
        """
        Give a board and determine the winner. If no winner, return None.
        """
        if len(self.moved) < 2*self.numberToWin-1:  # No player has put numberToWin stones on the board
            return None

        for m in self.moved[::-1]:
            curr_player = self.states[m]

            # Horizontal
            # Here we try to find a interval whose elements are have same color with last_player, the interval is [)
            left_bd = m
            right_bd = m + 1
            while self.states.get(left_bd-1, Board.kEmpty) == curr_player and left_bd % self.__width != 0:
                left_bd -= 1
            while self.states.get(right_bd, Board.kEmpty) == curr_player and right_bd % self.__width != 0:
                right_bd += 1
            if (right_bd - left_bd) >= self.numberToWin:
                return curr_player

            # Vertical
            left_bd = m
            right_bd = m + self.__width
            count = 1
            while self.states.get(left_bd - self.__width, Board.kEmpty) == curr_player:
                left_bd -= self.__width
                count += 1
            while self.states.get(right_bd, Board.kEmpty) == curr_player:
                right_bd += self.__width
                count += 1
            if count >= self.numberToWin:
                return curr_player

            # main diagonal
            left_bd = m
            right_bd = m + self.__width + 1
            count = 1
            while self.states.get(left_bd - 1 - self.__width, Board.kEmpty) == curr_player and left_bd % self.__width != 0:
                left_bd -= (self.__width + 1)
                count += 1
            while self.states.get(right_bd, Board.kEmpty) == curr_player and right_bd % self.__width != 0:
                right_bd += (self.__width + 1)
                count += 1
            if count >= self.numberToWin:
                return curr_player

            # deputy diagonal
            left_bd = m
            right_bd = m + self.__width - 1
            count = 1
            while self.states.get(left_bd + 1 - self.__width, Board.kEmpty) == curr_player and left_bd % self.__width != self.__width - 1:
                left_bd -= (self.__width - 1)
                count += 1
            while self.states.get(right_bd, Board.kEmpty) == curr_player and right_bd % self.__width != self.__width - 1:
                right_bd += (self.__width - 1)
                count += 1
            if count >= self.numberToWin:
                return curr_player
        return None

    def gameEnd(self):
        """
        Check whether the game is terminal. Return a boolean value and the winner. 
        If no winner, return None.
        """
        winner = self.fastGetWinner()
        if winner is not None:
            return True, winner
        elif not self.availables:
            return True, None
        else:
            return False, None

    def printBoard(self):
        print("Current turn: [{}]".format(Board.kStoneChar[self.__current_player]))
        curr_state = np.zeros([self.__width, self.__height],
                              dtype=np.int) + Board.kEmpty
        if self.__last_move:
            last_h_idx, last_w_idx = self.moveToLocation(self.__last_move)
        else:
            last_h_idx, last_w_idx = -1, -1

        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            black_moves = moves[players == Board.kPlayerBlack]
            white_moves = moves[players == Board.kPlayerWhite]
            curr_state[black_moves // self.__width, black_moves %
                       self.__width] = Board.kPlayerBlack
            curr_state[white_moves // self.__width, white_moves %
                       self.__width] = Board.kPlayerWhite

        for w in range(self.__width):
            print("{0:8d}".format(w), end='')
        print('\n\n')
        for h in range(self.__height):
            print("{0:4d}".format(h), end='')
            for w in range(self.__width):
                if curr_state[h, w] == Board.kEmpty:
                    print("+".center(8), end='')
                elif curr_state[h, w] == Board.kPlayerBlack:
                    if h == last_h_idx and w == last_w_idx:
                        print("[@]".center(8), end='')
                    else:
                        print("@".center(8), end='')
                else:
                    if h == last_h_idx and w == last_w_idx:
                        print("[O]".center(8), end='')
                    else:
                        print("O".center(8), end='')
            print("{0:4d}".format(h), end='\n\n\n')
        for w in range(self.__width):
            print("{0:8d}".format(w), end='')
        print('\n')

    @property
    def current_player(self):
        return self.__current_player

    @property
    def last_move_location(self):
        return self.moveToLocation(self.__last_move)

    @property
    def last_move(self):
        return self.__last_move

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    @property
    def is_empty(self):
        if self.moved:
            return False
        else:
            return True

    @staticmethod
    def randomPlayer():
        return np.random.randint(0,2)
    
    @staticmethod
    def opponent(player_color):
        return 1 - player_color