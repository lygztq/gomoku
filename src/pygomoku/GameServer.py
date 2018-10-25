import numpy as np
from pygomoku.Board import Board
from pygomoku import Player

def change_color(color):
    if color == Board.kPlayerBlack:
        return Board.kPlayerWhite
    elif color == Board.kPlayerWhite:
        return Board.kPlayerBlack
    else:
        return color

class GameServer(object):
    """The server of gomoku game.

    The GameServer class determines which class of game will go on
    and control the game.

    Attributes:
        board: A pygomoku.Board.Board instance, the board we use during game.
        mode: The mode of game. Valid values are GameServer.kSelfPlayGame and 
            GameServer.kNormalGame.
        player1: A pygomoku.Player instance. Player #1 who will play first.
        player2: A pygomoku.Player instance. Player #2. This player will be None if current game is self-play game.
        silent: A Boolean Value. If True, Game server will not print game info during game. 

    Constants:
        kSelfPlayGame: A constant. This value means current game is a self-play
            game which played by computer itself.
        kNormalPlayGame: A constant. This value means current game is a normal 
            game with two players(either human or ai).

    """

    kSelfPlayGame = 0
    kNormalPlayGame = 1

    def __init__(self, board, mode, player1, player2=None, silent=False):
        self.board = board
        self.mode = mode
        self.silent = silent
        
        # Get player1
        if isinstance(player1, Player.Player):
            self.player1 = player1
            self.player1.silent = silent
        else:
            raise ValueError("Invalid type for player 1, expect pygomoku.Player.Player, get {}".format(type(player1)))

        # Get Player2, if self-play, set Player2 None.
        if mode == GameServer.kNormalPlayGame:
            if isinstance(player2, Player.Player):
                if player2.color == player1.color:
                    raise ValueError("Player 2 has same color with Player1 !")
                self.player2 = player2
                self.player2.silent = silent
            else:
                raise ValueError("Invalid type for player 2, expect pygomoku.Player.Player, get {}".format(type(player2)))
        elif mode == GameServer.kSelfPlayGame:
            player2 = None
        else:
            raise ValueError("Invalid value for 'mode' attribute of GameServer.")
    
    def showGameInfo(self):
        """Show game informations.

        If game mode is normal play, this method will print player
        information first and then the board state.
        If game mode is self-play, only the board state will be printed.
        """
        if self.mode == GameServer.kNormalPlayGame:
            if isinstance(self.player1, Player.Player):
                print("[Player1]:\n-------------")
                print(self.player1)
                print('\n')
            if isinstance(self.player2, Player.Player):
                print("[Player2]:\n-------------")
                print(self.player2)
                print('\n')
        else:
            print("Self Play\n")
        self.board.printBoard()
    
    def _startNormalGame(self):
        """Start a normal game.
        
        Can be human vs. human or human vs. AI.
        Return color of the winner.
        """
        # initial board
        self.board.initBoard(self.player1.color)
        
        # key: color, value: player
        players = {
            self.player1.color: self.player1,
            self.player2.color: self.player2
        }

        stone_color = {
            Board.kPlayerBlack: "@",
            Board.kPlayerWhite: "O"
        }

        if not self.silent:
            self.showGameInfo()
        
        # The main process of a game
        while True:
            current_player = players[self.board.current_player]
            move = current_player.getAction(self.board)
            while not self.board.play(move):
                print("Invalid movement.")
            if not self.silent:
                self.showGameInfo()
            is_end, winner = self.board.gameEnd()
            if is_end:
                if not self.silent:
                    if winner is not None:
                        print("Game end with winner {}(color {}).".format(players[winner].name, stone_color[winner]))
                    else:
                        print("Game end with no winner.")
                return winner
    
    def _startSelfPlayGame(self):
        """Start a self-play game using a MCTS player.

        This method will reuse the search tree and store the 
        self-play data: (state, mcts_probs, z) for training

        Return:
            winner: An integer whose value can be Board.kPlayerBlack,
                Board.kPlayerWhite or None.
            states_batch: A numpy array with shape (N, 4, board_height, board_width)
                which shows the state of board during the game.
            action_probs_batch: A numpy array with shape (N, board_height*board_width)
                which shows the probability of each move of each step during game.
            winner_vec: A numpy array with shape (N, ) which shows the winner of the game, also
                represents the evaluate value of each state of board.
        """
        self.board.initBoard()
        states_batch, action_probs_batch, current_players_batch = [], [], []
        while True:
            move, probs = self.player1.getAction(self.board,
                                                   return_policy_vec=True)
            # Get training data
            states_batch.append(self.board.currentState())
            action_probs_batch.append(probs)
            current_players_batch.append(self.board.current_player)
            
            # move
            self.board.play(move)
            self.player1.color = change_color(self.player1.color)
            if not self.silent:
                self.showGameInfo()
            
            is_end, winner = self.board.gameEnd()
            if is_end:
                winner_vec = np.zeros(len(current_players_batch))
                if winner is not None: # if has winner
                    winner_vec[np.array(current_players_batch) == winner] = 1.0
                    winner_vec[np.array(current_players_batch) != winner] = -1.0
                
                self.player1.reset()
                if not self.silent:
                    if winner is not None:
                        print("Game end with winner [{}]".format(Board.kStoneChar[winner]))
                    else:
                        print("Game end with no winner.")
                # return winner, zip(states_batch, action_probs_batch, winner_vec)
                return winner, np.array(states_batch), np.array(action_probs_batch), winner_vec

    def startGame(self):
        """Start the game.
        """
        if self.mode == GameServer.kNormalPlayGame:
            return self._startNormalGame()
        else:
            return self._startSelfPlayGame()
    
