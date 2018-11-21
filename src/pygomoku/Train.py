import json
import numpy as np
from math import log as ln
from datetime import datetime
from copy import deepcopy
from pygomoku.Board import Board
from pygomoku.GameServer import GameServer
from pygomoku.mcts.Networks import NeuralNetwork
from pygomoku.Player import DNNMCTSPlayer, PureMCTSPlayer
from pygomoku.mcts.progressbar import ProgressBar

class TrainServer(object):
    def __init__(self, network, config, reuse=False):
        self.config = config
        self.board = Board(width=config["board_width"],
                           height=config["board_height"],
                           numberToWin=config["number_to_win"])
        self.entropy_upper_bound = ln(config["board_width"] * config["board_height"])

        # check the network is a Neural network.
        if not isinstance(network, NeuralNetwork):
            raise TypeError("The type of given network is invaild.")

        # check size
        if (network.width != config["board_width"] or
                network.height != config["board_height"]):
            raise ValueError("The size of given neural network is not equal to"
                             "the size of board.")
        self.network = network
        self.reuse = reuse
        if reuse:
            print("Using last time network parameters.")
            self.network.restore(config["model_path"])

        self.player = DNNMCTSPlayer(Board.kPlayerBlack, self.network,
                                    weight_c=config["MCTS_exploration_weight"],
                                    compute_budget=config["MCTS_compute_budget"],
                                    exploration_level=config["player_exploration_level"],
                                    self_play=True)

        self.game_server = GameServer(self.board, GameServer.kSelfPlayGame,
                                      self.player, silent=True)
        self.state_batch_buffer = None
        self.policy_batch_buffer = None
        self.winner_vec_buffer = None
        self.validation_player_compute_budget = config["validation_player_compute_budget"]

        # self.learning_rate_magnitude = self.config["learning_rate_magnitude"]

    def dataAugment(self, state_batch, policy_batch, winner_vec):
        """Rotate and filp data.

        shape of state_batch: (N, 4, height, width)
        shape of policy_batch: (N, height*width)
        shape of winner_vac: (N, )
        """
        width = self.board.width
        height = self.board.height
        num_batch = state_batch.shape[0]
        aug_state_batch = np.zeros([8] + list(state_batch.shape))
        aug_policy_batch = np.zeros([8] + list(policy_batch.shape))
        aug_winner_vec = np.tile(winner_vec, 8)

        for i in range(1, 5, 1):
            # The shape of state: [4 x height x width]
            # rotation
            index = (i-1)*2
            aug_state_batch[index] = np.rot90(state_batch, i, (2, 3))
            aug_policy_batch[index] = np.rot90(policy_batch.reshape(
                num_batch, height, width), i, (1, 2)).reshape(num_batch, height*width)

            # filp left and right
            aug_state_batch[index+1] = np.flip(aug_state_batch[index], axis=-1)
            aug_policy_batch[index+1] = np.flip(aug_policy_batch[index].reshape(
                num_batch, height, width), axis=-1).reshape(num_batch, height*width)


        return (aug_state_batch.reshape(8*num_batch, 4, height, width), 
                aug_policy_batch.reshape(8*num_batch, height*width), 
                aug_winner_vec)

    def getTrainingData(self):
        """Get the training data for one epoch.
        """
        # first delete training data of the last epoch.
        if self.state_batch_buffer is not None:
            del self.state_batch_buffer
            del self.policy_batch_buffer
            del self.winner_vec_buffer

        self.state_batch_buffer = []
        self.policy_batch_buffer = []
        self.winner_vec_buffer = []

        game_per_epoch = self.config["game_per_epoch"]

        for i in range(game_per_epoch):
            TrainServer.log_output("[Collecting training data {}/{}]".format(i, game_per_epoch))
            TrainServer.resetPlayer(self.player, Board.kPlayerBlack)
            _, state_batch, policy_batch, winner_vec = self.game_server.startGame()
            aug_state_data, aug_policy_data, aug_winner_vec = self.dataAugment(
                state_batch, policy_batch, winner_vec)
            self.state_batch_buffer.append(aug_state_data)
            self.policy_batch_buffer.append(aug_policy_data)
            self.winner_vec_buffer.append(aug_winner_vec)

        self.state_batch_buffer = np.concatenate(self.state_batch_buffer)
        self.policy_batch_buffer = np.concatenate(self.policy_batch_buffer)
        self.winner_vec_buffer = np.concatenate(self.winner_vec_buffer)

    @staticmethod
    def resetPlayer(player, reset_color):
        player.reset()
        player.color = reset_color

    def networkUpdate(self):
        num_data = self.state_batch_buffer.shape[0]
        iter_per_epoch = self.config["iter_per_epoch"]

        for i in range(iter_per_epoch):
            mask = np.random.choice(num_data, self.config["batch_size"])
            curr_loss, curr_entropy = self.network.trainStep(
                self.state_batch_buffer[mask],
                self.policy_batch_buffer[mask],
                self.winner_vec_buffer[mask],
                self.config["base_learning_rate"]
            )
            TrainServer.log_output("[Iteration {}/{}] loss: {}\tentropy: {}/{}".format(i, iter_per_epoch, curr_loss, curr_entropy, self.entropy_upper_bound))
        
    def networkValidation(self):
        board = deepcopy(self.board)
        TrainServer.resetPlayer(self.player, Board.randomPlayer())
        self.player.self_play = False
        oppo_player = PureMCTSPlayer(Board.opponent(self.player.color), 
                                     compute_budget=self.validation_player_compute_budget)
        
        TrainServer.log_output("[validation...]")

        num_validation_game = self.config["num_validation_game"]
        num_win_game = 0
        for _ in range(num_validation_game):
            TrainServer.resetPlayer(self.player, Board.randomPlayer())
            TrainServer.resetPlayer(oppo_player, Board.opponent(self.player.color))

            if self.player.color == Board.kPlayerBlack:
                player1 = self.player
                player2 = oppo_player
            else:
                player1 = oppo_player
                player2 = self.player

            val_game_saver = GameServer(board, mode=GameServer.kNormalPlayGame, 
                                        player1=player1, player2=player2,
                                        silent=True)

            winner = val_game_saver.startGame()
            if winner == self.player.color:
                num_win_game += 1
        self.player.self_play = True
        return num_win_game*1.0 / num_validation_game

    def startTrain(self):
        TrainServer.log_output("[Start training...]")
        num_epoch = self.config["num_epoches"]
        save_every = self.config["save_every"]
        validate_every = self.config["validation_every"]
        best_win_rate = 0.0

        for i in range(num_epoch):
            print("\n---\n")
            TrainServer.log_output("[Epoch] ({}/{})".format(i+1, num_epoch))
            self.getTrainingData()
            self.networkUpdate()

            if not ((i+1) % validate_every):
                win_rate = self.networkValidation()
                TrainServer.log_output("win rate at Epoch {} is {}".format(i+1, win_rate))
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    TrainServer.log_output("New Best Model with winning rate: {}".format(best_win_rate))
                    self.network.save(self.config["best_model_path"])
                    if (best_win_rate == 1.0 and 
                        self.validation_player_compute_budget < 5000):
                        self.validation_player_compute_budget += 1000
                        best_win_rate = 0.0

            if not ((i+1) % save_every):
                self.network.save(self.config["model_path"])

    @staticmethod
    def readConfig(config_path):
        """Read the keyword configuration from
        a JSON configuration file. 
        """
        with open(config_path, 'r') as config_file:
            return json.load(config_file)
    
    @staticmethod
    def log_output(output):
        time_now = datetime.now()
        print("{}\t|\t{}".format(time_now, output))
