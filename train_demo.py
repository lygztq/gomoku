import sys
sys.path.append('./src')

from pygomoku.Train import TrainServer
from pygomoku.mcts.Networks import SimpleCNN

config = TrainServer.readConfig('./test_config.json')

width = config["board_width"]
height = config["board_height"]

network = SimpleCNN(height, width)

ts = TrainServer(network, config)

ts.startTrain()
