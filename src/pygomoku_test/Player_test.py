import unittest
import inspect
from pygomoku import Player
from pygomoku.Board import Board

class TestGomokuHumanPlayer(unittest.TestCase):
    def setUp(self):
        self.player = Player.HumanPlayer(Board.kPlayerBlack)
        self.board = Board()
    
    def test_color(self):
        self.assertEqual(Board.kPlayerBlack, self.player.color, 'Get error in {} when test color property.'.format(__file__))
        self.player.color = Board.kPlayerWhite
        self.assertEqual(Board.kPlayerWhite, self.player.color, 'Get error in {} when test color property.'.format(__file__))

    def test_name(self):
        self.assertEqual(self.player.name, Player.HumanPlayer.kDefaultName, 'Get error in {} when test name property.'.format(__file__))
        self.player.name = "Jack"
        self.assertEqual(self.player.name, "Jack", 'Get error in {} when test name property.'.format(__file__))
