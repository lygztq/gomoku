import unittest

from pygomoku.Board import Board


class TestGomokuBoard(unittest.TestCase):
    def setUp(self):
        self.board = Board()

    def test_moveToLocation(self):
        error_report = "Got error in moveToLocation"
        self.assertEqual(self.board.moveToLocation(15), [1, 0], error_report)
        self.assertEqual(self.board.moveToLocation(0), [0, 0], error_report)
        self.assertEqual(self.board.moveToLocation(17), [1, 2], error_report)

    def test_initBoard(self):
        error_report = "Got error in initBoard"
        self.board.initBoard(Board.kPlayerWhite)
        self.assertEqual(self.board.current_player,
                         Board.kPlayerWhite, error_report + ' -> initial player')
        self.assertEqual(self.board.last_move, None,
                         error_report + " -> last move")
        self.assertEqual(self.board.width, 15, error_report + " -> width")
        self.assertEqual(self.board.height, 15, error_report + " -> height")

    def test_locationToMove(self):
        error_report = "Got error in locationToMove"
        self.assertEqual(self.board.locationToMove([1, 0]), 15, error_report)
        self.assertEqual(self.board.locationToMove([0, 0]), 0, error_report)
        self.assertEqual(self.board.locationToMove([1, 2]), 17, error_report)

    def test_play_and_undo(self):
        self.assertFalse(self.board.undo(),
                         "Got error in undo when moved is empty")
        self.board.play(17)
        self.assertEqual(self.board.last_move, 17, "Got error in move")
        self.assertFalse(self.board.play(
            17), "Got error in put stone on a stone")
        self.board.undo()
        self.assertEqual(self.board.last_move, None, "Got error in undo")

    def test_gameEnd(self):
        for i in range(4):
            self.board.play(i)
            self.board.play(i+self.board.width)
        self.board.play(4)
        self.assertListEqual(list(self.board.gameEnd()), [
                             True, Board.kPlayerBlack], "Got error in gameEnd")
        
        self.board.initBoard()

        for i in range(4):
            self.board.play(i+7*self.board.width)
            self.board.play(i+8*self.board.width)

        for i in range(2):
            self.board.play(i)
            self.board.play(i+3*self.board.width)
        self.board.play(2)
        self.board.play(3)
        self.board.play(4)
        dubug_result = self.board.gameEnd()
        self.assertEqual(list(self.board.gameEnd()), [False, None], "Got error in gameEnd")
