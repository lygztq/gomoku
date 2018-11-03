#ifndef CPPGOMOKU_BOARD_H
#define CPPGOMOKU_BOARD_H

#include <unordered_set>
#include <vector>
#include <utility>
#include <algorithm>

// #include "cppgomoku/common.h"
#include "cppgomoku/common.h"
#include "cppgomoku/state.h"
namespace gomoku {
    class Board
    {
    public:
        const static int kPlayerWhite = 0;
        const static int kPlayerBlack = 1;
        const static int kPlayerEmpty = -1;
        const static int kOutOfBoard = -2;
        const static char kBlackStone = '@';
        const static char kWhiteStone = 'O';
        const static char kEmptyStone = '+';

    private:
        int mWidth;
        int mHeight;
        int mNumberToWin;
        int mCurrentPlayer;
        int mLastMove;
        std::unordered_set<int> mAvailables;
        std::vector<int> mMoved;
        State mState;

        void changePlayer() {
            mCurrentPlayer = (mCurrentPlayer == kPlayerBlack) ? kPlayerWhite : kPlayerBlack;
        }
        bool checkSingleMove(int move, int last_player);

    public:
        Board(int height, int width, int number_to_win);
        Board(const Board &b);
        Board & operator=(const Board &b);
        void initBoard(int start_player=Board::kPlayerBlack);
        bool isValidMove(int move);
        Location moveToLocation(int move);
        int locationToMove(Location &location);
        bool play(int move);
        bool undo();
        std::vector<State> currentState();
        int fastGetWinner();
        int getWinner();
        bool gameEnd(int &color);
        void printBoard();
        int currentPlayerColor();
        Location LastMoveLocation();
        int lastMove();
        int getWidth();
        int getHeight();
        std::unordered_set<int> & getAvailables() {return mAvailables;}
        State & getState() {return mState;}
        bool isEmpty();
        static int opponentColor(int color);
        static char stone(int color);
    };
} // end of namespace gomoku
#endif // CPPGOMOKU_BOARD_H