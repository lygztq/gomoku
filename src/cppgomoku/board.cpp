#include "cppgomoku/board.h"

namespace gomoku
{
    int Board::opponentColor(int color) {
        return (color == kPlayerBlack) ? kPlayerWhite : kPlayerBlack;
    }
    char Board::stone(int color)  {
        switch (color) {
            case kPlayerBlack: return kBlackStone;
            case kPlayerWhite: return kWhiteStone;
            case kPlayerEmpty: return kEmptyStone;
            default: return '?';
        }
    }

    Board::Board(int height, int width, int number_to_win)
    :mHeight(height), mWidth(width), mNumberToWin(number_to_win) {
        if (height < number_to_win || width < number_to_win) {
            throw std::length_error("Board width or height can not be" 
                                    "less than the number of stone to win.\n");
        }
        initBoard();
    }

    Board::Board(const Board &b)
    :mHeight(b.mHeight), mWidth(b.mWidth), mNumberToWin(b.mNumberToWin),
    mCurrentPlayer(b.mCurrentPlayer), mLastMove(b.mLastMove) {
        mAvailables = b.mAvailables;
        mMoved = b.mMoved;
        mState = b.mState;
    }

    Board & Board::operator=(const Board &b) {
        if (&b == this) return *this;

        this->mWidth = b.mWidth;
        this->mHeight = b.mHeight;
        this->mNumberToWin = b.mNumberToWin;
        this->mCurrentPlayer = b.mCurrentPlayer;
        this->mLastMove = b.mLastMove;
        this->mAvailables = b.mAvailables;
        this->mMoved = b.mMoved;
        this->mState = b.mState;
        return *this;
    }    

    void Board::initBoard(int start_player) {
        std::vector<int> available_range = std::move(range<int>(0, mHeight*mWidth));
        mCurrentPlayer = start_player;
        mAvailables = std::unordered_set<int>(available_range.begin(), available_range.end());
        mMoved = std::vector<int>();
        mState = State(mHeight, mWidth, kPlayerEmpty);
        mLastMove = Board::kPlayerEmpty;
    }

    bool Board::isValidMove(int move) {
        if ( move < 0 || move >= static_cast<int>(mWidth * mHeight) ) return false;

        if (std::find(mMoved.begin(), mMoved.end(), move) != mMoved.end()) return false;

        return true;
    }

    Location Board::moveToLocation(int move) {
        Location l;
        l.h_index = move / mWidth;
        l.w_index = move % mWidth;

        return l;
    }

    int Board::locationToMove(Location &location) {
        return location.h_index * mWidth + location.w_index;
    }

    bool Board::play(int move) {
        if (!isValidMove(move)) return false;
        mState.get(move) = mCurrentPlayer;
        mAvailables.erase(move);
        mMoved.push_back(move);
        changePlayer();
        mLastMove = move;
        return true;
    }

    bool Board::undo() {
        if (mMoved.empty()) return false;
        mState.get(mLastMove) = kPlayerEmpty;
        mAvailables.insert(mLastMove);
        mMoved.pop_back();
        changePlayer();

        if (!mMoved.empty()) mLastMove = mMoved.back();
        else mLastMove = kPlayerEmpty;
        return true;
    }

    std::vector<State> Board::currentState() {
        /** Return the board state from the perspective of the current player.
         *   state shape: 4 * height * width
         *   
         *   board_state[0]: current board state with only current player's stone
         *   board_state[1]: current board state with only opponent's stones
         *   board_state[2]: only one stone, indicate the last move(opponent made this move).
         *   board_state[3]: indicate the player to play, 0 for white, 1 for black
         */
        std::vector<State> ret_state_batch(4, State(mHeight, mWidth, 0));

        ret_state_batch[0] = mState.singleColorState(mCurrentPlayer);
        ret_state_batch[1] = mState.singleColorState(opponentColor(mCurrentPlayer));
        ret_state_batch[2].get(mLastMove) = 1;
        ret_state_batch[3].flush(mCurrentPlayer);

        return ret_state_batch;
    }

    bool Board::checkSingleMove(int move, int last_player) {
        int width = static_cast<int>(mWidth);
        int height = static_cast<int>(mHeight);
        int count = 0;

        // Horizontal
        int left_bd = move, right_bd = move + 1;
        while (left_bd >= 0 && mState.get(left_bd-1) == last_player && left_bd % width) --left_bd;
        while (right_bd < (mWidth*mHeight) && mState.get(right_bd) == last_player && right_bd % width) ++right_bd;
        if ((right_bd - left_bd) >= mNumberToWin) return true;

        // Vertical
        left_bd = move;
        right_bd = move + width;
        count = 1;
        while (left_bd >= 0 && mState.get(left_bd - width) == last_player) {
            left_bd -= width;
            ++count;
        }
        while (right_bd < (mWidth*mHeight) && mState.get(right_bd) == last_player) {
            right_bd += width;
            ++count;
        }
        if (count >= mNumberToWin) return true;

        // main diagonal
        left_bd = move;
        right_bd = move + width + 1;
        count = 1;
        while (left_bd >= 0 && mState.get(left_bd-1-width) == last_player && left_bd % width) {
            left_bd -= (width + 1);
            ++count;
        }
        while (right_bd < (mWidth*mHeight) && mState.get(right_bd) == last_player && right_bd % width) {
            right_bd += (width + 1);
            ++count;
        }
        if (count >= mNumberToWin) return true;

        // deputy diagonal
        left_bd = move;
        right_bd = move + width - 1;
        count = 1;
        while (left_bd >= 0 && mState.get(left_bd + 1 - width) == last_player && (left_bd%width) != width-1) {
            left_bd -= (width - 1);
            ++count;
        }
        while (right_bd < (mWidth*mHeight) && mState.get(right_bd) == last_player && (right_bd % width) != width-1) {
            right_bd += (width - 1);
            ++count;
        }
        if (count >= mNumberToWin) return true;

        return false;
    }

    int Board::fastGetWinner() {
        /** 
         * If the game is plain sailing, i.e. the only operation is play stone and remove stone from board,
         * then the last move will end the game, and only the last move can determine the winner.
         */
        if (mMoved.size() < 2 * mNumberToWin - 1) return kPlayerEmpty;

        int num_moved = static_cast<int>(mMoved.size());
        for(int i=num_moved-1; i>=num_moved-2; --i) {
            int move = mMoved[i];
            int count = 0;
            int last_player = mState.get(move);
            if(checkSingleMove(move, last_player)) return last_player;
        }
        
        return kPlayerEmpty;
    }

    int Board::getWinner() {
        if (mMoved.size() < 2 * mNumberToWin - 1) return kPlayerEmpty;

        int num_moved = static_cast<int>(mMoved.size());
        for(int i=num_moved-1; i >= 2*mNumberToWin-1; --i) {
            int move = mMoved[i];
            int count = 0;
            int last_player = mState.get(move);
            if(checkSingleMove(move, last_player)) return last_player;
        }

        return kPlayerEmpty;
    }

    bool Board::gameEnd(int &color) {
        color = fastGetWinner();
        if (color != kPlayerEmpty || mAvailables.empty()) {
            return true;
        }
        else {
            return false;
        }
    }

    void Board::printBoard() {
        printf("Current turn: [%c]\n", stone(mCurrentPlayer));
        Location last_location;
        if (mLastMove != kPlayerEmpty) 
            last_location = moveToLocation(mLastMove);
        else {
            last_location.h_index = kPlayerEmpty;
            last_location.w_index = kPlayerEmpty;
        }
        
        char *buffer = new char [50];
        char *stone_buffer = new char [5];
        for (int w=0; w<mWidth; ++w) 
            printf("%6d", w);
        printf("\n\n");

        for (int h=0; h<mHeight; ++h) {
            printf("%3d", h);
            for (int w=0; w<mWidth; ++w) {
                switch (mState.get(h, w)) {
                    case kPlayerEmpty:
                        print_center(buffer, kEmptyStone, 6);
                        break;
                    case kPlayerBlack:
                        if (h == last_location.h_index && 
                            w == last_location.w_index) {
                            sprintf(stone_buffer, "[%c]", kBlackStone);
                            print_center(buffer, stone_buffer, 6);
                        }
                        else 
                            print_center(buffer, kBlackStone, 6);
                        break;
                    case kPlayerWhite:
                        if (h == last_location.h_index && 
                            w == last_location.w_index) {
                            sprintf(stone_buffer, "[%c]", kWhiteStone);
                            print_center(buffer, stone_buffer, 6);
                        }
                        else 
                            print_center(buffer, kWhiteStone, 6);
                        break;
                }
            }
            printf("%-3d\n\n", h);
        }

        for (int w=0; w<mWidth; ++w) 
            printf("%6d", w);
        printf("\n\n\n");

        delete buffer;
        delete stone_buffer;
    }

    int Board::currentPlayerColor() {
        return mCurrentPlayer;
    }

    Location Board::LastMoveLocation() {
        return moveToLocation(mLastMove);
    }

    int Board::lastMove() {
        return mLastMove;
    }

    int Board::getWidth() {
        return mWidth;
    }
    
    int Board::getHeight() {
        return mHeight;
    }

    bool Board::isEmpty() {
        return mMoved.empty();
    }

} // gomoku
