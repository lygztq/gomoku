#ifndef GOMOKU_GAME_SERVER_H
#define GOMOKU_GAME_SERVER_H

#include "cppgomoku/board.h"
#include "cppgomoku/player.h"


namespace gomoku
{
    class GameServer {
    private:
        Board *board;
        bool silent;
        Player *player1;
        Player *player2;
    public:
        GameServer(Board *board, Player *p1, Player *p2, bool silent);
        ~GameServer(){}
        void showGameInfo();
        int startGame(); //return the color of winner.
    };
} // gomoku


#endif // GOMOKU_GAME_SERVER_H