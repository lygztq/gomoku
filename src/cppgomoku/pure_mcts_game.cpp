#include "cppgomoku/common.h"
#include "cppgomoku/player.h"
#include "cppgomoku/board.h"
#include "cppgomoku/game_server.h"

using namespace gomoku;

int main() {
    Board b(9,9,5);
    HumanPlayer player1(Board::kPlayerBlack, "test_player");
    PureMCTSPlayer player2(Board::kPlayerWhite, "test AI player", 10.0, 50000);

    GameServer gs(&b, &player1, &player2, false);
    gs.startGame();
}