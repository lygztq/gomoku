#include "cppgomoku/common.h"
#include "cppgomoku/player.h"
#include "cppgomoku/board.h"
#include "cppgomoku/game_server.h"

using namespace gomoku;

bool define_player(Player *&p, char type, int color) {
    switch (type) {
            case 'h': 
                p = new HumanPlayer(color, "Human Player"); 
                break;
            case 'c':
                p = new PureMCTSPlayer(color, "Pure MCTS player", 10.0, 80000); 
                break;
            default:
                printf("Wrong type parameter for player, expect \'h\' or \'c\'\n");
                return false;
        }
    return true;
}

int main(int argc, char *argv[]) {
    // parse parameters
    // first parameter is the type of player1, 'h' for human, 'c' for computer.
    // second parameter is the type of player2 'h' for human, 'c' for computer.
    if (argc < 3) {
        printf("At least two parameters!\n");
        return 0;
    } 
    char player1_type = argv[1][0];
    char player2_type = argv[2][0];
    Player *player1;
    Player *player2;

    // define player    
    if (!define_player(player1, player1_type, Board::kPlayerBlack)) return 0;
    if (!define_player(player2, player2_type, Board::kPlayerWhite)) return 0;

    Board b(9,9,5);
    GameServer gs(&b, player1, player2, false);
    gs.startGame();

    delete player1;
    delete player2;

    return 0;
}