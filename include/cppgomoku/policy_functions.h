#ifndef CPPGOMOKU_POLICY_FUNCTIONS_H
#define CPPGOMOKU_POLICY_FUNCTIONS_H

#include <vector>
#include <random>

#include "cppgomoku/board.h"
#include "cppgomoku/common.h"

namespace gomoku
{
    std::vector<MoveProbPair> rollout_policy_fn(Board &board); 
    std::vector<MoveProbPair> MCTS_Expand_policy_fn(Board &board);
} // gomoku


#endif // CPPGOMOKU_POLICY_FUNCTIONS_H