#include "cppgomoku/policy_functions.h"


namespace gomoku
{
    std::vector<MoveProbPair> rollout_policy_fn(Board &board) {
        int len = static_cast<int>(board.getAvailables().size());
        std::vector<MoveProbPair> policy(len, MoveProbPair());

        static std::random_device r;
        std::default_random_engine generator(r());
        std::uniform_real_distribution<float> dist(0.0, 1.0);

        int index = 0;
        for (int m: board.getAvailables()) {
            policy[index].move = m;
            policy[index++].prob = dist(generator);
        }

        return policy;
    }

    std::vector<MoveProbPair> MCTS_Expand_policy_fn(Board &board) {
        int len = static_cast<int>(board.getAvailables().size());
        float prob = 1.0 / len;

        std::vector<MoveProbPair> policy(len, MoveProbPair());
        int index = 0;
        for (int m: board.getAvailables()) {
            policy[index].move = m;
            policy[index++].prob = prob;
        }
        
        return policy;
    }
} // gomoku
