#ifndef GOMOKU_MCTS_H
#define GOMOKU_MCTS_H

#include <unordered_map>
#include <memory>
#include <limits>
#include "cppgomoku/common.h"
#if DEBUG == 1
#include <algorithm>
#endif
#include "cppgomoku/board.h"
#include "cppgomoku/policy_functions.h"

namespace gomoku
{
    typedef decltype(MCTS_Expand_policy_fn) expandFunc;
    typedef decltype(rollout_policy_fn) rolloutFunc;

    struct MCTSTreeNode {
        MCTSTreeNode * parent;
        // move->child map
        std::unordered_map<int, MCTSTreeNode*> children;
        int visit_times;
        float Q_value;
        float U_value;
        float probability;
 
        MCTSTreeNode(){
            parent = nullptr; visit_times = 0; probability=0.0;
            Q_value = 0.0; U_value = 0.0;
        }
        MCTSTreeNode(MCTSTreeNode* parent, float prior_prob);
        ~MCTSTreeNode();
        void kill();
        MCTSTreeNode * leaveOneChild(int move);
        void expand(const std::vector<MoveProbPair> &policy);
        void expand(std::vector<MoveProbPair> &&polcy);
        void backPropagation(const float bp_value);
        float evaluate(float weight_c);
        void update(float bp_value);
        MCTSTreeNode* select(float weight_c, int &action);
        bool isLeaf() {return children.empty();}
        bool isRoot() {return parent == nullptr;}
        float getQValue() {return Q_value;}
    };

    class MonteCarloSearchTreeBase {
    public:
        MonteCarloSearchTreeBase() {}
        virtual ~MonteCarloSearchTreeBase() {}
        virtual void playout(Board &s) = 0;
        virtual void reset() = 0;
        virtual int getMove(Board &s, float exploration_level=1) = 0;
        virtual void updateWithMove(int last_move) = 0;
        virtual MCTSTreeNode * getRoot() = 0;
    };

    class PureMonteCarloSearchTree : public MonteCarloSearchTreeBase {
    private:
        MCTSTreeNode * root;
        float weight_c;
        int compute_budget;
        bool silent;
        int expand_bound;
        int rollout_limit;
        expandFunc *expand_func;
        rolloutFunc *rollout_func;
    
    public:
        PureMonteCarloSearchTree(){}
        PureMonteCarloSearchTree(float weight_c, int compute_budget, 
                                 int expand_bound, bool silent, 
                                 int rollout_limit, expandFunc *expand_fn,
                                 rolloutFunc *rollout_fn);
        ~PureMonteCarloSearchTree() {delete root;}
        void reset();
        float evaluateRollout(Board &board, int limit);
        void playout(Board &s);
        int getMove(Board &s, float exploration_level=1);
        void updateWithMove(int last_move);
        MCTSTreeNode * getRoot() {return root;}
        void setSilent() {silent = true;}
        void unsetSilent() {silent = false;}
        int getComputeBudget() {return compute_budget;}
    };
} // gomoku


#endif // GOMOKU_MCTS_H