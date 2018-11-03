#include "cppgomoku/common.h"


namespace gomoku
{
    MoveProbPair::MoveProbPair(int move, float prob) {
        this->move = move;
        this->prob = prob;
    }

    std::vector<float> softmax(const std::vector<float> &x) {
        float max_item = max<float>(x);

        std::vector<float> result(x.size(), 0.0);
        int len = (int)x.size();
        for (int i=0;i<len;++i) {
            result[i] = std::exp(x[i] - max_item);
        }

        float item_sum = sum<float>(result);
        for (float &i: result) i /= item_sum;

        return result;
    }

    float sigmoid(float x) {
        return (1.0 + std::tanh(x/2)) / 2;
    }

    std::vector<float> sigmoid_vec(const std::vector<float> &v) {
        int len = (int)v.size();
        std::vector<float> result(len, 0.0);
        
        for (int i=0; i<len; ++i) {
            result[i] = sigmoid(v[i]);
        }

        return result;
    }

    int most_likely_move(std::vector<MoveProbPair> &policy) {
        if (policy.empty()) return 0;

        int return_move = policy[0].move;
        float max_prob = 0.0;

        for(auto p: policy) {
            if (p.prob > max_prob) {
                max_prob = p.prob;
                return_move = p.move;
            }
        }

        return return_move;
    }

    int most_likely_move(std::vector<MoveProbPair> &&policy) {
        if (policy.empty()) return 0;

        int return_move = policy[0].move;
        float max_prob = 0.0;

        for(auto p: policy) {
            if (p.prob > max_prob) {
                max_prob = p.prob;
                return_move = p.move;
            }
        }

        return return_move;
    }

    std::vector<float> uniform_random_vector(int len) {
        std::vector<float> probs(len, 0.0);
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(0.0, 1.0);

        for(float &p: probs) 
            p = distribution(generator);

        return probs;
    }
} // gomoku
