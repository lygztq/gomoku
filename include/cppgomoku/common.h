#ifndef CPPGOMOKU_COMMON_H
#define CPPGOMOKU_COMMON_H

#include <vector>
#include <cstdio>
#include <string>
#include <cstring>
#include <random>
#include <cmath>

#define DEBUG 0

namespace gomoku
{
    // common datatypes
    struct MoveProbPair {
        int move;
        float prob;

        MoveProbPair(){}
        MoveProbPair(int move, float prob);
        bool operator<(const MoveProbPair &p) {
            return this->prob < p.prob;
        }
        bool operator>(const MoveProbPair &p) {
            return this->prob > p.prob;
        }
        bool operator==(const MoveProbPair &p) {
            return this->prob == p.prob && this->move == p.move;
        }
    };

    enum class HumanPlayerInputMode : int {
        kFromStdin = 0,
        kFromGUI = 1,
        kFromOther = 2
    };

    // helper functions
    // math function
    std::vector<float> softmax(const std::vector<float> &x);

    float sigmoid(float x);

    std::vector<float> sigmoid_vec(const std::vector<float> &v);

    template<typename T>
    T sum(const std::vector<T> &vec) {
        T sum = 0;
        for (T i :vec) sum += i;
        return sum;
    }

    template<typename T>
    T max(const std::vector<T> &vec) {
        if (vec.empty()) return 0;
        T max_item = vec[0];
        for (T i: vec) {
            if (i > max_item) max_item = i;
        }
        return max_item;
    }

    int most_likely_move(std::vector<MoveProbPair> &policy);
    int most_likely_move(std::vector<MoveProbPair> &&policy);

    // range function
    template<typename T> 
    std::vector<T> range(T begin, T end) {
        std::vector<T> range_vec(end-begin, begin);
        int len = range_vec.size();

        for(int i=1; i<len; ++i) {
            range_vec[i] = begin + i;
        }

        return range_vec;
    }

    // random functions
    std::vector<float> uniform_random_vector(int len);

    // print function
    inline void print_center(char *buffer, std::string str, int width) {
        int c_pos = (width-1) / 2;
        int len = static_cast<int>(str.size());
        int left_space = c_pos - len/2;
        sprintf(buffer, "%%%dc%s%%%dc", left_space, str.c_str(), width-left_space-len);
        printf(buffer, ' ', ' ');
    }

    inline void print_center(char *buffer, char *str, int width) {
        int c_pos = (width-1) / 2;
        int len = strlen(str);
        int left_space = c_pos - len/2;
        sprintf(buffer, "%%%dc%s%%%dc", left_space, str, width-len-left_space);
        printf(buffer, ' ', ' ');
    }

    inline void print_center(char *buffer, char character, int width) {
        int pos = (width-1) / 2;
        sprintf(buffer, "%%%dc%c%%%dc", pos, character, width-1-pos);
        printf(buffer, ' ', ' ');
    }
} // end of namespace gomoku

#endif // CPPGOMOKU_COMMON_H