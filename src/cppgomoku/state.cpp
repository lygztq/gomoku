#include "cppgomoku/state.h"
namespace gomoku
{
    State::State(int h, int w, int color) {
        height = h; width = w;
        state = std::vector<int>(h * w + 1, color);
    }

    State::State(const State &s) {
        height = s.height;
        width = s.width;
        state = s.state;
    }

    State::State(State && s) {
        height = s.height;
        width = s.width;
        state = std::move(s.state);
    }

    State & State::operator=(const State &s) {
        if(&s == this) return *this;

        this->state = s.state;
        this->width = s.width;
        this->height = s.height;
        return *this;
    }

    State & State::operator=(State && s) {
        height = s.height;
        width = s.width;
        state = std::move(s.state);
        return *this;
    }

    State State::singleColorState(int color) {
        State single_color_state(height, width, 0);

        int len = height * width;
        for (int i=0; i<len; ++i) {
            if (state[i] == color) {
                single_color_state.state[i] = 1;
            }
        }
        return single_color_state;
    }

    void State::flush(int color) {
        for (int &s: state) {
            s = color;
        }
    }

    int & State::get(Location l) {
        return get(l.h_index, l.w_index);
    }

    int & State::get(int h_index, int w_index) {
        return state[width * h_index + w_index];
    }

    int & State::get(int move) {
        return state[move];
    }
    
} // gomoku
