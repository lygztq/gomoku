#ifndef CPPGOMOKU_STATE_H
#define CPPGOMOKU_STATE_H

#include <vector>
#include <utility>

namespace gomoku
{ 
    struct Location {
        int h_index;
        int w_index;
    };

    struct State
    {
        std::vector<int> state;
        int height;
        int width;

        State() {}
        State(int h, int w, int color);
        State(const State &s);
        State(State && s);
        State & operator=(const State &s);
        State & operator=(State && s);
        State singleColorState(int color);
        void flush(int color);
        int & get(Location l);
        int & get(int h_index, int w_index);
        int & get(int move);
    };
    
} // gomuku


#endif // CPPGOMOKU_STATE_H