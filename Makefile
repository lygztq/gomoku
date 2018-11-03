# compile params
CXX := g++

PROJECT_NAME := cppgomoku

OBJ_DIR := ./obj

BIN_DIR := ./bin

SRC_DIR := ./src

vpath %.cpp $(SRC_DIR)/$(PROJECT_NAME)

INC := -I ./include

SRCS := $(notdir $(wildcard $(SRC_DIR)/$(PROJECT_NAME)/*.cpp)) #board.cpp common.cpp game_server.cpp mcts.cpp player.cpp policy_functions.cpp state.cpp pure_mcts_game.cpp

OBJS := $(patsubst %.o, $(OBJ_DIR)/$(PROJECT_NAME)/%.o, $(SRCS:.cpp=.o))

TARGET := pure_mcts_game

all: $(BIN_DIR)/$(PROJECT_NAME)/$(TARGET)

$(BIN_DIR)/$(PROJECT_NAME)/$(TARGET):$(OBJS)
	mkdir -p $(BIN_DIR)/$(PROJECT_NAME)
	$(CXX) -O3 -o $@ $^

build_obj_dir:
	mkdir -p $(OBJ_DIR)/$(PROJECT_NAME)

$(OBJ_DIR)/$(PROJECT_NAME)/%.o:%.cpp build_obj_dir
	$(CXX) -O3 $(INC) -c -o $@ $<

.PHONY : clean

clean : 
	rm -rf $(OBJ_DIR)/$(PROJECT_NAME)
	rm -rf $(BIN_DIR)/$(PROJECT_NAME)