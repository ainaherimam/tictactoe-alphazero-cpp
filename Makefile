CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++20 -Wno-unused-parameter

# List of source files
SRCS = main.cpp board.cpp cell_state.cpp console_interface.cpp game.cpp player.cpp
# List of object files
OBJS = $(SRCS:.cpp=.o)

# Name of the output binary
TARGET = MCTS-Fanorona

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
