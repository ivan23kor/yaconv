CXX := clang++
OBJ := $(patsubst %.cpp,%.o,$(wildcard *.cpp))

BLIS := -lblis

LIBS := $(BLIS)

yaconv: $(OBJ)
	$(CXX) $^ -o $@ $(LIBS)

.PHONY: main.o

main.o: main.cpp
	$(CXX) $(DIMS) -c $<

%.o: %.cpp
	$(CXX) -c $<

clean:
	$(RM) yaconv *.o
