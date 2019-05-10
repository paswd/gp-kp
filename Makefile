FLAGS=-ccbin clang++-3.8 -std=c++11 --compiler-options -stdlib=libc++ -Wno-deprecated-gpu-targets
COMPILLER=nvcc
LIBS=-lm -lGL -lGLU -lGLEW -lglut

#all: lib start
all: start

#start: main.o
#	$(COMPILLER) $(FLAGS) -o da-lab4 main.o -L. lib/lib-z-search.a

start: main.cu
	$(COMPILLER) $(FLAGS) -o gp-kp main.cu $(LIBS)

test: test.cu
	$(COMPILLER) $(FLAGS) -o test test.cu $(LIBS)

clean:
	rm gp-kp
