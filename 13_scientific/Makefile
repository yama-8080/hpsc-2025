# necessary to activate pytorch!
module:
	module load cuda
	module load forge
clean:
	rm p.dat u.dat v.dat a.out
10_cavity-py: 10_cavity.py
	python 10_cavity.py
10_cavity-cpp: 10_cavity.cpp
	g++ 10_cavity.cpp
	./a.out
10_cavity-cpp-debug: 10_cavity.cpp
	g++ 10_cavity.cpp -g
	ddt ./a.out
10_cavity-cuda: 10_cavity.cu
	nvcc 10_cavity.cu
	./a.out
10_cavity-cuda-debug: 10_cavity.cu
	nvcc 10_cavity.cu -g -G
	ddt ./a.out
plot: 10_plot.py
	python 10_plot.py