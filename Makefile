# Put the compute capability of your NVIDIA GPU here.
# Visit https://developer.nvidia.com/cuda-gpus for a partial list.
ARCH=sm_35

all: libspmat.a

libspmat.a: spmat.o Makefile
	if [ ! -f ./matio/lib/libmatio.a ]; then ./get_matio.sh; fi && \
	ar rcs libspmat.a spmat.o ./matio/src/*.o

spmat.o: spmat.cu spmat.h Makefile
	nvcc -O3 -arch=$(ARCH) -rdc=true -c -o spmat.o spmat.cu -I./matio/include

tester: tester.cu libspmat.a Makefile
	nvcc -O3 -arch=$(ARCH) -rdc=true -o tester tester.cu -L. -lspmat -lz -lm

clean:
	rm -fv tester
