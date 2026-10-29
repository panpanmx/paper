MPICC ?= /vol8/home/njust_ztc/hnu/software/mpich-4.2.0/bin/mpicc
ENV_ROOT ?= /vol8/appsoftware/mt3000_programming_env/hthreads
LIB_FLAGS = -L../bin/libs -Wl,-rpath=\$$ORIGIN/../bin/libs:/vol8/home/njust_ztc/hnu/jpp/gemm_mpi/bin/libs

all: gemm_openblas_bench

gemm_openblas_bench: gemm_openblas_bench.c ../host_code/operator.c
	$(MPICC) -O2 gemm_openblas_bench.c ../host_code/operator.c -std=c99 \
		-I../include -I$(ENV_ROOT)/include \
		$(ENV_ROOT)/lib/libhthread_host.a $(LIB_FLAGS) \
		-lopenblas -lpthread -fopenmp -lm \
		-o gemm_openblas_bench

clean:
	rm -f gemm_openblas_bench
