MPICC ?= /vol8/home/njust_ztc/hnu/software/mpich-4.2.0/bin/mpicc
ENV_ROOT=/vol8/appsoftware/mt3000_programming_env/hthreads
LIB_FLAGS=-L../bin/libs -Wl,-rpath=\$$ORIGIN/libs:\$$ORIGIN/../bin/libs:/vol8/home/njust_ztc/hnu/jpp/gemm_mpi/bin/libs

ALL: gemm_host_AB_opt gemm_host_AB_opt2 gemm_host_AB_opt3 gemm_host_AB_opt4 gemm_host_AB_opt5 gemm_host_AB_opt6 gemm_host_ASR

gemm_host_AB_opt: gemm_host_AB_opt.c operator.c
	$(MPICC) -O2 gemm_host_AB_opt.c operator.c -std=c99 -I./include -I$(ENV_ROOT)/include $(ENV_ROOT)/lib/libhthread_host.a $(LIB_FLAGS) -lpthread -fopenmp -o gemm_host_AB_opt -l m
	rm -f ../bin/gemm_host_AB_opt && cp gemm_host_AB_opt ../bin/

gemm_host_AB_opt2: gemm_host_AB_opt2.c operator.c
	$(MPICC) -O2 gemm_host_AB_opt2.c operator.c -std=c99 -I./include -I$(ENV_ROOT)/include $(ENV_ROOT)/lib/libhthread_host.a $(LIB_FLAGS) -lpthread -fopenmp -o gemm_host_AB_opt2 -l m
	rm -f ../bin/gemm_host_AB_opt2 && cp gemm_host_AB_opt2 ../bin/

gemm_host_AB_opt3: gemm_host_AB_opt3.c operator.c
	$(MPICC) -O2 gemm_host_AB_opt3.c operator.c -std=c99 -I./include -I$(ENV_ROOT)/include $(ENV_ROOT)/lib/libhthread_host.a $(LIB_FLAGS) -lpthread -fopenmp -o gemm_host_AB_opt3 -l m
	rm -f ../bin/gemm_host_AB_opt3 && cp gemm_host_AB_opt3 ../bin/

gemm_host_AB_opt4: gemm_host_AB_opt4.c operator.c
	$(MPICC) -O2 gemm_host_AB_opt4.c operator.c -std=c99 -I./include -I$(ENV_ROOT)/include $(ENV_ROOT)/lib/libhthread_host.a $(LIB_FLAGS) -lpthread -fopenmp -o gemm_host_AB_opt4 -l m
	rm -f ../bin/gemm_host_AB_opt4 && cp gemm_host_AB_opt4 ../bin/

gemm_host_AB_opt5: gemm_host_AB_opt5.c operator.c
	$(MPICC) -O2 gemm_host_AB_opt5.c operator.c -std=c99 -I./include -I$(ENV_ROOT)/include $(ENV_ROOT)/lib/libhthread_host.a $(LIB_FLAGS) -lpthread -fopenmp -o gemm_host_AB_opt5 -l m
	rm -f ../bin/gemm_host_AB_opt5 && cp gemm_host_AB_opt5 ../bin/

gemm_host_AB_opt6: gemm_host_AB_opt6.c operator.c
	$(MPICC) -O2 gemm_host_AB_opt6.c operator.c -std=c99 -I./include -I$(ENV_ROOT)/include $(ENV_ROOT)/lib/libhthread_host.a $(LIB_FLAGS) -lpthread -fopenmp -o gemm_host_AB_opt6 -l m
	rm -f ../bin/gemm_host_AB_opt6 && cp gemm_host_AB_opt6 ../bin/

gemm_host_ASR: gemm_host_ASR.c operator.c
	$(MPICC) -O2 gemm_host_ASR.c operator.c -std=c99 -I./include -I$(ENV_ROOT)/include $(ENV_ROOT)/lib/libhthread_host.a $(LIB_FLAGS) -lpthread -fopenmp -o gemm_host_ASR -l m
	rm -f ../bin/gemm_host_ASR && cp gemm_host_ASR ../bin/

clean:
	rm -f gemm_host_AB_opt gemm_host_AB_opt2 gemm_host_AB_opt3 gemm_host_AB_opt4 gemm_host_AB_opt5 gemm_host_AB_opt6 gemm_host_ASR
