GCCROOT=/vol8/appsoftware/mt3000_programming_env/dsp_compiler
ENV_ROOT=/vol8/appsoftware/mt3000_programming_env/hthreads
CC=MT-3000-gcc
AR=MT-3000-ar
LD=MT-3000-ld

export LD_LIBRARY_PATH=/vol8/appsoftware/mt3000_programming_env/third-party-lib:$LD_LIBRARY_PATH
CFLAGS=-c -O2 -g -gdwarf-2 -fenable-m3000 -ffunction-sections -flax-vector-conversions -I./ -I$(ENV_ROOT)/include -I$(GCCROOT)/include/
LDFLAGS= -L$(ENV_ROOT)/lib --gc-sections -Tdsp.lds

# DEV=kernel_B
DEV=kernel_AS
# DEV=kernel_pipeline
# DEV=kernel_pipe_real
# DEV=kernel_G

SRC=${DEV}.c
OBJ=${DEV}.o
EXE=${DEV}.out
DAT=${DEV}.dat

ALL: $(EXE)
	$(GCCROOT)/bin/MT-3000-makedat -J $(EXE)
	cp $(DAT) ../bin

$(OBJ): $(SRC)
	$(GCCROOT)/bin/$(CC) -Wall $(CFLAGS) $(SRC) -o $(OBJ)

$(EXE): $(OBJ)
	$(GCCROOT)/bin/$(LD) $(LDFLAGS) $(OBJ) $(ENV_ROOT)/lib/libhthread_device.a $(GCCROOT)/lib/vlib3000.a $(GCCROOT)/lib/slib3000.a -o $(EXE)
#	$(GCCROOT)/bin/$(LD) $(OBJ) $(ENV_ROOT)/lib/libhthread_device.a $(GCCROOT)/lib/vlib3000.a $(GCCROOT)/lib/slib3000.a -o $(EXE)

clean:
	rm -f $(EXE) $(OBJ) $(DAT)
