#export LD_LIBRARY_PATH=/vol8/home/njust_ztc/hnu/yhy/gemm_mpi/bin/lib:$LD_LIBRARY_PATH

export UCX_MEMTYPE_CACHE=y

num_threads=24


level=16384
matrixA_real=/vol8/home/njust_ztc/hnu/jpp/gemm_mpi/test_data/realH${level}.bin
matrixA_imag=/vol8/home/njust_ztc/hnu/jpp/gemm_mpi/test_data/imagH${level}.bin
matrixB_real=/vol8/home/njust_ztc/hnu/jpp/gemm_mpi/test_data/realH${level}.bin
matrixB_imag=/vol8/home/njust_ztc/hnu/jpp/gemm_mpi/test_data/imagH${level}.bin

# HOST_EXE=read_data
# HOST_EXE=gemm_host_ASR
HOST_EXE=gemm_host_AB


DEVICE_DAT=kernel_AS.dat


cd ../device_code
make
make clean

cd ../host_code
make
make clean

cd ../bin
yhrun -p mt_work -N 70 --ntasks-per-node 4 --cpus-per-task=1 --mpi=pmix --exclude=cn[195,290,1985,4478,8202] ./${HOST_EXE} ${matrixA_real} ${matrixA_imag} ${matrixB_real} ${matrixB_imag} ${num_threads}  >> ../log/gather.log
export UCX_TLS=^cma
# yhrun -p mt_work -N 512 -n 4 --mpi=pmix --exclude=cn[195,290,411,1985,4478,8202] ./${HOST_EXE} ${matrixA_real} ${matrixA_imag} ${matrixB_real} ${matrixB_imag} ${num_threads}  >> ../log/gather.log
# yhrun -p mt_work -N 1 -n 1 --mpi=pmix --exclude=cn[195,290,411,1985,4478,8202] ./${HOST_EXE} ${matrixA} ${matrixB} ${num_threads}  >> ../log/gather.log
# yhrun -p njust -N 700 -n 2800 --mpi=pmix --nodelist=cn[8195-8216,8219-8257,8260-8272,8275-8279,8285-8292,8294-8295,8304-8366,8369-8378,8380-8407,8417-8431,8437-8439,8446-8449,8451-8472,8474-8480,8482-8492,8494-8498,8500-8504,8506-8513,8515-8522,8527-8530,8532-8535,8542-8546,8548-8576,8578-8592,8595-8598,8600-8614,8616-8622,8627-8643,8697-8745,8747-8762,8764-8767,8769-8791,8798-8805,8809-8821,8833-8855,8857-8860,8862-8865,8872-8894,8898-8900,8905-8907,8909-8914,8916-8995,9128-9131,9133-9135,9140-9144,9146-9161,9165-9171,9178-9182,9186-9189,9195-9204,9207-9215] ./${HOST_EXE} ${matrixA_real} ${matrixA_imag} ${matrixB_real} ${matrixB_imag}  > ../log/gather.log
#yhrun -p njust -N 8 -n 32 ./${HOST_EXE} ${matrixA_real} ${matrixA_imag} ${matrixB_real} ${matrixB_imag}  > ../log/gather.log

rm ${HOST_EXE}
rm ${DEVICE_DAT}