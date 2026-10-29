export LD_LIBRARY_PATH=/vol8/home/njust_ztc/hnu/yhy/gemm_mpi/bin/libs:$LD_LIBRARY_PATH

Rlevel=2
level=10
matrixA_real=/vol8/home/njust_ztc/hnu/yhy/gemm_mpi/Rdata/Areal${Rlevel}x${level}.bin
matrixA_imag=/vol8/home/njust_ztc/hnu/yhy/gemm_mpi/Rdata/Aimag${Rlevel}x${level}.bin
matrixB_real=/vol8/home/njust_ztc/hnu/yhy/gemm_mpi/Rdata/Breal${level}x${level}.bin
matrixB_imag=/vol8/home/njust_ztc/hnu/yhy/gemm_mpi/Rdata/Bimag${level}x${level}.bin


#matrixA_real=/vol8/home/njust_ztc/hnu/yhy/gemm_mpi/Rdata/A_real${level}.bin
#matrixA_imag=/vol8/home/njust_ztc/hnu/yhy/gemm_mpi/Rdata/A_imag${level}.bin
#matrixB_real=/vol8/home/njust_ztc/hnu/yhy/gemm_mpi/Rdata/B_real${level}.bin
#matrixB_imag=/vol8/home/njust_ztc/hnu/yhy/gemm_mpi/Rdata/B_imag${level}.bin
#level=30
#matrixA_real=/vol8/home/njust_ztc/hnu/yhy/gemm_mpi/scaleData/realH${level}.bin
#matrixA_imag=/vol8/home/njust_ztc/hnu/yhy/gemm_mpi/scaleData/imagH${level}.bin
#matrixB_real=/vol8/home/njust_ztc/hnu/yhy/gemm_mpi/scaleData/realH${level}.bin
#matrixB_imag=/vol8/home/njust_ztc/hnu/yhy/gemm_mpi/scaleData/imagH${level}.bin

#EXE=gemm_host_single
HOST_EXE=gemm_host_ASR

DEVICE_DAT=kernel_AS.dat

cd ../device_code
make
make clean

cd ../host_code
make
make clean

cd ../bin   

yhrun -p mt_work -N 1 --ntasks-per-node 4 --cpus-per-task=1 --exclude=cn[195,290,1985,4478,8202] ./${HOST_EXE} ${matrixA_real} ${matrixA_imag} ${matrixB_real} ${matrixB_imag}  > ../log/gather2.log
#yhrun -p njust -N 700 -n 2800 --nodelist=cn[8195-8216,8219-8257,8260-8272,8275-8279,8285-8292,8294-8295,8304-8366,8369-8378,8380-8407,8417-8431,8437-8439,8446-8449,8451-8472,8474-8480,8482-8492,8494-8498,8500-8504,8506-8513,8515-8522,8527-8530,8532-8535,8542-8546,8548-8576,8578-8592,8595-8598,8600-8614,8616-8622,8627-8643,8697-8745,8747-8762,8764-8767,8769-8791,8798-8805,8809-8821,8833-8855,8857-8860,8862-8865,8872-8894,8898-8900,8905-8907,8909-8914,8916-8995,9128-9131,9133-9135,9140-9144,9146-9161,9165-9171,9178-9182,9186-9189,9195-9204,9207-9215] ./${HOST_EXE} ${matrixA_real} ${matrixA_imag} ${matrixB_real} ${matrixB_imag}  > ../log/gather.log
#yhrun -p njust -N 8 -n 32 ./${HOST_EXE} ${matrixA_real} ${matrixA_imag} ${matrixB_real} ${matrixB_imag}  > ../log/gather.log

rm ${HOST_EXE}
rm ${DEVICE_DAT}