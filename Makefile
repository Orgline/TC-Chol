CXX = nvcc
# CULDFLAGS=-L/usr/local/cuda/lib
CFLAGS = -O2 -arch sm_80 -I $(CUDA_PATH)/include\cholesky.h
LFLAGS = -L $(CUDA_PATH)/lib64 -lcublas -lcublasLt -lcusolver -lcurand -lcudart -lcuda
CC = gcc

all: util dtcpotrf dtcpotrf_opt tc_rtrsm tc_btrsm tc_syrk tc_potrf

util: util.cu
	nvcc  $(CFLAGS) $(LFLAGS) -c util.cu
dtcpotrf: dtcpotrf.cu
	nvcc  $(CFLAGS) -c dtcpotrf.cu
	nvcc  $(LFLAGS) dtcpotrf.o util.o -o dtcpotrf
dtcpotrf_opt: dtcpotrf_opt.cu
	nvcc  $(CFLAGS) -c dtcpotrf_opt.cu
	nvcc  $(LFLAGS) dtcpotrf_opt.o util.o -o dtcpotrf_opt
tc_rtrsm: tc_rtrsm.cu
	nvcc  $(CFLAGS) -c tc_rtrsm.cu
	nvcc  $(LFLAGS) tc_rtrsm.o util.o -o tc_rtrsm
tc_btrsm: tc_btrsm.cu
	nvcc  $(CFLAGS) -c tc_btrsm.cu
	nvcc  $(LFLAGS) tc_btrsm.o util.o -o tc_btrsm
tc_syrk: tc_syrk.cu
	nvcc  $(CFLAGS) -c tc_syrk.cu
	nvcc  $(LFLAGS) tc_syrk.o util.o -o tc_syrk
tc_potrf: tc_potrf.cu
	nvcc  $(CFLAGS) -c tc_potrf.cu tc_syrk.cu tc_rtrsm.cu util.cu
	nvcc  $(LFLAGS) tc_potrf.o tc_rtrsm.o tc_syrk.o util.o -o tc_potrf
rpotrf: rpotrf.cu
	nvcc  $(CFLAGS) -c rpotrf.cu rsyrk.cu rtrsm.cu util.cu
	nvcc  $(LFLAGS) rpotrf.o rtrsm.o rsyrk.o util.o -o rpotrf

clean: 
	rm util.o dtcpotrf_opt.o dtcpotrf_opt dtcpotrf.o dtcpotrf tc_rtrsm.o tc_rtrsm tc_btrsm.o tc_btrsm tc_syrk.o tc_syrk tc_potrf.o tc_potrf
