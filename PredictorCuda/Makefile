################################################################################
#
# Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################

# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

# These flags will override any settings
ifeq ($(i386),1)
	OS_SIZE = 32
	OS_ARCH = i686
endif

ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif

# Flags to detect either a Linux system (linux) or Mac OSX (darwin)
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# Location of the CUDA Toolkit binaries and libraries
CUDA_PATH       ?= /usr/local/cuda-12.8
CUDA_INC_PATH   ?= $(CUDA_PATH)/include 
CUDA_INC_PATH_COMMON ?= ../include 

CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
ifneq ($(DARWIN),)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
  endif
endif

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             ?= g++

# Extra user flags
EXTRA_NVCCFLAGS ?= -Xcompiler -fopenmp
EXTRA_LDFLAGS   ?= -fopenmp

# CUDA code generation flags
GENCODE_SM60    := -gencode arch=compute_86,code=sm_86
GENCODE_FLAGS   := $(GENCODE_SM60)

# OS-specific build flags
ifneq ($(DARWIN),) 
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH) 
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

# Debug build flags
ifeq ($(dbg),1)
      CCFLAGS   += -g
      NVCCFLAGS += -g -G --ptxas-options -v
      TARGET := debug
else
      TARGET := release
endif

# Parametros
DATOS ?= ./datosEx.json
CAMPOS ?= tmax tmed tmin prec velmedia
NHILOS ?= 4

# Target rules
all: build

build: PredictorCuda

Arima.o: Arima.cu Arima.h
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) -I$(CUDA_INC_PATH) -I$(CUDA_INC_PATH_COMMON) $(GENCODE_FLAGS) -o $@ -c $<

cJSON.o: cJSON.c cJSON.h
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) -I$(CUDA_INC_PATH) -I$(CUDA_INC_PATH_COMMON) $(GENCODE_FLAGS) -o $@ -c $<

PredictorCuda.o: JSONParserOMP.cu Arima.h cJSON.h
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) -I$(CUDA_INC_PATH) -I$(CUDA_INC_PATH_COMMON) $(GENCODE_FLAGS) -o $@ -c $<

PredictorCuda: Arima.o PredictorCuda.o cJSON.o
	$(GCC) $(CCFLAGS) -o $@ $^ $(LDFLAGS) $(EXTRA_LDFLAGS)
	mkdir -p ../../bin/$(OSLOWER)/$(TARGET)
	cp $@ ../../bin/$(OSLOWER)/$(TARGET)

run: build
	@echo "----------------------------------------"
	@echo "Input: $(DATOS)"
	@echo "Campos a analizar: $(CAMPOS)"
	@echo "Numero de Hilos por proceso: $(NHILOS)"
	@echo "----------------------------------------"
	./PredictorCuda $(DATOS) $(NHILOS) $(CAMPOS)

clean:
	@echo "Limpiando archivos generados..."
	rm -f PredictorCuda *.o
	@echo "Archivos generados eliminados."

help:
	@echo "Opciones Disponibles:"
	@echo "  all       - Build (default)"
	@echo "  run       - Ejecutar (default)"
	@echo "  clean     - Limpieza"
	@echo ""
	@echo "Parametros de Ejecución:"
	@echo "  DATOS    - Input (default: datos.json)"
	@echo "  CAMPOS   - Campos analizados (default: tmax tmed prec velmedia)"
	@echo "  NHILOS   - Numero de Hilos por proceso (default: 4)"
	@echo "  Cambiable usando export VARIABLE=NuevoValor"