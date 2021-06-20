# Code for Fast k-Nearest Neighbour Search via Prioritized DCI
# 
# This code implements the method described in the Prioritized DCI paper, which 
# can be found at https://arxiv.org/abs/1703.00440
# 
# Copyright (C) 2017    Ke Li
# 
# 
# This file is part of the Dynamic Continuous Indexing reference implementation.
# 
# The Dynamic Continuous Indexing reference implementation is free software: 
# you can redistribute it and/or modify it under the terms of the GNU Affero 
# General Public License as published by the Free Software Foundation, either 
# version 3 of the License, or (at your option) any later version.
# 
# The Dynamic Continuous Indexing reference implementation is distributed in 
# the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the 
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
# See the GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with the Dynamic Continuous Indexing reference implementation.  If 
# not, see <http://www.gnu.org/licenses/>.

# Build Instructions:
# 1. Set the BLAS variable to "netlib", "atlas", "openblas" or "mkl"
# 2. Set NETLIB_DIR, ATLAS_DIR, OPENBLAS_DIR or MKL_DIR to the directory for your BLAS implementation
# 3. Set PYTHON_DIR and NUMPY_DIR to the directories for Python and Numpy (Note that the code only supports Python 2.7)
# 4. Run make

CC=gcc
BLAS=netlib
NETLIB_DIR=/usr/lib/libblas
ATLAS_DIR=/usr/lib/atlas-base
OPENBLAS_DIR=/usr/lib/openblas-base
MKL_DIR=/opt/intel/mkl
PYTHON_DIR=/usr/include/python2.7
NUMPY_DIR=/usr/local/lib/python2.7/dist-packages/numpy/core/include
# NUMPY_DIR=/cfarhomes/mmeshry/anaconda2/lib/python2.7/site-packages/numpy/core/include

SRC_DIR=src
INCL_DIR=include
BUILD_DIR=build

GEN_FLAGS=-Wall -O3 -std=gnu99 -m64 -fopenmp -flto
LIB_FLAGS=-lm

OBJ_FILES=dci.o util.o
OBJ_PATHS=$(addprefix $(BUILD_DIR)/,$(OBJ_FILES))
INCL_FILES=dci.h util.h
INCL_PATHS=$(addprefix $(INCL_DIR)/,$(INCL_FILES))
ALL_INCL_DIRS=$(PYTHON_DIR) $(NUMPY_DIR) $(INCL_DIR)

ifeq ($(BLAS), netlib)
    LIB_FLAGS += -L$(NETLIB_DIR) -Wl,-rpath $(NETLIB_DIR) -lblas
endif
ifeq ($(BLAS), atlas)
    LIB_FLAGS += -L$(ATLAS_DIR) -Wl,-rpath $(ATLAS_DIR) -ltatlas
endif
ifeq ($(BLAS), openblas)
    LIB_FLAGS += -L$(OPENBLAS_DIR) -Wl,-rpath $(OPENBLAS_DIR) -lopenblas
endif
ifeq ($(BLAS), mkl)
    ALL_INCL_DIRS += $(MKL_DIR)/include
    GEN_FLAGS += -DUSE_MKL
    LIB_FLAGS += -L$(MKL_DIR)/lib/intel64 -Wl,-rpath $(MKL_DIR)/lib/intel64 -lmkl_rt -lpthread -ldl
endif

ALL_INCL_FLAGS=$(addprefix -I,$(ALL_INCL_DIRS))

.PHONY: all

all: $(BUILD_DIR)/_dci.so
	ln -sf $(BUILD_DIR)/_dci.so .
	ln -sf $(SRC_DIR)/dci.py .

$(BUILD_DIR)/%.so: $(BUILD_DIR)/py%.o $(OBJ_PATHS)
	$(CC) -shared -o $@ $^ $(GEN_FLAGS) $(LIB_FLAGS)

$(BUILD_DIR)/%.o : $(SRC_DIR)/%.c $(INCL_PATHS)
	mkdir -p $(BUILD_DIR)
	$(CC) -c -o $@ $< -fPIC $(GEN_FLAGS) $(ALL_INCL_FLAGS)

.PHONY: clean

clean:
	rm -rf $(BUILD_DIR) *.pyc dci.py _dci.so
