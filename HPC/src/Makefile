# Workstation build of the bare-plate programs from src/ (a copy of bare_plate_tutorial/src).
# Build-time dependencies are taken from the original package (PKG): bundled MFEM 4.9.1, hypre, otinum, MKL headers,
# the pre-compiled Enzyme kernel object and the run-time libraries in PKG/lib (reached here through the lib link).
#   make cpu   -> $(BIN)/thermal_enzyme_src   (CPU, same code as the shipped $(BIN)/thermal_enzyme)
#   make gpu   -> $(BIN)/thermal_enzyme_gpu   (Goldak laser source + its first-order AD on the GPU, src/gpu/; GPU_SOURCE=0 at run time = CPU path)
#                 derivatives on the GPU by forward-mode dual numbers (nvcc); also the Newton assembly (K refill, boundary
#                 terms, capacitance; src/gpu/gpu_newton.cu, GPU_NEWTON=0 at run time = CPU loops)
#   make mech  -> $(BIN)/mech_main_aa_src     (mechanical solver from src/mechanical, CPU)
#   make mech_gpu -> $(BIN)/mech_main_aa_gpu  (same source, nvcc -DUSE_GPU_MECH: element/point loops on the GPU, PARDISO on the CPU;
#                 -fmad=false keeps the per-point arithmetic identical to the CPU; GPU_MECH=0 at run time = CPU loops)
#   make gpu_enzyme -> $(BIN)/thermal_enzyme_gpu_enzyme   same, derivatives on the GPU by Enzyme (clang 21 -x cuda + ClangEnzyme-21.so,
#                 toolchain in ../../../toolchain: LLVM 21.1.8 release + Enzyme v0.0.303 built against it)
# Paths contain a space ("Daniel Morales"): every path variable below is used quoted.
ROOT      := $(abspath .)
PKG       := $(abspath ../../bare_plate_tutorial)
CXX       ?= g++
MFEM_INC  := -I"$(PKG)/MFEM/include"
MFEM_LIB  := -L"$(PKG)/MFEM/lib" -lmfem -lrt
HYPRE_INC := -DUSE_HYPRE_AMG -I"$(PKG)/third_party/hypre/include"
HYPRE_LIB := "$(PKG)/third_party/hypre/lib/libHYPRE.a"
OTI_INC   := -I"$(PKG)/third_party/otinum"
MKL_INC   := -I"$(PKG)/third_party/mkl_include"
MKL_LIB   := "$(PKG)/lib/libmkl_rt.so.2" -lpthread -lm -ldl
RTLIBS    := "$(PKG)/lib/libstdc++.so.6" "$(PKG)/lib/libgomp.so.1"
RPATH     := -Wl,-rpath,'$$ORIGIN/../lib' -Wl,--disable-new-dtags
CXXFLAGS  := -O3 -std=c++17 -fopenmp
LDFLAGS   := -no-pie                  # Ubuntu g++ defaults to PIE; the bundled libmfem.a is non-PIC (built on the cluster)
TH        := src/thermal
# Build parameters (defaults = the IN625 programs of this folder). Another material / place:
#   make gpu_enzyme MATERIAL=IN718 OBJ=<abs build dir> BIN=<abs bin dir>   (used by MATH_DT/single_tracks/Makefile)
OBJ       ?= build
BIN       ?= bin
MATERIAL  ?=
MATFLAG   := $(if $(MATERIAL),-DMATERIAL_$(MATERIAL),)
SHAPES    ?= 1x1 1x2 1x3 1x4 2x1 2x2 2x3 2x4 3x1 3x2 3x3 3x4 4x1 4x2 4x3 4x4 5x1 5x2 5x3 5x4 10x1 10x2
SHAPE_OBJS:= $(patsubst %,$(OBJ)/eval_%.o,$(SHAPES))
AO_OBJS   := $(OBJ)/eval_registry.o $(SHAPE_OBJS)
DEPS      := $(wildcard $(TH)/*.hpp $(TH)/*.h) src/gpu/solver_switch.h src/gpu/cudss_backend.h
# Enzyme kernel object: built here from src/thermal/fb_enzyme_kernels.cpp (+ fb_enzyme_fns.hpp, shared with the GPU) with the
# local clang 21 + Enzyme v0.0.303 toolchain. ENZ_OBJ_SHIPPED is the cluster-built object of the package (for comparison).
ENZ_OBJ_SHIPPED := "$(PKG)/third_party/enzyme/fb_enzyme_kernels.o"
ENZ_OBJ   := $(OBJ)/fb_enzyme_kernels.o
TH_FLAGS  := $(MATFLAG) -DUSE_ENZYME_AD -DOTI_ORDER_VAL=1 $(MFEM_INC) $(OTI_INC) $(MKL_INC) $(HYPRE_INC) -I$(TH)
TH_LIBS   := $(MFEM_LIB) $(HYPRE_LIB) $(MKL_LIB) $(RTLIBS) $(RPATH)
CUDA      ?= /usr/local/cuda-11.8
NVCC      := $(CUDA)/bin/nvcc
# GPU compute capabilities to build for (fat binaries): 75 = Quadro RTX 4000 (workstation); cluster: GPU_ARCHS="70 75 80"
# (V100 = 7.0, A100 = 8.0)
GPU_ARCHS ?= 75
NV_ARCH   := $(foreach a,$(GPU_ARCHS),-gencode arch=compute_$(a),code=sm_$(a))
NVFLAGS   := -O3 -std=c++17 $(NV_ARCH) -ccbin $(CXX)
# GPU programs: ONE CUDA runtime, 12.2 (from ../../../toolchain/cuda12, needed by cuDSS); the kernels compiled by nvcc 11.8 /
# clang (sm_75 code) run on it. The CPU/GPU solver switch (src/gpu/solver_switch.h, cudss_backend.*) is in every GPU program.
CUDA12    := $(abspath ../../../toolchain/cuda12/nvidia)
CUDA12_INC:= -I"$(CUDA12)/cuda_runtime/include" -I"$(CUDA12)/cu12/include" -idirafter $(CUDA)/include
CUDA_LIB  := -L"$(CUDA12)/cu12/lib" -l:libcudss.so.1 -L"$(CUDA12)/cuda_runtime/lib" -l:libcudart.so.12 \
             -Wl,-rpath,"$(CUDA12)/cu12/lib:$(CUDA12)/cuda_runtime/lib:$(CUDA12)/cublas/lib:$(CUDA12)/cuda_nvrtc/lib"
SWITCH    := -DUSE_SOLVER_SWITCH
TOOLCHAIN := $(abspath ../../../toolchain)
CLANGCU   := $(TOOLCHAIN)/llvm-21.1.8/bin/clang++
ENZ_PLUG  := $(TOOLCHAIN)/enzyme-build/Enzyme/ClangEnzyme-21.so
CLANGCU_FLAGS := -O3 -std=c++17 -x cuda --cuda-path=$(CUDA) $(foreach a,$(GPU_ARCHS),--cuda-gpu-arch=sm_$(a)) -Wno-unknown-cuda-version -fplugin="$(ENZ_PLUG)" -DGPU_SRC_ENZYME

.PHONY: cpu gpu gpu_enzyme mech mech_gpu clean
mech: $(BIN)/mech_main_aa_src
mech_gpu: $(BIN)/mech_main_aa_gpu
cpu: $(BIN)/thermal_enzyme_src
gpu: $(BIN)/thermal_enzyme_gpu
gpu_enzyme: $(BIN)/thermal_enzyme_gpu_enzyme

$(BIN)/thermal_enzyme_src: $(TH)/full_bridge_activation_tier1_oti.cpp $(DEPS) $(AO_OBJS) $(ENZ_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(TH_FLAGS) $< $(AO_OBJS) $(ENZ_OBJ) -o $@ $(TH_LIBS)

$(BIN)/thermal_enzyme_gpu: $(TH)/full_bridge_activation_tier1_oti.cpp $(DEPS) src/gpu/gpu_source.h src/gpu/gpu_newton.h $(AO_OBJS) $(OBJ)/gpu_source.o $(OBJ)/gpu_newton.o $(OBJ)/gpu_enzyme_sens.o $(OBJ)/cudss_backend.o $(OBJ)/gpu_krylov.o $(ENZ_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(TH_FLAGS) -DUSE_GPU_SOURCE -DUSE_GPU_NEWTON -DUSE_GPU_SENS $(SWITCH) -Isrc/gpu $< $(AO_OBJS) $(OBJ)/gpu_source.o $(OBJ)/gpu_newton.o $(OBJ)/gpu_enzyme_sens.o $(OBJ)/cudss_backend.o $(OBJ)/gpu_krylov.o $(ENZ_OBJ) -o $@ $(TH_LIBS) $(CUDA_LIB)

# Jacobi-BiCGSTAB for the sensitivity system (SENS_SOLVER=krylov), GPU version of src/thermal/krylov_sens.hpp
$(OBJ)/gpu_krylov.o: src/gpu/gpu_krylov.cu src/gpu/gpu_krylov.h | $(OBJ)
	$(NVCC) $(NVFLAGS) -c $< -o $@

# Newton assembly kernels: -fmad=false keeps the per-element arithmetic identical to the CPU loops
$(OBJ)/gpu_newton.o: src/gpu/gpu_newton.cu src/gpu/gpu_newton.h | $(OBJ)
	$(NVCC) $(NVFLAGS) -fmad=false $(MATFLAG) -I$(TH) -c $< -o $@

$(OBJ)/gpu_source.o: src/gpu/gpu_source.cu src/gpu/gpu_source.h src/gpu/srck_cell.hpp $(TH)/src_kernel.hpp | $(OBJ)
	$(NVCC) $(NVFLAGS) -I$(TH) -c $< -o $@

$(BIN)/thermal_enzyme_gpu_enzyme: $(TH)/full_bridge_activation_tier1_oti.cpp $(DEPS) src/gpu/gpu_source.h src/gpu/gpu_newton.h $(AO_OBJS) $(OBJ)/gpu_source_enzyme.o $(OBJ)/gpu_newton.o $(OBJ)/gpu_enzyme_sens.o $(OBJ)/cudss_backend.o $(OBJ)/gpu_krylov.o $(ENZ_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(TH_FLAGS) -DUSE_GPU_SOURCE -DUSE_GPU_NEWTON -DUSE_GPU_SENS $(SWITCH) -Isrc/gpu $< $(AO_OBJS) $(OBJ)/gpu_source_enzyme.o $(OBJ)/gpu_newton.o $(OBJ)/gpu_enzyme_sens.o $(OBJ)/cudss_backend.o $(OBJ)/gpu_krylov.o $(ENZ_OBJ) -o $@ $(TH_LIBS) $(CUDA_LIB)

# sensitivity blocks (a)(b)(c): Enzyme in device code; -ffp-contract=off = no fused multiply-add (as the CPU object)
$(OBJ)/gpu_enzyme_sens.o: src/gpu/gpu_enzyme_sens.cu src/gpu/gpu_enzyme_sens.h $(TH)/fb_enzyme_fns.hpp | $(OBJ)
	env -u CPATH "$(CLANGCU)" $(CLANGCU_FLAGS) -ffp-contract=off $(MATFLAG) -I$(TH) -Isrc/gpu -c $< -o $@

# CPATH is cleared: the oneAPI headers it adds must not enter the clang CUDA compilation
$(OBJ)/gpu_source_enzyme.o: src/gpu/gpu_source.cu src/gpu/gpu_source.h src/gpu/srck_cell.hpp $(TH)/src_kernel.hpp | $(OBJ)
	env -u CPATH "$(CLANGCU)" $(CLANGCU_FLAGS) -I$(TH) -c $< -o $@

$(BIN)/mech_main_aa_src: src/mechanical/mech_main_aa.cpp $(wildcard src/mechanical/*.hpp)
	$(CXX) $(CXXFLAGS) $(MKL_INC) $< -o $@ $(MKL_LIB) $(RPATH)

$(BIN)/mech_main_aa_gpu: $(OBJ)/mech_main_aa_gpu.o $(OBJ)/cudss_backend.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(MKL_LIB) $(CUDA_LIB) $(RPATH)

$(OBJ)/mech_main_aa_gpu.o: src/mechanical/mech_main_aa.cpp $(wildcard src/mechanical/*.hpp) src/gpu/mech_gpu.cuh src/gpu/solver_switch.h src/gpu/cudss_backend.h | $(OBJ)
	$(NVCC) -x cu -O3 -std=c++17 $(NV_ARCH) -fmad=false -ccbin $(CXX) -Xcompiler -fopenmp -DUSE_GPU_MECH $(SWITCH) -Isrc/gpu $(MKL_INC) -c $< -o $@

$(OBJ)/fb_enzyme_kernels.o: $(TH)/fb_enzyme_kernels.cpp $(TH)/fb_enzyme_kernels.h $(TH)/fb_enzyme_fns.hpp $(TH)/src_kernel.hpp | $(OBJ)
	env -u CPATH "$(CLANGCU)" -O2 -std=c++17 -fPIC -fno-exceptions -fplugin="$(ENZ_PLUG)" -DMFEM_USE_ENZYME $(MATFLAG) $(MFEM_INC) -I"$(PKG)/MFEM/include/mfem" -I$(TH) -c $< -o $@

# same CPU program linked with the shipped Enzyme object (reference for the locally built one)
$(BIN)/thermal_enzyme_src_shippedenz: $(TH)/full_bridge_activation_tier1_oti.cpp $(DEPS) $(AO_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(TH_FLAGS) $< $(AO_OBJS) $(ENZ_OBJ_SHIPPED) -o $@ $(TH_LIBS)

# cuDSS backend of the solver switch (host code only: g++ against the CUDA 12 runtime and cuDSS headers)
$(OBJ)/cudss_backend.o: src/gpu/cudss_backend.cpp src/gpu/cudss_backend.h | $(OBJ)
	$(CXX) -O2 -std=c++17 $(CUDA12_INC) -c $< -o $@

$(OBJ) $(BIN):
	mkdir -p $@
$(OBJ)/shapes.inc: Makefile | $(OBJ)
	@rm -f $@; for s in $(SHAPES); do m=$${s%x*}; n=$${s#*x}; echo "SHAPE($$m,$$n)" >> $@; done
$(OBJ)/eval_registry.o: $(TH)/eval_registry.cpp $(OBJ)/shapes.inc $(TH)/anyorder_eval.hpp | $(OBJ)
	$(CXX) $(CXXFLAGS) $(OTI_INC) -I$(TH) -I$(OBJ) -c $< -o $@
$(OBJ)/eval_%.o: $(TH)/eval_shape.cpp $(TH)/anyorder_eval.hpp $(TH)/oti_query_type.hpp | $(OBJ)
	$(CXX) $(CXXFLAGS) $(OTI_INC) -I$(TH) -DJET_M=$(word 1,$(subst x, ,$*)) -DJET_N=$(word 2,$(subst x, ,$*)) -c $< -o $@

clean:
	rm -rf $(OBJ) $(BIN)/thermal_enzyme_src $(BIN)/thermal_enzyme_gpu $(BIN)/thermal_enzyme_gpu_enzyme $(BIN)/mech_main_aa_src $(BIN)/mech_main_aa_gpu
