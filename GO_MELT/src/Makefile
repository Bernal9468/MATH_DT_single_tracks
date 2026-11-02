# GO_MELT build (serial MFEM for mesh I/O only; OpenMP; no OTI, no MKL, no hypre)
MFEM_DIR = /work/gyj329/MFEM/implementations/bridge_few_layers/bridge_4layer_kokkos/My_kokkos/mfem_openmp/src
include $(MFEM_DIR)/config/config.mk
BIN = ../bin
all: $(BIN)/gm_single
$(BIN)/gm_single: gm_single.cpp gm_core.hpp gm_sens.hpp material_in718.hpp fb_enzyme_kernels.o
	mkdir -p $(BIN)
	$(MFEM_CXX) $(MFEM_FLAGS) -std=c++17 -O3 -fopenmp -I. $< fb_enzyme_kernels.o -o $@ $(MFEM_LIBS) -L/home/gyj329/anaconda3/lib -Wl,-rpath,/home/gyj329/anaconda3/lib -fopenmp
clean:
	rm -f $(BIN)/gm_single
all: $(BIN)/gm_levels
$(BIN)/gm_levels: gm_levels.cpp gm_core.hpp material_in718.hpp
	mkdir -p $(BIN)
	$(MFEM_CXX) $(MFEM_FLAGS) -std=c++17 -O3 -fopenmp -I. $< -o $@ $(MFEM_LIBS) -L/home/gyj329/anaconda3/lib -Wl,-rpath,/home/gyj329/anaconda3/lib -fopenmp
all: $(BIN)/gm_moving
$(BIN)/gm_moving: gm_moving.cpp gm_core.hpp gm_sens.hpp material_in718.hpp fb_enzyme_kernels.o
	mkdir -p $(BIN)
	$(MFEM_CXX) $(MFEM_FLAGS) -std=c++17 -O3 -fopenmp -I. $< fb_enzyme_kernels.o -o $@ $(MFEM_LIBS) -L/home/gyj329/anaconda3/lib -Wl,-rpath,/home/gyj329/anaconda3/lib -fopenmp

ENZYME_PLUGIN = /work/gyj329/MFEM/implementations/bridge_few_layers/bridge_case_enzyme/third_party/enzyme_build/Enzyme/ClangEnzyme-21.so
fb_enzyme_kernels.o: fb_enzyme_kernels.cpp fb_enzyme_kernels.h fb_enzyme_fns.hpp src_kernel.hpp
	/usr/lib64/llvm21/bin/clang++ -O2 -std=c++17 -fPIC -fno-exceptions -fplugin=$(ENZYME_PLUGIN) -DMFEM_USE_ENZYME -DMATERIAL_IN718 -I$(MFEM_DIR) -I. -c $< -o $@ 2>/dev/null
