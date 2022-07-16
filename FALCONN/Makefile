INC_DIR=src/include/falconn
TEST_DIR=src/test
BENCH_DIR=src/benchmark
PYTHON_DIR=src/python
GTEST_DIR=external/googletest/googletest
GMOCK_DIR=external/googletest/googlemock
TEST_BIN_DIR=test_bin
PYTHON_WRAPPER_DIR=python_wrapper
PYTHON_WRAPPER_FILE=src/python/wrapper/python_wrapper.cc
PYTHON_PKG_DIR=python_package
DOC_DIR=doc
RM = gio trash -f


ALL_HEADERS = $(INC_DIR)/homogenizer.h $(INC_DIR)/core/nn_query_makkdkdkfcache.h $(INC_DIR)/config.h $(INC_DIR)/core/lsh_table.h $(INC_DIR)/core/cosine_distance.h $(INC_DIR)/core/euclidean_distance.h $(INC_DIR)/core/composite_hash_table.h $(INC_DIR)/core/stl_hash_table.h $(INC_DIR)/core/polytope_hash.h $(INC_DIR)/core/flat_hash_table.h $(INC_DIR)/core/probing_hash_table.h $(INC_DIR)/core/hyperplane_hash.h $(INC_DIR)/core/heap.h $(INC_DIR)/core/prefetchers.h $(INC_DIR)/core/incremental_sorter.h $(INC_DIR)/core/lsh_function_helpers.h $(INC_DIR)/core/hash_table_helpers.h $(INC_DIR)/core/data_storage.h $(INC_DIR)/core/nn_query.h $(INC_DIR)/lsh_nn_table.h $(INC_DIR)/wrapper/cpp_wrapper_impl.h $(INC_DIR)/falconn_global.h $(TEST_DIR)/test_utils.h  $(INC_DIR)/core/data_transformation.h $(INC_DIR)/core/bit_packed_vector.h $(INC_DIR)/core/bit_packed_flat_hash_table.h $(INC_DIR)/core/random_projection_sketches.h $(INC_DIR)/experimental/pipes.h $(INC_DIR)/experimental/code_generation.h

CXX=g++
#/media/mydrive/vtune/compiler/latest/linux/bin/clang++
#CXXFLAGS=-std=c++14 -DNDEBUG -Wall -Wextra -march=native -O3 -I external/eigen -I src/include -I external/simple-serializer -I external/nlohmann -I../Common
CXXFLAGS=-std=c++17 -g -w -march=native -O3 -I external/eigen -I src/include -I external/simple-serializer -I external/nlohmann -I external/wyhash -I../Common
CXXFLAGS_DEBUG=-std=c++17 -g -w -march=native -O0 -I external/eigen -I src/include -I external/simple-serializer -I external/nlohmann -I external/wyhash -I../Common #-fsanitize=address
LDFLAGS=-pthread
# add -DDEBUG to debug
NUMPY_INCLUDE_DIR= $(shell python -c "import numpy; print(numpy.get_include())")
TARGETS =  falconn-axequaltoy jlt-filter debug-axequaltoy


all: $(TARGETS)

clean:
	rm -rf obj
	rm -rf $(TEST_BIN_DIR)
	rm -rf $(DOC_DIR)/html
	rm -rf $(PYTHON_WRAPPER_DIR)
	rm -rf $(PYTHON_PKG_DIR)
	rm -f random_benchmark
	rm -f test-output.txt
	$(RM) $(TARGETS)

falconn-axequaltoy: FalconnMain_axequaltoy.cc 
	-$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

debug-axequaltoy: FalconnMain_axequaltoy.cc 
	-$(CXX) $(CXXFLAGS_DEBUG) $(LDFLAGS) -o $@ $^

jlt-filter: src/filter/filter.cc 
	-$(CXX) $(CXXFLAGS) $(LDFLAGS) -o JLT-filter $^
