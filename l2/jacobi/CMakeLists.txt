cmake_minimum_required(VERSION 3.16)
project(jacobi_fork LANGUAGES C CXX)

# The forked llama.cpp is a subdirectory of this project.
# Build it directly:
#   cd l2/jacobi/llama.cpp && mkdir build && cd build && cmake .. && make -j
#
# Or use build.sh at the project root:
#   bash l2/jacobi/build.sh

set(FORK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")

if(EXISTS "${FORK_DIR}/CMakeLists.txt")
    message(STATUS "Building forked llama.cpp at ${FORK_DIR}")
    add_subdirectory("${FORK_DIR}" "${CMAKE_BINARY_DIR}/llama_fork")
else()
    message(WARNING "Forked llama.cpp not found at ${FORK_DIR}. Run: git clone --depth 1 https://github.com/ggerganov/llama.cpp.git ${FORK_DIR}")
endif()
