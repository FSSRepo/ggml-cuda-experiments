cmake_minimum_required(VERSION 3.0)
project(GGML-Cuda)

set(SOURCE_FILES
    ggml_conv2d.c
)

set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
)

include_directories(${INCLUDE_DIRS})

add_subdirectory(ggml)

# Create an executable target
add_executable(ggml-conv2d ${SOURCE_FILES})

target_link_libraries(ggml-conv2d PUBLIC ggml)