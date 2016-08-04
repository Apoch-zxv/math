project(math)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(math ${SOURCE_FILES})
ADD_DEFINITIONS("-O3")
target_link_libraries(math matio armadillo lapack blas)