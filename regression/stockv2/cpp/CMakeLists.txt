project(examples)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package (gflags REQUIRED)

include_directories (${gflags_INCLUDE_DIR})
include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/src/common
${CMAKE_CURRENT_SOURCE_DIR}/src/image
${CMAKE_CURRENT_SOURCE_DIR}/src/free_type
)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/common SRCS_common)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/image SRCS_image)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/free_type SRCS_free_type)
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src/show SRCS_show)

add_executable(show ${SRCS_common} ${SRCS_image} ${SRCS_show})
target_link_libraries(show 
${OpenCV_LIBS}
gflags
pthread
libcnpy.a
z)
