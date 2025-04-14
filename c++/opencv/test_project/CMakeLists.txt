cmake_minimum_required(VERSION 3.10)

project(TestProject)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_PREFIX_PATH "D:/Users/user/Downloads/opencv/build" CACHE STRING "OpenCV installation prefix")

find_package(OpenCV REQUIRED)
include_directories(${OPenCV_INCLUDE_DIRS})

file(GLOB SOURCES "main.cpp")

add_executable(main ${SOURCES})

target_link_libraries(main ${OpenCV_LIBS})