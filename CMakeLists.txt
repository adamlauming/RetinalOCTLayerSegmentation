cmake_minimum_required(VERSION 3.14)
project(layerSegmentation)

set(CMAKE_CXX_STANDARD 14)

add_executable(layerSegmentation main.cpp SGSmooth.cpp SGSmooth.hpp)

find_package(OpenCV REQUIRED)
target_link_libraries(layerSegmentation ${OpenCV_LIBS})