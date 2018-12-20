cmake_minimum_required(VERSION 3.1)
project(cpptensorflow)
set(CMAKE_CXX_STANDARD 11)

link_directories(/home/jwei/tensorflow_c/codes)

include_directories(
	/home/jwei/envs/python3.5/lib/python3.5/site-packages/tensorflow/include
	/home/jwei/tensorflow
	/home/jwei/tensorflow/bazel-genfiles
	/home/jwei/tensorflow/bazel-bin/tensorflow
	/home/jwei/tensorflow_c/eigen3
)

set(OpenCV_DIR /usr/share/OpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

link_libraries("/home/jwei/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so")
link_libraries("/home/jwei/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so")

add_executable(${PROJECT_NAME} main.cpp ann_model_loader.h model_loader_base.h ann_model_loader.cpp)
#add_executable(${PROJECT_NAME} main2.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
#target_link_libraries(cpptensorflow tensorflow_cc tensorflow_framework)
