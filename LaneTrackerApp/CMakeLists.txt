# This compiles and links Executable called "TUeLaneTracker"
cmake_minimum_required (VERSION 2.8.11)

SET(CMAKE_MODULE_PATH 	 ${CMAKE_SOURCE_DIR}/cmake/Modules ${CMAKE_MODULE_PATH} )

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost 1.62.0 REQUIRED COMPONENTS program_options )
find_package(TensorFlow REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -O0 -std=c++11 -I/home/msz/.cache/bazel/_bazel_msz/d3830c3f96dd72d2690991a9d766dda3/external/com_google_absl -I/home/msz/tensorflow/tensorflow -I/home/msz/tensorflow/bazel-tensorflow/external/eigen_archive -I/home/msz/tensorflow/bazel-tensorflow/external/protobuf_archive/src -I/home/msz/tensorflow/bazel-genfiles -I/home/msz/tensorflow -ltensorflow_cc -ltensorflow_framework -L /home/msz/.cache/bazel/_bazel_msz/d3830c3f96dd72d2690991a9d766dda3/execroot/org_tensorflow/bazel-out/host/bin/tensorflow/ -L /home/msz/.cache/bazel/_bazel_msz/d3830c3f96dd72d2690991a9d766dda3/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/")

SET(LDFLAGS "${LDFLAGS} -ltensorflow_cc -ltensorflow_framework -L /home/msz/.cache/bazel/_bazel_msz/d3830c3f96dd72d2690991a9d766dda3/execroot/org_tensorflow/bazel-out/host/bin/tensorflow/ -L /home/msz/.cache/bazel/_bazel_msz/d3830c3f96dd72d2690991a9d766dda3/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/")
#add_definitions (-D_GLIBCXX_USE_CXX11_ABI=0)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.c )
add_executable (TUeLaneTracker ${SOURCES})
TARGET_LINK_LIBRARIES(TUeLaneTracker SHARED tensorflow_framework )
TARGET_LINK_LIBRARIES(TUeLaneTracker SHARED tensorflow_cc )

# Link the executable to the TUeLDT library. Since the TUeLDT library has
# public include directories we will use those includes when buiding TUeLaneTracker

target_include_directories (TUeLaneTracker PUBLIC	${Boost_INCLUDE_DIRS}
							${TensorFlow_INCLUDE_DIRS})

target_link_libraries 	   (TUeLaneTracker PUBLIC 	TUeLDT 	
							${Boost_LIBRARIES} 
							${OpenCV_LIBS} 
							${CMAKE_THREAD_LIBS_INIT}
							${TensorFlow_LIBRARIES})


install(TARGETS  TUeLaneTracker DESTINATION install)
