# Locates the tensorFlow library and include directories.

include(FindPackageHandleStandardArgs)
unset(TENSORFLOW_FOUND)

find_path(TensorFlow_INCLUDE_DIR
        NAMES
        tensorflow/core
        tensorflow/cc
        third_party
        HINTS
        /usr/local/include/google/tensorflow
        /usr/include/google/tensorflow
	~/tensorflow)

find_library(TensorFlowCC_LIBRARY NAMES libtensorflow_cc.so tenfsorflow_framework.so
        HINTS
        /usr/lib
        /usr/local/lib
	~/tensorflow
	/home/msz/.cache/bazel/_bazel_msz
	/home/msz/.cache/bazel/_bazel_msz/d3830c3f96dd72d2690991a9d766dda3/execroot/org_tensorflow/bazel-out/host/bin/tensorflow
	/home/msz/.cache/bazel/_bazel_msz/d3830c3f96dd72d2690991a9d766dda3/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/)

find_library(TensorFlowFramework_LIBRARY NAMES libtensorflow_framework.so
        HINTS
        /usr/lib
        /usr/local/lib
	~/tensorflow
	/home/msz/.cache/bazel/_bazel_msz
	/home/msz/.cache/bazel/_bazel_msz/d3830c3f96dd72d2690991a9d766dda3/execroot/org_tensorflow/bazel-out/host/bin/tensorflow
	/home/msz/.cache/bazel/_bazel_msz/d3830c3f96dd72d2690991a9d766dda3/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/)

# set TensorFlow_FOUND
find_package_handle_standard_args(TensorFlow DEFAULT_MSG TensorFlow_INCLUDE_DIR TensorFlowCC_LIBRARY TensorFlowFramework_LIBRARY)

# set external variables for usage in CMakeLists.txt
if(TENSORFLOW_FOUND)
    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
    set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR})
endif()

# hide locals from GUI
mark_as_advanced(TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)