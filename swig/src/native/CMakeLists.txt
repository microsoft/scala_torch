################################################################
# A minimal CMake file that is compatible with sbt-jni         #
#                                                              #
# All settings required by sbt-jni have been marked so, please #
# add/modify/remove settings to build your specific library.   #
################################################################

cmake_minimum_required(VERSION 3.15.0)
if (WIN32)
# Explained here:
# https://cmake.org/cmake/help/v3.15/prop_tgt/MSVC_RUNTIME_LIBRARY.html
# The policy is needed to set the runtime library. This chosen runtime
# library is described here:
#  https://docs.microsoft.com/en-us/cpp/build/reference/md-mt-ld-use-run-time-library?view=vs-2019
# Selecting it fixes a crash observed when running libtorch with multiple threads
# after compilation.
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
cmake_policy(SET CMP0091 NEW)
endif()

## See https://discuss.pytorch.org/t/libtorch-static-library/73178/9

## https://stackoverflow.com/questions/51047978/cmake-could-not-find-jni
set(JAVA_AWT_LIBRARY NotNeeded)
set(JAVA_JVM_LIBRARY NotNeeded)

# Note that we use the value of the environment variable STATIC_PYTORCH if available
# TODO Currently, STATIC_PYTORCH ON only works for MacOS. Windows compiles but
#   fails to load at runtime with "DLL initializer failed"
option(STATIC_PYTORCH "Is Pytorch Statically linked?" OFF)

if(DEFINED ENV{STATIC_PYTORCH})
set(STATIC_PYTORCH $ENV{STATIC_PYTORCH})
endif()
if(DEFINED ENV{TORCH_DIR})
# Fix the path separator characters, so that cmake doesn't
# terminate with "/s isn't a valid control character"-style errors
file(TO_CMAKE_PATH "$ENV{TORCH_DIR}" TORCH_DIR)
endif()
message (STATUS "Static Pytorch: ${STATIC_PYTORCH}")
message (STATUS "torch dir?: ${TORCH_DIR}")

if (NOT TORCH_DIR)
get_filename_component(TORCH_DIR
                       "${CMAKE_CURRENT_LIST_DIR}/../../../libtorch"
                       ABSOLUTE)
endif()
message (STATUS "final torch dir: ${TORCH_DIR}")

list(APPEND CMAKE_MODULE_PATH "${TORCH_DIR}/../cmake/Modules")

if (NOT WIN32)
list(APPEND CMAKE_MODULE_PATH "/usr/local/cmake/Modules")
endif()

# If we're statically linking pytorch, we link in some dynamic libraries
# from conda such as mkl and mkl-include
if (STATIC_PYTORCH)

  if (NOT WIN32)
    if (NOT CONDA_PATH)
    execute_process(COMMAND which conda OUTPUT_VARIABLE CONDA_PATH)
    get_filename_component(CONDA_PATH
                           "${CONDA_PATH}/../../"
                           ABSOLUTE)
    endif()
    message (STATUS "Conda Path: ${CONDA_PATH}")

  list(APPEND CMAKE_PREFIX_PATH "${CONDA_PATH}/share/cmake-3.14")
  endif()
endif()

message (STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message (STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}/share/cmake-3.14")


# Define project and related variables
# (required by sbt-jni) please use semantic versioning
#
project (torch_swig)
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)

# Taken from pytorch/CMakeLists.txt
set(CMAKE_CXX_STANDARD 14)

# Setup JNI
find_package(JNI REQUIRED)
if (JNI_FOUND)
    message (STATUS "JNI include directories: ${JNI_INCLUDE_DIRS}")
endif()


# Torch
find_package(Torch REQUIRED PATHS "${TORCH_DIR}/share/cmake")
if (TORCH_FOUND)
    message (STATUS "Torch include directories: ${TORCH_INCLUDE_DIRS}")
endif()

if (STATIC_PYTORCH)
    #Protobuf
    set(Protobuf_DEBUG ON)
    if (WIN32)
        find_package(Protobuf REQUIRED PATHS "${TORCH_DIR}/cmake")
    else()
        find_package(Protobuf REQUIRED PATHS "${TORCH_DIR}/lib/cmake")
    endif()
    message (STATUS "${Protobuf_FOUND}")
    message (STATUS "${Protobuf_VERSION}")
    # TODO: these aren't getting set???
    message (STATUS "Protobuf include directories: ${Protobuf_INCLUDE_DIRS}")
    message (STATUS "Protobuf libraries: ${Protobuf_LIBRARIES}")
    message (STATUS "Protobuf libraries: ${PROTOBUF_LIBRARIES}")

endif(STATIC_PYTORCH)

# Include directories
include_directories(.)
include_directories(include)
include_directories(${JNI_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${PROTOBUF_INCLUDE_DIRS})
if (NOT STATIC_PYTORCH)
    include_directories(${GFLAGS_INCLUDE_DIR})
endif (NOT STATIC_PYTORCH)

# Sources
file(GLOB LIB_SRC
  "../../target/src_managed/native/*.cxx"
)

# compiler flags
add_compile_options(-O2)
if (MSVC)
  add_compile_options(/bigobj)
else()
  # swig depends on type punning to do its work and you have to tell gcc you're going to do that.
  add_compile_options(-fno-strict-aliasing)
endif()
# -fpermissive: gcc (correctly, but pedantically) regards long and long long as different types, but
# they're actually the same under all sane modern UNIXes.
# (They are different under windows and we'll need to address that)
# (adpauls): I'm not sure why -fpermissive is gone but I'll leave this comment
# here anyways


# TODO: this shouldn't be necessary, but FindTorch and FindProtobuf aren't working
# and there are probably other libs (onnx, fbgemm) that aren't getting picked up either.
# a bunch of things aren't getting found because pytorch really doesn't like static linking, but we will prevail
if (STATIC_PYTORCH)
    set(ABS_LIB_DIR "${TORCH_DIR}/lib/")
    link_directories("${ABS_LIB_DIR}")
endif (STATIC_PYTORCH)

# Setup installation targets
# (required by sbt-jni) major version should always be appended to library name
#
set (LIB_NAME ${PROJECT_NAME}${PROJECT_VERSION_MAJOR})
add_library(${LIB_NAME} SHARED ${LIB_SRC})

if (STATIC_PYTORCH)
    if (WIN32)
      target_link_libraries(${LIB_NAME} libprotobuf onnx onnx_proto caffe2_protos clog cpuinfo c10 foxi_loader Caffe2_perfkernels_avx Caffe2_perfkernels_avx2 Caffe2_perfkernels_avx512)
    else()
      # TODO "-framework Accelerate" is MacOS specific. Will have to figure out what to do for Linux.
      target_link_libraries(${LIB_NAME} -lprotobuf -lasmjit -lfbgemm -lcaffe2_protos -lonnx -lonnx_proto -lfoxi_loader -lqnnpack -lsleef "-framework Accelerate" -Wl,-rpath,${CONDA_PATH}/lib)
    endif()
endif (STATIC_PYTORCH)

# libs
if (NOT STATIC_PYTORCH)
    target_link_libraries(${LIB_NAME} "${MKL_LIBRARIES}")
    target_link_libraries(${LIB_NAME} "${GFLAGS_LIBRARIES}")
endif (NOT STATIC_PYTORCH)
target_link_libraries(${LIB_NAME} "${TORCH_LIBRARIES}")
if (NOT WIN32)
    target_link_libraries(${LIB_NAME} "-lpthread")
endif()

message (STATUS "torch libs : ${TORCH_LIBRARIES}")
install(TARGETS ${LIB_NAME} LIBRARY DESTINATION .)
