cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(nz)

#对于AMD GPU 同时使用rocm的libtorch和opengl需要在终端添加
#export ROCM_PATH=/opt/rocm
#export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(MUJOCO_FOLDER /opt/mujoco/lib/cmake)
find_package(mujoco REQUIRED PATHS ${MUJOCO_FOLDER} NO_DEFAULT_PATH)
if (mujoco_FOUND)
message(STATUS "Find mujoco")
endif()

find_package(Eigen3 REQUIRED)

# set(TORCH_FOLDER /home/albusgive/libtorch-cxx11-abi-shared-with-deps-2.6.0+rocm6.2.4/libtorch/share/cmake)
set(TORCH_FOLDER /home/albusgive2/libtorch-cxx11-abi-shared-with-deps-2.6.0+cu124/libtorch/share/cmake)
find_package(Torch REQUIRED PATHS ${TORCH_FOLDER} NO_DEFAULT_PATH)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(gs2mj gs2mj.cpp gamepad.cpp)

target_link_libraries(gs2mj "${TORCH_LIBRARIES}" mujoco::mujoco glut GL GLU glfw ${EIGEN3_LIBS} udev)
