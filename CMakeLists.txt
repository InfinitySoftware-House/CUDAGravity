cmake_minimum_required(VERSION 3.24)
project(Gravity LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Build for the GPU in this machine by default (RTX 5060 = sm_120).
# Override with -DCMAKE_CUDA_ARCHITECTURES=89 etc. for other cards.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

# --- GLFW (windowing / input) ---
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.4)
FetchContent_MakeAvailable(glfw)

# --- GLAD2 (OpenGL loader, generated at configure time) ---
FetchContent_Declare(glad
  GIT_REPOSITORY https://github.com/Dav1dde/glad.git
  GIT_TAG        v2.0.6
  SOURCE_SUBDIR  cmake)
FetchContent_MakeAvailable(glad)
glad_add_library(glad_gl_core STATIC API gl:core=3.3)

find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(gravity
  src/main.cpp
  src/barnes_hut.cu)

target_link_libraries(gravity PRIVATE
  glfw
  glad_gl_core
  OpenGL::GL
  CUDA::cudart)

set_target_properties(gravity PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF)
