cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_op LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 11)

find_package(Torch REQUIRED)
find_package(CUDA)
find_package(Python COMPONENTS Interpreter Development)

message("Python_FOUND:${Python_FOUND}")
message("Python_VERSION:${Python_VERSION}")
message("Python_Development_FOUND:${Python_Development_FOUND}")
message("Python_LIBRARIES:${Python_LIBRARIES}")

message(STATUS "foo include dir: ${PYTHON_LIBRARIES}")
message(STATUS "foo include dir: ${PYTHON_INCLUDE_DIR}")

# Define our library target
add_library(custom_op SHARED ops.cu)

# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( custom_op
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link against LibTorch
target_link_libraries(custom_op ${TORCH_LIBRARIES})
target_link_libraries(custom_op ${Python_LIBRARIES})
target_include_directories(custom_op PRIVATE "C:/Users/hchoi/anaconda3/envs/torch/include")

# add_executable(test main.cu)
# set_property(TARGET test 
#              PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# target_compile_features(test PRIVATE cxx_std_11)
# # target_link_libraries(test PUBLIC custom_op)
# target_link_libraries(test ${TORCH_LIBRARIES})
# target_link_libraries(test ${Python_LIBRARIES})
# target_include_directories(test PRIVATE "C:/Users/hchoi/anaconda3/envs/torch/include")

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  # set(CMAKE_CUDA_FLAGS "/wd4819 /wd4244 /wd4005 /wd4067 /wd4267 /wd4805 /wd4251 /wd4624 /wd4275")
  # add_compile_options("/wd4819 /wd4244 /wd4005 /wd4067 /wd4267 /wd4805 /wd4251 /wd4624 /wd4275")
  add_custom_command(TARGET custom_op
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:custom_op>)
endif (MSVC)