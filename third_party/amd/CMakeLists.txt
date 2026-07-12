include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  find_package(LLD REQUIRED CONFIG PATHS "${LLD_DIR}" NO_DEFAULT_PATH)
  include_directories(${LLD_INCLUDE_DIRS})
  message(STATUS "Found LLD distro-package @ ${LLD_DIR} and LLD include dirs @ ${LLD_INCLUDE_DIRS}")
  # PerfModelBindings.cpp holds the single-source pybind API (also used by the
  # standalone perf_model.so). It calls generateCandidates/rankConfigs/etc. from
  # TritonAMDPerfModel (PerfModel.cpp). We do NOT add TritonAMDPerfModel to
  # LINK_LIBS here: it is registered via add_triton_library, so it is already
  # linked into libtriton through the global TRITON_LIBS aggregation (see
  # top-level CMakeLists.txt: TRITON_LIBRARIES -> target_link_libraries(triton)).
  # This is the same resolution path triton_amd.cc used before this refactor;
  # adding it explicitly would double-link the OBJECT library (duplicate symbols).
  add_triton_plugin(TritonAMD
    ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_amd.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/python/PerfModelBindings.cpp
    LINK_LIBS TritonAMDGPUToLLVM TritonAMDGPUTransforms TritonAMDGPUDialectToLLVM)
  # Let the plugin translation units find PerfModelBindings.h.
  target_include_directories(TritonAMD PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/python)
  target_link_libraries(TritonAMD PRIVATE Python3::Module pybind11::headers lldCommon lldELF)
endif()
add_subdirectory(test)
add_subdirectory(unittest)
