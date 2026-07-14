project(rpphal)

set(RPP_HAL_VERSION "${rpp_VERSION}" CACHE INTERNAL "")
set(RPP_HAL_LIBRARIES "rpphal" CACHE INTERNAL "")
set(RPP_HAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")
set(RPP_HAL_HEADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/rpp_hal_core.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/rpp_hal_imgproc.hpp"
  CACHE INTERNAL "")

find_package(rpp 3.1.0 REQUIRED)

add_library(rpphal STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/rpp_utils.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/rpp_executor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core_rpp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgproc_rpp.cpp"
)

target_include_directories(rpphal PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  ${CMAKE_SOURCE_DIR}/modules/core/include
  ${CMAKE_SOURCE_DIR}/modules/imgproc/include
)

target_link_libraries(rpphal PUBLIC rpp::rpp)

# RPP 3.x unified API exposes both CPU (HOST) and GPU (HIP) backends from the
# same library when built with BACKEND=HIP. Only in that case do we need HIP
# headers/runtime and the RPP_BACKEND_HIP compile definition.
if(rpp_BACKEND_TYPE STREQUAL "HIP")
  if(NOT TARGET hip::host AND EXISTS "/opt/rocm/lib/libamdhip64.so")
    target_link_libraries(rpphal PUBLIC /opt/rocm/lib/libamdhip64.so)
  elseif(NOT TARGET hip::host)
    find_package(hip REQUIRED CONFIG HINTS ${ROCM_PATH}/lib/cmake/hip)
    target_link_libraries(rpphal PUBLIC hip::host)
  endif()
  target_compile_definitions(rpphal PUBLIC GPU_SUPPORT=1 __HIP_PLATFORM_AMD__=1 RPP_BACKEND_HIP=1)
endif()

set_target_properties(rpphal PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH} DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}")

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(rpphal EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()
