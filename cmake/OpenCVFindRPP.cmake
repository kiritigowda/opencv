# ----------------------------------------------------------------------------
# Detect AMD ROCm Performance Primitives (RPP)
#
# The RPP HAL requires the RPP 3.x unified API (single library exposing both the
# HOST/CPU and HIP/GPU backends). RPP 3.1.0 is the minimum supported version.
# ----------------------------------------------------------------------------

set(RPP_MIN_VERSION "3.1.0")

# First probe without a version constraint so we can emit a precise message when
# an installed-but-too-old RPP is present.
find_package(rpp QUIET)

if(rpp_FOUND AND rpp_VERSION VERSION_LESS RPP_MIN_VERSION)
  set(HAVE_RPP FALSE)
  message(STATUS "RPP: found ${rpp_VERSION} but >= ${RPP_MIN_VERSION} is required, disabling rpp HAL")
elseif(rpp_FOUND)
  set(HAVE_RPP TRUE)
  message(STATUS "RPP found: ${rpp_VERSION}")
  message(STATUS "    includes: ${rpp_INCLUDE_DIR}")
  message(STATUS "    libs: ${rpp_LIBRARIES}")
else()
  set(HAVE_RPP FALSE)
  message(STATUS "RPP: Not found (>= ${RPP_MIN_VERSION} required)")
endif()
