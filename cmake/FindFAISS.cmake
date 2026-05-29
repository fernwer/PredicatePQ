# - Try to find FAISS
# This module defines:
#   FAISS_FOUND
#   FAISS_INCLUDE_DIRS
#   FAISS_LIBRARIES
#   FAISS::faiss

find_path(FAISS_INCLUDE_DIR
  NAMES faiss/Index.h faiss/impl/ProductQuantizer.h
  HINTS
    ${FAISS_ROOT}
    $ENV{FAISS_ROOT}
    ${CMAKE_PREFIX_PATH}
  PATH_SUFFIXES include
)

find_library(FAISS_LIBRARY
  NAMES faiss
  HINTS
    ${FAISS_ROOT}
    $ENV{FAISS_ROOT}
    ${CMAKE_PREFIX_PATH}
  PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FAISS
  REQUIRED_VARS FAISS_INCLUDE_DIR FAISS_LIBRARY
)

if(FAISS_FOUND)
  set(FAISS_INCLUDE_DIRS ${FAISS_INCLUDE_DIR})
  set(FAISS_LIBRARIES ${FAISS_LIBRARY})

  if(NOT TARGET FAISS::faiss)
    add_library(FAISS::faiss UNKNOWN IMPORTED)
    set_target_properties(FAISS::faiss PROPERTIES
      IMPORTED_LOCATION "${FAISS_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${FAISS_INCLUDE_DIR}"
    )
  endif()
endif()

mark_as_advanced(FAISS_INCLUDE_DIR FAISS_LIBRARY)