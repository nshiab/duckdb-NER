cmake_minimum_required(VERSION 3.12)

# Set extension name here
set(TARGET_NAME ner)

# DuckDB's extension distribution supports vcpkg. As such, dependencies can be added in ./vcpkg.json and then
# used in cmake with find_package. Feel free to remove or replace with other dependencies.
# Note that it should also be removed from vcpkg.json to prevent needlessly installing it..

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include)

set(EXTENSION_SOURCES src/ner_extension.cpp src/ner_model.cpp)

# bert.cpp Configuration
set(BERT_STATIC ON CACHE BOOL "" FORCE)
set(BERT_NATIVE OFF CACHE BOOL "" FORCE)
set(BERT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BERT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Patch submodules to avoid build issues in CI environments
function(patch_cmake_file file_path search_string replace_string)
    if(EXISTS ${file_path})
        file(READ ${file_path} content)
        string(REPLACE "${search_string}" "${replace_string}" content "${content}")
        file(WRITE ${file_path} "${content}")
    endif()
endfunction()

patch_cmake_file(${CMAKE_CURRENT_SOURCE_DIR}/third_party/bert.cpp/ggml/CMakeLists.txt "cmake_minimum_required (VERSION 3.0)" "cmake_minimum_required (VERSION 3.12)")
patch_cmake_file(${CMAKE_CURRENT_SOURCE_DIR}/third_party/bert.cpp/ggml/CMakeLists.txt "cmake_minimum_required (VERSION 3.5)" "cmake_minimum_required (VERSION 3.12)")
patch_cmake_file(${CMAKE_CURRENT_SOURCE_DIR}/third_party/bert.cpp/CMakeLists.txt "add_subdirectory(examples)" "")
patch_cmake_file(${CMAKE_CURRENT_SOURCE_DIR}/third_party/bert.cpp/CMakeLists.txt "add_subdirectory(models)" "")
patch_cmake_file(${CMAKE_CURRENT_SOURCE_DIR}/third_party/bert.cpp/ggml/CMakeLists.txt "add_subdirectory(examples)" "")
patch_cmake_file(${CMAKE_CURRENT_SOURCE_DIR}/third_party/bert.cpp/ggml/CMakeLists.txt "add_subdirectory(tests)" "")

# ggml Configuration (inside bert.cpp)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE) # Disable metal for CI portability for now

add_subdirectory(third_party/bert.cpp)
include_directories(third_party/bert.cpp)
include_directories(third_party/bert.cpp/ggml/include/ggml)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Link bert/ggml in both the static library as the loadable extension
target_link_libraries(${EXTENSION_NAME} bert ggml)
target_link_libraries(${LOADABLE_EXTENSION_NAME} bert ggml)

set(BERT_EXPORT_TARGETS bert ggml)

install(
  TARGETS ${BERT_EXPORT_TARGETS}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
