cmake_minimum_required(VERSION 3.10)
project(NoA)
enable_language(C)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

find_package(OpenMP)
find_package(MPI REQUIRED)
find_package(HDF5 COMPONENTS C)
find_package(protobuf-c COMPONENTS C)

find_package(PkgConfig REQUIRED)
pkg_search_module(UUID REQUIRED uuid)
pkg_search_module(PROTO-C REQUIRED libprotobuf-c)

find_program(Protobuf_PROTOC-C_EXECUTABLE
    NAMES protoc-c
    DOC "The Google Protocol Buffers Compiler C"
    PATHS
    ${Protobuf_SRC_ROOT_FOLDER}/vsprojects/${_PROTOBUF_ARCH_DIR}Release
    ${Protobuf_SRC_ROOT_FOLDER}/vsprojects/${_PROTOBUF_ARCH_DIR}Debug
)

PROTOBUF_GENERATE_C(PROTO_SRC PROTO_HEAD src/object.proto)

set(NOA_SRC
    src/initialization.c
    src/container.c
    src/metadata.c
    src/object.c
    src/backends/hdf5_backend.c
    src/backends/binary_backend.c
    ${PROTO_SRC}
)

set(APP_SRC
    src/examples/main.c
)

if(OpenMP_C_FOUND AND MPI_CXX_FOUND AND HDF5_FOUND AND PROTO-C_FOUND)
    # build libaoi
    add_library(noa SHARED ${NOA_SRC})

    # link dep
    target_link_libraries(noa PRIVATE
        ${UUID_LINK_LIBRARIES}
        ${PROTO-C_LINK_LIBRARIES}
        ${HDF5_C_LIBRARIES}
        OpenMP::OpenMP_C
        MPI::MPI_C)

    # header files including the generated proto
    target_include_directories(noa PRIVATE 
        include
        include/private
        ${CMAKE_CURRENT_BINARY_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}
        ${UUID_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}
        ${MPI_INCLUDE_PATH})

    # PIC for shared library
    set_property(TARGET noa PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_compile_options(noa PRIVATE -O3)

    # build example app
    add_executable(app.out ${APP_SRC})
    target_compile_options(app.out PUBLIC -O3 ${ARCH})
    target_link_libraries(app.out PUBLIC
        noa
        MPI::MPI_C)
    target_include_directories(noa PUBLIC
        ${MPI_INCLUDE_PATH}
        ${CMAKE_CURRENT_BINARY_DIR}/src
        include) 
endif()

message("MPI include path:  ${MPI_C_INCLUDE_DIRS}")
message("MPI library path:  ${MPI_C_LIBRARIES}")
message("HDF5 library path: ${HDF5_C_LIBRARIES}")
message("HDF5 include path: ${HDF5_INCLUDE_DIRS}")
message("Protoc library:    ${PROTO-C_LINK_LIBRARIES}")
message("UUID library:      ${UUID_LINK_LIBRARIES}")
