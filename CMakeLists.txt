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

option(USE_MERO "Build Mero Object Store Backend" OFF)

find_package(MPI REQUIRED)
find_package(HDF5 COMPONENTS C HL)
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

if(MPI_C_FOUND AND HDF5_FOUND AND PROTO-C_FOUND)
    # build libaoi
    add_library(noa SHARED ${NOA_SRC})

    # link dep
    target_link_libraries(noa PRIVATE
        ${UUID_LINK_LIBRARIES}
        ${PROTO-C_LINK_LIBRARIES}
        ${HDF5_C_LIBRARIES}
        ${HDF5_HL_LIBRARIES}
        MPI::MPI_C)

    # header files including the generated proto
    target_include_directories(noa PRIVATE 
        include
        include/private
        ${PROTO-C_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}
        ${UUID_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}
        ${MPI_INCLUDE_PATH})

    if(USE_MERO)
        target_link_libraries(noa PRIVATE /home/users/project/SHARED/libaoi/libaoi.so)
        target_include_directories(noa PRIVATE /home/users/project/SHARED/libaoi)
        target_compile_definitions(noa PRIVATE USE_MERO)
    endif()

    # PIC for shared library
    target_compile_options(noa PRIVATE -O3 -fPIC -Wall)

    # build example app
    add_executable(app.out ${APP_SRC})
    target_compile_options(app.out PUBLIC -O3 -Wall)
    target_link_libraries(app.out PUBLIC
        ${PROTO-C_LINK_LIBRARIES}
        noa
        MPI::MPI_C)
    target_include_directories(app.out PUBLIC
        ${PROTO-C_INCLUDE_DIRS}
        ${MPI_INCLUDE_PATH}
        ${CMAKE_CURRENT_BINARY_DIR}/src
        include) 
endif()

message("MPI include path:  ${MPI_C_INCLUDE_DIRS}")
message("MPI library path:  ${MPI_C_LIBRARIES}")
message("HDF5 library path: ${HDF5_C_LIBRARIES}")
message("HDF5 include path: ${HDF5_INCLUDE_DIRS}")
message("Protoc library:    ${PROTO-C_LINK_LIBRARIES}")
message("Protoc include:    ${PROTO-C_INCLUDE_DIRS}")
message("UUID library:      ${UUID_LINK_LIBRARIES}")
if (USE_MERO)
    message("Use Mero: ON")
else()
    message("Use Mero: OFF")
endif()
