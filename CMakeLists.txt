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
    # build the storage engine to interact with motr if using cotrx
    if(USE_MERO)
        pkg_search_module(JSON-C REQUIRED json-c)
        pkg_search_module(MOTR REQUIRED motr)
        find_package(OpenSSL REQUIRED)
        add_library(noa_motr STATIC src/storage/noa_motr.c)
        target_include_directories(noa_motr PRIVATE
                                   include/storage
                                   ${JSON-C_INCLUDE_DIRS}
                                   ${MOTR_INCLUDE_DIRS}
                                   ${SSL_INCLUDE_DIRS})
        target_link_libraries(noa_motr PRIVATE
                              ${JSON-C_LINK_LIBRARIES}
                              ${SSL_LINK_LIBRARIES}
                              ${MOTR_LIBRARIES})

        # hack to fix a conflict with the config.h from json-c
        list(REMOVE_ITEM MOTR_CFLAGS "-includeconfig.h")
        list(APPEND MOTR_CFLAGS -include/usr/include/motr/config.h)
        target_compile_options(noa_motr PUBLIC -fPIC)
        target_compile_options(noa_motr PRIVATE
                               ${MOTR_CFLAGS})
    endif()

    # build libaoi
    add_library(noa SHARED ${NOA_SRC})

    # link dep
    target_link_libraries(noa PRIVATE
        ${UUID_LINK_LIBRARIES}
        ${PROTO-C_LINK_LIBRARIES}
        ${HDF5_C_LIBRARIES}
        ${HDF5_HL_LIBRARIES}
        MPI::MPI_C)

    # link to motr storage engine if enabling cortx
    if(USE_MERO)
        target_link_libraries(noa PRIVATE noa_motr)
    endif()

    # header files including the generated proto
    target_include_directories(noa PRIVATE 
        include
        include/private
        include/storage
        ${PROTO-C_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}
        ${UUID_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}
        ${MPI_INCLUDE_PATH})

    # PIC for shared library
    target_compile_options(noa PRIVATE -O3 -fPIC -Wall)

    # build example app
    add_executable(app.out ${APP_SRC})
    target_compile_options(app.out PUBLIC -O3 -Wall)
    # add mero def
    if(USE_MERO)
        target_compile_definitions(noa PRIVATE USE_MERO)
    endif()

    target_link_libraries(app.out PUBLIC
        ${PROTO-C_LINK_LIBRARIES}
        noa
        MPI::MPI_C)

    target_include_directories(app.out PUBLIC
        ${PROTO-C_INCLUDE_DIRS}
        ${MPI_INCLUDE_PATH}
        ${CMAKE_CURRENT_BINARY_DIR}/src
        include) 
else()
    message("Error finding packages!")
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
