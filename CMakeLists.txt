cmake_minimum_required(VERSION 3.20)
project(sloshing3d VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Options ---
option(SLOSHING_BUILD_TESTS "Build unit tests" ON)
option(SLOSHING_USE_ACCELERATE "Use Apple Accelerate framework" OFF)

# Auto-detect Apple platform
if(APPLE)
    set(SLOSHING_USE_ACCELERATE ON)
endif()

# --- Dependencies via FetchContent ---
include(FetchContent)

# GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# GLM
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
)
FetchContent_MakeAvailable(glm)

# glad (OpenGL 4.1 Core loader) — we generate and include it directly
# We'll add glad as a local source in external/

# GoogleTest
if(SLOSHING_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# --- glad library (OpenGL 4.1 Core) ---
add_library(glad STATIC
    external/glad/src/glad.c
)
target_include_directories(glad PUBLIC external/glad/include)

# --- Main library (solver + simulation) ---
file(GLOB_RECURSE SLOSHING_SOURCES src/*.cpp)
# Remove main.cpp from library sources
list(FILTER SLOSHING_SOURCES EXCLUDE REGEX ".*main\\.cpp$")

find_package(Threads REQUIRED)

add_library(sloshing_lib STATIC ${SLOSHING_SOURCES})
target_include_directories(sloshing_lib PUBLIC include)
target_link_libraries(sloshing_lib PUBLIC glm::glm glad glfw Threads::Threads)

if(SLOSHING_USE_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        target_link_libraries(sloshing_lib PUBLIC ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(sloshing_lib PUBLIC SLOSHING_USE_ACCELERATE)
    endif()
endif()

# --- Executable ---
add_executable(sloshing3d src/main.cpp)
target_link_libraries(sloshing3d PRIVATE sloshing_lib)

# Copy shaders to build directory
file(GLOB SHADER_FILES shaders/*.vert shaders/*.frag)
foreach(SHADER ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    configure_file(${SHADER} ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME} COPYONLY)
endforeach()

# --- Tests ---
if(SLOSHING_BUILD_TESTS)
    file(GLOB_RECURSE TEST_SOURCES tests/*.cpp)
    add_executable(sloshing_tests ${TEST_SOURCES})
    target_link_libraries(sloshing_tests PRIVATE sloshing_lib GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(sloshing_tests)
endif()
