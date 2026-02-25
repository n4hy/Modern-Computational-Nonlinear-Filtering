# PKF CMakeLists.txt

# Find Eigen3
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 3.4 REQUIRED)
endif()

# Header-only library for PKF core
add_library(PKF_Lib INTERFACE)
target_include_directories(PKF_Lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(PKF_Lib INTERFACE Eigen3::Eigen)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(PKF_Lib INTERFACE OpenMP::OpenMP_CXX)
endif()

# Example Executable
add_executable(pkf_example src/example_main.cpp)
target_link_libraries(pkf_example PRIVATE PKF_Lib OptMathKernels)
enable_lto_if_supported(pkf_example)

# Test Executable
add_executable(pkf_test tests/test_particle.cpp)
target_link_libraries(pkf_test PRIVATE PKF_Lib OptMathKernels)
enable_lto_if_supported(pkf_test)

# Enable warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(pkf_example PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    target_compile_options(pkf_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

# Optimization for Release
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(pkf_example PRIVATE -O3)
    target_compile_options(pkf_test PRIVATE -O3)
endif()

# --- Vulkan Shader Deployment ---

# 1. Try to find pre-compiled SPV in the dependency build directory
# The dependency puts them in ${BINARY_DIR}/src
# Unset cached result to avoid stale paths after clean builds
unset(VEC_ADD_SHADER_SPV CACHE)
find_file(VEC_ADD_SHADER_SPV "vec_add.comp.spv"
    PATHS
        ${optimizedkernels_BINARY_DIR}/src
        ${OptimizedKernels_BINARY_DIR}/src
    NO_DEFAULT_PATH
    DOC "Path to compiled vec_add.comp.spv"
)

# 2. Try to find source in the dependency source directory
# The dependency source is in src/vulkan/shaders/vec_add.comp.glsl
unset(VEC_ADD_SHADER_SRC CACHE)
find_file(VEC_ADD_SHADER_SRC "vec_add.comp.glsl"
    PATHS
        ${optimizedkernels_SOURCE_DIR}/src/vulkan/shaders
        ${OptimizedKernels_SOURCE_DIR}/src/vulkan/shaders
    NO_DEFAULT_PATH
    DOC "Path to vec_add.comp.glsl source"
)

if(VEC_ADD_SHADER_SPV AND EXISTS "${VEC_ADD_SHADER_SPV}")
    message(STATUS "Found pre-compiled shader: ${VEC_ADD_SHADER_SPV}")
    configure_file(${VEC_ADD_SHADER_SPV} ${CMAKE_CURRENT_BINARY_DIR}/vec_add.comp.spv COPYONLY)
elseif(VEC_ADD_SHADER_SRC)
    message(STATUS "Found shader source: ${VEC_ADD_SHADER_SRC}")
    find_program(GLSLC_EXECUTABLE glslc)
    if(GLSLC_EXECUTABLE)
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vec_add.comp.spv
            COMMAND ${GLSLC_EXECUTABLE} -fshader-stage=compute ${VEC_ADD_SHADER_SRC} -o ${CMAKE_CURRENT_BINARY_DIR}/vec_add.comp.spv
            DEPENDS ${VEC_ADD_SHADER_SRC}
            COMMENT "Compiling Vulkan shader vec_add.comp.glsl"
        )
        # Ensure example depends on this
        add_custom_target(CompilePKFShaders ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vec_add.comp.spv)
        add_dependencies(pkf_example CompilePKFShaders)
        add_dependencies(pkf_test CompilePKFShaders)
    else()
        message(WARNING "glslc not found. PKF Vulkan acceleration will fail at runtime (missing vec_add.comp.spv).")
    endif()
else()
    message(WARNING "Could not find vec_add.comp.spv or vec_add.comp.glsl in OptimizedKernels dependency.")
    message(STATUS "Search paths tried (Binary): ${optimizedkernels_BINARY_DIR}/src")
    message(STATUS "Search paths tried (Source): ${optimizedkernels_SOURCE_DIR}/src/vulkan/shaders")
endif()
