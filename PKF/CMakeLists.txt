# PKF CMakeLists.txt

# Find Eigen3
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 3.4 REQUIRED)
endif()

# Header-only library for PKF core
add_library(PKF_Lib INTERFACE)
# Common/include carries TestCheck.h (and FilterMath.h). Using it as an -I root,
# not a "../../Common/include" relative escape, keeps the headers valid once
# install(DIRECTORY) flattens every -I root into a single include/nlf.
# $<BUILD_INTERFACE>/$<INSTALL_INTERFACE>: PKF_Lib is part of the exported nlf
# package, and install(EXPORT) rejects bare source-tree paths on an exported
# target. The install root is the flat <prefix>/include/nlf.
target_include_directories(PKF_Lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/Common/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/nlf>)
target_link_libraries(PKF_Lib INTERFACE Eigen3::Eigen)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(PKF_Lib INTERFACE OpenMP::OpenMP_CXX)
endif()

# Example Executable
add_executable(pkf_example src/example_main.cpp)
target_link_libraries(pkf_example PRIVATE PKF_Lib OptMathKernels)
enable_lto_if_supported(pkf_example)
nlf_add_warning_flags(pkf_example)

# Test Executable
add_executable(pkf_test tests/test_particle.cpp)
target_link_libraries(pkf_test PRIVATE PKF_Lib OptMathKernels)
enable_lto_if_supported(pkf_test)
nlf_add_warning_flags(pkf_test)

# Const-correctness regression test introduced by the audit-remediation pass.
add_executable(test_particle_const tests/test_particle_const.cpp)
target_link_libraries(test_particle_const PRIVATE PKF_Lib OptMathKernels)
enable_lto_if_supported(test_particle_const)
nlf_add_warning_flags(test_particle_const)

# Warning flags are now applied uniformly via nlf_add_warning_flags() (see root
# CMakeLists.txt). Per-target compile options are kept only for -O3 selection.

# Optimization for Release
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(pkf_example PRIVATE -O3)
    target_compile_options(pkf_test PRIVATE -O3)
    target_compile_options(test_particle_const PRIVATE -O3)
endif()

# --- Vulkan Shader Deployment ---

# 1. Try to find pre-compiled SPV in the dependency build directory
# The dependency puts them in ${BINARY_DIR}/src
# Unset cached result to avoid stale paths after clean builds
unset(VEC_ADD_SHADER_SPV CACHE)
find_file(VEC_ADD_SHADER_SPV "vec_add.comp.spv"
    PATHS
        ${OPTMATH_BINARY_DIR}/src
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
        ${OPTMATH_SOURCE_DIR}/src/vulkan/shaders
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
        find_program(GLSLANG_VALIDATOR glslangValidator)
        if(GLSLANG_VALIDATOR)
            add_custom_command(
                OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vec_add.comp.spv
                COMMAND ${GLSLANG_VALIDATOR} -V -S comp ${VEC_ADD_SHADER_SRC} -o ${CMAKE_CURRENT_BINARY_DIR}/vec_add.comp.spv
                DEPENDS ${VEC_ADD_SHADER_SRC}
                COMMENT "Compiling Vulkan shader vec_add.comp.glsl (glslangValidator)"
            )
            add_custom_target(CompilePKFShaders ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vec_add.comp.spv)
            add_dependencies(pkf_example CompilePKFShaders)
            add_dependencies(pkf_test CompilePKFShaders)
        else()
            message(WARNING "Neither glslc nor glslangValidator found. PKF Vulkan acceleration will fail at runtime (missing vec_add.comp.spv).")
        endif()
    endif()
else()
    message(WARNING "Could not find vec_add.comp.spv or vec_add.comp.glsl in OptimizedKernels dependency.")
    message(STATUS "Search paths tried (Binary): ${OPTMATH_BINARY_DIR}/src")
    message(STATUS "Search paths tried (Source): ${OPTMATH_SOURCE_DIR}/src/vulkan/shaders")
endif()

# 2026-11-15 audit: exact moments after init / all-dead reset; resampling precision.
add_executable(test_pkf_audit tests/test_pkf_audit.cpp)
target_link_libraries(test_pkf_audit PRIVATE PKF_Lib OptMathKernels)
enable_lto_if_supported(test_pkf_audit)
nlf_add_warning_flags(test_pkf_audit)
