# PKF CMakeLists.txt

# Find Eigen3
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 3.4 REQUIRED)
endif()

# Header-only library for PKF core
add_library(PKF_Lib INTERFACE)
target_include_directories(PKF_Lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(PKF_Lib INTERFACE Eigen3::Eigen)

# Example Executable
add_executable(pkf_example src/example_main.cpp)
target_link_libraries(pkf_example PRIVATE PKF_Lib OptMathKernels)

# Test Executable
add_executable(pkf_test tests/test_particle.cpp)
target_link_libraries(pkf_test PRIVATE PKF_Lib OptMathKernels)

# Bearing Only Test
add_executable(test_bearing_only tests/test_bearing_only.cpp)
target_link_libraries(test_bearing_only PRIVATE PKF_Lib OptMathKernels)

# Enable warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(pkf_example PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    target_compile_options(pkf_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    target_compile_options(test_bearing_only PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

# Optimization for Release
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(pkf_example PRIVATE -O3)
    target_compile_options(pkf_test PRIVATE -O3)
    target_compile_options(test_bearing_only PRIVATE -O3)
endif()

# --- Shader Copying Logic ---
if(TARGET optmath_shaders)
    # Determine the source directory for compiled shaders
    # FetchContent usually sets <Name>_BINARY_DIR
    if(DEFINED OptimizedKernels_BINARY_DIR)
        set(SHADERS_SRC_DIR "${OptimizedKernels_BINARY_DIR}/src")
    elseif(DEFINED optimizedkernels_BINARY_DIR)
        set(SHADERS_SRC_DIR "${optimizedkernels_BINARY_DIR}/src")
    else()
        message(WARNING "Could not determine OptimizedKernels binary dir. Shaders might not be copied.")
    endif()

    set(EXPECTED_SHADERS
        vec_add.comp.spv
        vec_sub.comp.spv
        vec_mul.comp.spv
        vec_div.comp.spv
        vec_dot.comp.spv
        vec_norm.comp.spv
        mat_add.comp.spv
        mat_sub.comp.spv
        mat_mul.comp.spv
        mat_transpose.comp.spv
        mat_scale.comp.spv
        mat_vec_mul.comp.spv
        mat_outer_product.comp.spv
        mat_elementwise_mul.comp.spv
        convolution_1d.comp.spv
        convolution_2d.comp.spv
        correlation_1d.comp.spv
        correlation_2d.comp.spv
        reduce_sum.comp.spv
        reduce_max.comp.spv
        reduce_min.comp.spv
        scan_prefix_sum.comp.spv
        fft_radix2.comp.spv
        ifft_radix2.comp.spv
        fft_radix4.comp.spv
        ifft_radix4.comp.spv
    )

    if(DEFINED SHADERS_SRC_DIR)
        foreach(SHADER ${EXPECTED_SHADERS})
            # Copy to PKF binary dir (where executables are)
            add_custom_command(TARGET pkf_example POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${SHADERS_SRC_DIR}/${SHADER}
                $<TARGET_FILE_DIR:pkf_example>/${SHADER}
            )
            add_custom_command(TARGET pkf_test POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${SHADERS_SRC_DIR}/${SHADER}
                $<TARGET_FILE_DIR:pkf_test>/${SHADER}
            )
            add_custom_command(TARGET test_bearing_only POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${SHADERS_SRC_DIR}/${SHADER}
                $<TARGET_FILE_DIR:test_bearing_only>/${SHADER}
            )
        endforeach()
    endif()
endif()
