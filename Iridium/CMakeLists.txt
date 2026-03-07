# Iridium-Next Satellite Tracking Examples
# Uses UKF with AOA and Doppler measurements

# Basic AOA tracking simulation
add_executable(iridium_aoa_tracking iridium_aoa_tracking.cpp)
target_include_directories(iridium_aoa_tracking PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(iridium_aoa_tracking PRIVATE cxx_std_20)
if(UNIX)
    target_link_libraries(iridium_aoa_tracking PRIVATE m)
endif()

# AOA vs AOA+Doppler comparison tool
add_executable(compare_aoa_doppler compare_aoa_doppler.cpp)
target_include_directories(compare_aoa_doppler PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(compare_aoa_doppler PRIVATE cxx_std_20)
if(UNIX)
    target_link_libraries(compare_aoa_doppler PRIVATE m)
endif()

# Complete Iridium tracking system with all features
add_executable(iridium_tracking_complete iridium_tracking_complete.cpp)
target_include_directories(iridium_tracking_complete PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(iridium_tracking_complete PRIVATE cxx_std_20)
if(UNIX)
    target_link_libraries(iridium_tracking_complete PRIVATE m)
endif()
