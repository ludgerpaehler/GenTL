add_executable(mcmc mcmc.cpp)
target_include_directories(mcmc PRIVATE ../include)
target_include_directories(mcmc PRIVATE ../third_party/Eigen)

add_executable(particle_filter particle_filter.cpp)
target_include_directories(particle_filter PRIVATE ../include)
target_include_directories(particle_filter PRIVATE ../third_party/Eigen)

add_executable(opengen_ss_benchmarks opengen_ss_example.cpp)
target_include_directories(opengen_ss_benchmarks PRIVATE ../include)
target_include_directories(opengen_ss_benchmarks PRIVATE ../third_party/Eigen)

add_executable(opengen_handcoded opengen_handcoded.cpp)
target_include_directories(opengen_handcoded PRIVATE ../include)
target_include_directories(opengen_handcoded PRIVATE ../third_party/Eigen)
