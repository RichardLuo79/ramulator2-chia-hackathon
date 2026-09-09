find_library(CHIA_SECCOMP_LIBRARY NAMES seccomp libseccomp.so.2)
if(NOT CHIA_SECCOMP_LIBRARY)
  message(FATAL_ERROR "The native CHIA runner requires libseccomp")
endif()

add_executable(ramulator-isolated isolated_sim.cpp)
if(RAMULATOR_CHIA_MODEL_API)
  target_compile_definitions(ramulator-isolated PRIVATE RAMULATOR_CHIA_MODEL_API)
endif()
target_link_libraries(ramulator-isolated PRIVATE
  ramulator fmt::fmt yaml-cpp::yaml-cpp ${CMAKE_DL_LIBS} ${CHIA_SECCOMP_LIBRARY}
)
set_target_properties(ramulator-isolated PROPERTIES
  OUTPUT_NAME "isolated_sim"
  RUNTIME_OUTPUT_DIRECTORY "${RAMULATOR_OUTPUT_DIRECTORY}"
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN"
)

add_executable(ramulator-lifecycle-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tests/utils/simulation_lifecycle_test.cpp
)
target_link_libraries(ramulator-lifecycle-test PRIVATE ramulator fmt::fmt yaml-cpp::yaml-cpp)
add_custom_target(check-simulation-lifecycle
  COMMAND $<TARGET_FILE:ramulator-lifecycle-test>
  DEPENDS ramulator-lifecycle-test
  COMMENT "Checking optimized simulation lifecycle against the tick-by-tick reference"
)

add_executable(ramulator-batch-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tests/utils/batch_simulation_test.cpp)
target_link_libraries(ramulator-batch-test PRIVATE ramulator fmt::fmt yaml-cpp::yaml-cpp)
add_custom_target(check-batch-simulation
  COMMAND $<TARGET_FILE:ramulator-batch-test>
  DEPENDS ramulator-batch-test
  COMMENT "Checking shared batch replay against the tick-only reference"
)

add_executable(ramulator-model-api-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tests/utils/atomic_model_api_test.cpp)
add_executable(ramulator-admission-test EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tests/utils/atomic_admission_test.cpp)
target_link_libraries(ramulator-admission-test PRIVATE ramulator fmt::fmt yaml-cpp::yaml-cpp)
add_custom_target(check-atomic-model-api
  COMMAND $<TARGET_FILE:ramulator-model-api-test>
  COMMAND $<TARGET_FILE:ramulator-admission-test>
  DEPENDS ramulator-model-api-test ramulator-admission-test
  COMMENT "Checking the model value API and trusted atomic admission lifecycle"
)

if(RAMULATOR_CHIA_MODEL_API)
  add_executable(ramulator-model-loader-test EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/tests/utils/atomic_model_loader_test.cpp)
  target_link_libraries(ramulator-model-loader-test PRIVATE ramulator ${CMAKE_DL_LIBS})
  foreach(failure MISSING_FACTORY WRONG_VERSION THROWING_VERSION)
    add_library(ramulator-loader-${failure} MODULE EXCLUDE_FROM_ALL
      ${PROJECT_SOURCE_DIR}/tests/utils/atomic_model_loader_fixture.cpp)
    target_compile_definitions(ramulator-loader-${failure} PRIVATE LOADER_${failure})
  endforeach()
  add_custom_target(check-atomic-model-loader
    COMMAND $<TARGET_FILE:ramulator-model-loader-test> $<TARGET_FILE:ramulator-atomic>
      $<TARGET_FILE:ramulator-loader-MISSING_FACTORY>
      $<TARGET_FILE:ramulator-loader-WRONG_VERSION>
      $<TARGET_FILE:ramulator-loader-THROWING_VERSION>
    DEPENDS ramulator-model-loader-test ramulator-atomic
      ramulator-loader-MISSING_FACTORY ramulator-loader-WRONG_VERSION
      ramulator-loader-THROWING_VERSION
    COMMENT "Checking explicit trusted-host model loading without controller preload")
endif()
