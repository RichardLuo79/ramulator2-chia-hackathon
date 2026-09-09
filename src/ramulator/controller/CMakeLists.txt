add_library(ramulator-controller OBJECT)

target_sources(
  ramulator-controller PRIVATE
  i_controller.h
  controller_base.h
  controller_base.cpp

  scheduler/i_scheduler.h
  scheduler/impl/frfcfs.cpp
  scheduler/impl/frfcfs_rowhit.cpp

  refresh/i_refresh_manager.h
  refresh/impl/all_bank.cpp
  refresh/impl/per_bank.cpp
  refresh/impl/hbm_per_bank_refresh.cpp
  refresh/impl/no_refresh.cpp

  rowpolicy/i_row_policy.h
  rowpolicy/impl/open.cpp
  rowpolicy/impl/closed_cap.cpp

  addr_mapper/i_addr_mapper.h
  addr_mapper/addr_mapper_base.h
  addr_mapper/addr_mapper_base.cpp
  addr_mapper/impl/ch_ra_ba_ro_co.cpp
  addr_mapper/impl/ro_ba_ra_co_ch.cpp
  addr_mapper/impl/mop4clxor.cpp
  addr_mapper/impl/passthrough_addr_mapper.cpp
  addr_mapper/impl/rit_addr_mapper.h
  addr_mapper/impl/rit_addr_mapper.cpp

  plugin/i_controller_plugin.h
  plugin/controller_validation_hook.h
  plugin/impl/command_counter.cpp
  plugin/impl/cmd_trace_recorder.cpp
  plugin/impl/req_trace_recorder.cpp
  plugin/impl/issued_command_validation_hook.cpp
  plugin/impl/rfm_manager.cpp
  plugin/impl/para.cpp
  plugin/impl/oracle_rh.cpp
  plugin/impl/twice.cpp
  plugin/impl/graphene.cpp
  plugin/impl/hydra.cpp
  plugin/impl/rrs.cpp
  plugin/impl/aqua.cpp
  plugin/impl/samsung_trr.cpp
  plugin/impl/hynix_trr.cpp
  plugin/impl/ideal_trr.cpp

  plugin/impl/bin_trace_recorder.cpp
  plugin/impl/live_trace_streamer.cpp

  impl/generic_ddr_controller.cpp
  impl/fixed_lat_controller.cpp
  impl/md1_controller.cpp
  impl/wmg1_controller.cpp
  impl/mess_controller.cpp
  impl/lpddr_controller_base.h
  impl/lpddr_controller_base.cpp
  impl/lpddr5_controller.cpp
  impl/lpddr6_controller.cpp
  impl/hbm_controller_base.h
  impl/hbm_controller_base.cpp
  impl/hbm12_controller.cpp
  impl/hbm34_controller.cpp
  impl/gddr7_controller.cpp
  impl/prac_controller.cpp
  impl/blockhammer_controller.cpp
)

if(RAMULATOR_CHIA_MODEL_API)
  target_sources(ramulator-controller PRIVATE
    impl/atomic_model_controller.cpp atomic_model/loader.cpp)
  target_link_libraries(ramulator PRIVATE ${CMAKE_DL_LIBS})
  add_library(ramulator-atomic MODULE ${PROJECT_SOURCE_DIR}/tools/chia_loop/model/seed.cpp)
elseif(RAMULATOR_CHIA_TOOLS)
  # Keep Atomic registration out of the trusted library. The runner loads this
  # module only after closing frontend inputs and enforcing its access policy.
  add_library(ramulator-atomic MODULE impl/atomic_controller.cpp)
  target_link_libraries(ramulator-atomic PRIVATE ramulator fmt::fmt yaml-cpp::yaml-cpp)
else()
  target_sources(ramulator-controller PRIVATE impl/atomic_controller.cpp)
endif()

if(RAMULATOR_CHIA_TOOLS)
  set_target_properties(ramulator-atomic PROPERTIES
    PREFIX ""
    OUTPUT_NAME "candidate"
    LIBRARY_OUTPUT_DIRECTORY "${RAMULATOR_OUTPUT_DIRECTORY}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN"
  )
endif()

target_link_libraries(
  ramulator
  PRIVATE
  ramulator-controller
)
