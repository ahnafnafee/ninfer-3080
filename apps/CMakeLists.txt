add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
ninfer_internal_includes(ninfer-serve)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)

add_executable(ninfer-calibrate calibrate/main.cpp)
ninfer_internal_includes(ninfer-calibrate)
target_link_libraries(ninfer-calibrate PRIVATE ninfer_calibration ninfer_runtime_support)

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
ninfer_internal_includes(ninfer-perplexity)
target_link_libraries(ninfer-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)

if(WIN32 AND NINFER_DIRECTSTORAGE)
  foreach(app IN ITEMS ninfer ninfer-serve)
    add_custom_command(TARGET ${app} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${NINFER_DIRECTSTORAGE_DLLS}
              $<TARGET_FILE_DIR:${app}>)
  endforeach()
endif()
