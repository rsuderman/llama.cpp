# The minimum revision records HRX behavior required by the backend and its Loom kernels.
# Source builds must prove that revision is in HEAD's ancestry before configuring dependencies.
# Only local Git history is read; missing history is an error, never a reason to fetch or skip the check.

function(ggml_hrx_check_minimum_revision SOURCE_DIR PIN_FILE)
    file(READ "${PIN_FILE}" GGML_HRX_MINIMUM_REVISION)
    string(STRIP "${GGML_HRX_MINIMUM_REVISION}" GGML_HRX_MINIMUM_REVISION)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PIN_FILE}")

    set(GGML_HRX_REVISION_HELP
        "HRX_SOURCE_DIR: ${SOURCE_DIR}\nRequired HRX commit: ${GGML_HRX_MINIMUM_REVISION}\nUse an HRX Git checkout at this commit or a descendant. Fetch missing history manually (unshallow a shallow clone if needed), then rerun CMake.")
    find_package(Git QUIET)
    if(NOT GIT_FOUND)
        message(FATAL_ERROR "Git is required to verify the HRX minimum revision. Install Git and rerun CMake.\n${GGML_HRX_REVISION_HELP}")
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" rev-parse --show-toplevel
        RESULT_VARIABLE GGML_HRX_ROOT_RESULT
        OUTPUT_VARIABLE GGML_HRX_GIT_ROOT
        ERROR_VARIABLE GGML_HRX_GIT_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT GGML_HRX_ROOT_RESULT EQUAL 0)
        message(FATAL_ERROR "Cannot identify the HRX Git checkout: ${GGML_HRX_GIT_ERROR}\n${GGML_HRX_REVISION_HELP}")
    endif()

    # Git also discovers enclosing repositories; an unpacked HRX tree inside one is not an HRX checkout.
    get_filename_component(GGML_HRX_SOURCE_REAL "${SOURCE_DIR}" REALPATH)
    get_filename_component(GGML_HRX_GIT_ROOT_REAL "${GGML_HRX_GIT_ROOT}" REALPATH)
    if(NOT GGML_HRX_SOURCE_REAL STREQUAL GGML_HRX_GIT_ROOT_REAL)
        message(FATAL_ERROR "HRX_SOURCE_DIR must be the root of its own Git checkout, not a directory inside ${GGML_HRX_GIT_ROOT}.\n${GGML_HRX_REVISION_HELP}")
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" rev-parse --verify "HEAD^{commit}"
        RESULT_VARIABLE GGML_HRX_HEAD_RESULT
        OUTPUT_VARIABLE GGML_HRX_HEAD
        ERROR_VARIABLE GGML_HRX_GIT_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT GGML_HRX_HEAD_RESULT EQUAL 0)
        message(FATAL_ERROR "Cannot resolve HRX HEAD: ${GGML_HRX_GIT_ERROR}\n${GGML_HRX_REVISION_HELP}")
    endif()
    string(APPEND GGML_HRX_REVISION_HELP "\nActual HRX HEAD: ${GGML_HRX_HEAD}")

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" cat-file -e "${GGML_HRX_MINIMUM_REVISION}^{commit}"
        RESULT_VARIABLE GGML_HRX_PIN_RESULT
        ERROR_VARIABLE GGML_HRX_GIT_ERROR)
    if(NOT GGML_HRX_PIN_RESULT EQUAL 0)
        message(FATAL_ERROR "Required HRX commit is unavailable locally: ${GGML_HRX_GIT_ERROR}\n${GGML_HRX_REVISION_HELP}")
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" merge-base --is-ancestor "${GGML_HRX_MINIMUM_REVISION}" "${GGML_HRX_HEAD}"
        RESULT_VARIABLE GGML_HRX_ANCESTRY_RESULT
        ERROR_VARIABLE GGML_HRX_GIT_ERROR)
    if(NOT GGML_HRX_ANCESTRY_RESULT EQUAL 0)
        message(FATAL_ERROR "HRX HEAD does not have the required commit in its available ancestry: ${GGML_HRX_GIT_ERROR}\n${GGML_HRX_REVISION_HELP}")
    endif()
    message(STATUS "HRX revision ${GGML_HRX_HEAD} satisfies minimum ${GGML_HRX_MINIMUM_REVISION}")
endfunction()
