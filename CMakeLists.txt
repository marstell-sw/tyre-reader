cmake_minimum_required(VERSION 3.16)

project(tyre_reader_v3 VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TYRE_READER_BUILD_CORE "Build the OpenCV/Tesseract analysis backend." ON)
option(TYRE_READER_BUILD_GUI "Build the Qt debug GUI." ON)

if(TYRE_READER_BUILD_CORE)
    find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
    find_package(PkgConfig QUIET)

    set(TYRE_READER_CORE_AVAILABLE OFF)
    if(OpenCV_FOUND AND PkgConfig_FOUND)
        pkg_check_modules(TESSERACT QUIET IMPORTED_TARGET tesseract)
        pkg_check_modules(LEPTONICA QUIET IMPORTED_TARGET lept)
        if(TESSERACT_FOUND AND LEPTONICA_FOUND)
            set(TYRE_READER_CORE_AVAILABLE ON)
        endif()
    endif()

    if(TYRE_READER_CORE_AVAILABLE)
        add_library(tyre_reader_core
            src/TesseractOcrEngine.cpp
            src/ImagePreprocessor.cpp
            src/TyreAnalyzer.cpp
            src/DatasetBenchmark.cpp
        )

        target_include_directories(tyre_reader_core
            PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${OpenCV_INCLUDE_DIRS}
        )

        target_link_libraries(tyre_reader_core
            PUBLIC
                ${OpenCV_LIBS}
                PkgConfig::TESSERACT
                PkgConfig::LEPTONICA
        )

        add_executable(tyre_reader_v3
            src/main.cpp
        )

        target_link_libraries(tyre_reader_v3
            PRIVATE
                tyre_reader_core
        )

        add_executable(tyre_dataset_prep
            src/DatasetImagePreprocessor.cpp
            src/dataset_prep_main.cpp
        )

        target_include_directories(tyre_dataset_prep
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${OpenCV_INCLUDE_DIRS}
        )

        target_link_libraries(tyre_dataset_prep
            PRIVATE
                ${OpenCV_LIBS}
        )

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(tyre_reader_core PRIVATE -Wall -Wextra -Wpedantic)
            target_compile_options(tyre_reader_v3 PRIVATE -Wall -Wextra -Wpedantic)
            target_compile_options(tyre_dataset_prep PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    else()
        message(WARNING "Skipping analysis backend targets because OpenCV, Tesseract, or Leptonica were not found.")
    endif()
endif()

if(TYRE_READER_BUILD_GUI)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)

    find_package(Qt6 QUIET COMPONENTS Core Gui Widgets)
    if(Qt6_FOUND)
        add_executable(tyre_reader_debug_gui
            gui/main_qt.cpp
            gui/MainWindow.cpp
            gui/MainWindow.h
            gui/BackendClient.cpp
            gui/BackendClient.h
        )

        target_link_libraries(tyre_reader_debug_gui
            PRIVATE
                Qt6::Core
                Qt6::Gui
                Qt6::Widgets
        )

        target_include_directories(tyre_reader_debug_gui
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/gui
        )

        target_compile_definitions(tyre_reader_debug_gui
            PRIVATE
                TYRE_READER_REPO_ROOT="${CMAKE_SOURCE_DIR}"
        )

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(tyre_reader_debug_gui PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    else()
        message(WARNING "Skipping Qt debug GUI because Qt6 Widgets was not found.")
    endif()
endif()
