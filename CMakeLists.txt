cmake_minimum_required(VERSION 3.16)

project(tyre_reader_v3 VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(PkgConfig REQUIRED)

pkg_check_modules(TESSERACT REQUIRED IMPORTED_TARGET tesseract)
pkg_check_modules(LEPTONICA REQUIRED IMPORTED_TARGET lept)

add_executable(tyre_reader_v3
    src/TesseractOcrEngine.cpp
    src/ImagePreprocessor.cpp
    src/TyreAnalyzer.cpp
    src/DatasetBenchmark.cpp
    src/main.cpp
)

target_include_directories(tyre_reader_v3
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(tyre_reader_v3
    PRIVATE
        ${OpenCV_LIBS}
        PkgConfig::TESSERACT
        PkgConfig::LEPTONICA
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tyre_reader_v3 PRIVATE -Wall -Wextra -Wpedantic)
endif()
