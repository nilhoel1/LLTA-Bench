# The following lines of boilerplate have to be in your project's
# temporary.cmake in the top-level project directory
cmake_minimum_required(VERSION 3.16)

# Set the target chip
set(IDF_TARGET "esp32c6")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(esp32c6_benchmark)
