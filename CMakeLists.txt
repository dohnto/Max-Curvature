cmake_minimum_required(VERSION 2.8)
project(MaxCurvature)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -g")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

file(GLOB_RECURSE MaxCurvature.Sources "src/*.cpp")
file(GLOB_RECURSE MaxCurvature.Headers "include/*.h")

add_library(${PROJECT_NAME} SHARED ${MaxCurvature.Sources} ${MaxCurvature.Headers})
target_include_directories(${PROJECT_NAME} PUBLIC include)

# OpenCV
find_package(OpenCV 2.4.7 EXACT REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( MaxCurvature ${OpenCV_LIBS} )

install (TARGETS ${PROJECT_NAME} DESTINATION lib)
install (FILES ${MaxCurvature.Headers} DESTINATION include)

add_subdirectory(examples)
