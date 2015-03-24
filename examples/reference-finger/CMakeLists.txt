cmake_minimum_required(VERSION 2.8)

project(MaxCurvature.Example.ReferenceFinger)

file(GLOB_RECURSE MaxCurvature.Example.ReferenceFinger.Sources "src/*.cpp")

add_executable(${PROJECT_NAME} ${MaxCurvature.Example.ReferenceFinger.Sources})

target_link_libraries(${PROJECT_NAME} MaxCurvature)


