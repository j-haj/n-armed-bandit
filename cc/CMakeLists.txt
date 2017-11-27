cmake_minimum_required(VERSION 3.2)
project(qlearner)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories("qlearner")

set(HEADERS
  qlearner/bandit.h
  qlearner/learner.h
  qlearner/mathutil.h
)

set(SOURCES
  qlearner/bandit.cc
  qlearner/learner.cc
  qlearner/mathutil.cc
)
add_executable(experiment qlearner/experiment.cc ${HEADERS} ${SOURCES})
