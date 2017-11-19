cmake_minimum_required(VERSION 3.2)
project(qlearner)

include_directories("qlearner")

add_executable(experiment qlearner/experiment.cc)
