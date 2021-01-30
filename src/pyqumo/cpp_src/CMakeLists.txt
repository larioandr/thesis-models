cmake_minimum_required(VERSION 3.17)
project(cpp_src)

set(CMAKE_CXX_STANDARD 14)

add_executable(TandemNetworkSim
        main.cpp
        Statistics.cpp
        Statistics.h
        System.h Records.cpp Records.h Base.cpp Base.h Components.h Components.cpp System.cpp Simulation.h Simulation.cpp Functions.h)
