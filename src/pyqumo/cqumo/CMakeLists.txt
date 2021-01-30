cmake_minimum_required(VERSION 3.17)
project(cqumo)

set(CMAKE_CXX_STANDARD 14)

add_executable(cqumo_tandem tandem/main.cpp tandem/Base.h tandem/Base.cpp tandem/Components.h tandem/Components.cpp tandem/Statistics.h tandem/Statistics.cpp tandem/System.h tandem/System.cpp tandem/Journals.h tandem/Journals.cpp tandem/Simulation.h tandem/Simulation.cpp tandem/Functions.h tandem/Functions.cpp)
