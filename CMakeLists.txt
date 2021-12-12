### CMakeList.txt


# Make sure that a sufficiently high version of CMake is installed
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

# Find our deal.ii installation
FIND_PACKAGE(deal.II 9.1.0 REQUIRED
  # Hints on where to find this installation
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )

# Fetch a set of cached variables
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Project definition
PROJECT(${TARGET} project)

INCLUDE_DIRECTORIES(
	include
	)

ADD_EXECUTABLE(project source/main.cc)
DEAL_II_SETUP_TARGET(project)
add_custom_target(indent clang-format -i source/*cc include/*h WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
