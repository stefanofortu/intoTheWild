cmake_minimum_required(VERSION 2.8)
project( intoTheWild )
find_package( OpenCV REQUIRED )


#Bring the headers, such as Student.h into the project
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(include)
find_library (VLFEAT libvl.so ./vlfeat/bin/glnxa64/)
#MESSAGE( STATUS "VLFEAT: " ${VLFEAT} )

set(SOURCES src/main.cpp src/funzioni.cpp)
add_executable( intoTheWild ${SOURCES} )

target_link_libraries( intoTheWild ${OpenCV_LIBS} )
target_link_libraries( intoTheWild ${VLFEAT} )

