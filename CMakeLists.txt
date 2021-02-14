cmake_minimum_required(VERSION 2.8.3)
project(ros_libfacedetection)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(facedetection REQUIRED)

find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  cv_bridge
  image_transport
  image_geometry
  sensor_msgs
  message_generation
)

add_message_files(
  FILES
  Rectangle.msg
  Vector2D.msg
  FaceObject.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
    CATKIN_DEPENDS
    roscpp
    roslib
    cv_bridge
    image_transport
    image_geometry
    sensor_msgs
    message_generation
)

include_directories(
  "include"
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${OpenMP_INCLUDE_DIRS}
  ${facedetection_INCLUDE_DIRS}
)

add_compile_options(-std=c++11 -DUSE_AVX_INSTRUCTIONS=ON)
add_compile_options(-fopenmp)

# HOPENET
add_executable(facedetector_node src/facedetector_node.cpp)
add_dependencies(facedetector_node ros_libfacedetection_generate_messages_cpp)
target_link_libraries(facedetector_node ${OpenMP_LIBS} facedetection ${catkin_LIBRARIES} ${OpenCV_LIBS})
