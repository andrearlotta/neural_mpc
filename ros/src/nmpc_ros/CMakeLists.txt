cmake_minimum_required(VERSION 3.0.2)
project(nmpc_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  vision_msgs
  std_msgs
  geometry_msgs
  nav_msgs
  message_generation
)

catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate services in the 'srv' folder
add_service_files(
   FILES
   GetTreesPoses.srv
)


## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs#   std_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  #INCLUDE_DIRS
  LIBRARIES nmpc_ros
  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs  geometry_msgs message_runtime
  #DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  nmpc_ros_node.py
  data_association_node.py
  baseline_node.py
  baseline_2_node.py
  nmpc_ros_node_decay.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

