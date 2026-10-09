cmake_minimum_required(VERSION 3.5)
project(g1_control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Threads REQUIRED)
find_package(g1_policy_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(unitree_api REQUIRED)
find_package(unitree_hg REQUIRED)

ament_python_install_package(${PROJECT_NAME})

add_executable(lowcmd_bridge src/lowcmd_bridge.cpp)
ament_target_dependencies(lowcmd_bridge g1_policy_msgs rclcpp std_msgs unitree_api unitree_hg)
target_link_libraries(lowcmd_bridge Threads::Threads)

add_executable(voice_announcer src/voice_announcer.cpp)
ament_target_dependencies(voice_announcer rclcpp std_msgs unitree_api)

install(TARGETS lowcmd_bridge voice_announcer DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS
  scripts/remote_inspector
  scripts/renet_policy
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(control_contracts test/test_package_skeleton.py)
endif()

ament_package()
