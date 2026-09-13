cmake_minimum_required(VERSION 3.8)
project(comms_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # Wi-Fi link / throughput / latency
  "msg/WifiLinkStatus.msg"
  "msg/IperfResult.msg"
  "msg/PingStat.msg"
  "msg/BurstResult.msg"
  # NTP
  "msg/NtpStatus.msg"
  # Nexmon CSI
  "msg/CsiFrame.msg"
  "msg/CsiStatus.msg"
  DEPENDENCIES std_msgs builtin_interfaces
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
