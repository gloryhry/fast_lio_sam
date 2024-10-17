# 使用althack/ros2作为基础镜像
FROM althack/ros:noetic-full

# 设置工作目录
WORKDIR /catkin_ws

# 安装必要的依赖项
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:borglab/gtsam-release-4.0 \
    && apt-get update \
    && apt-get install -y libgtsam-dev libgtsam-unstable-dev cmake

# 创建Software目录并克隆geographiclib库
RUN mkdir -p /Software \
    && cd /Software \
    && git clone https://github.com/geographiclib/geographiclib.git \
    && cd geographiclib \
    && mkdir build && cd build \
    && cmake .. \
    && make \
    && make install \ 
    && cd /Software \
    && git clone https://github.com/Livox-SDK/Livox-SDK.git \
    && cd Livox-SDK \
    && cd build && cmake .. \
    && make \
    && make install \
    && cd /Software \
    && git clone https://github.com/Livox-SDK/Livox-SDK2.git \
    && cd Livox-SDK2 \
    && mkdir build && cd build \
    && cmake .. && make -j \
    && make install \
    && cd / \
    && rm -rf /Software \
    && mkdir -p /catkin_ws/src/FAST-LIO-SAM \
    && cd /catkin_ws/src/ \
    && git clone https://github.com/Livox-SDK/livox_ros_driver.git \
    && git clone https://github.com/gloryhry/livox_ros_driver2_ros1.git

# 将本目录下所有文件复制到/catkin_ws/src/FAST-LIO-SAM下
COPY . /catkin_ws/src/FAST-LIO-SAM/

# 构建ROS包
RUN . /opt/ros/foxy/setup.sh  \
    && catkin_make livox_ros_driver_gencpp \
    && catkin_make livox_ros_driver2_gencpp \
    && catkin_make 
