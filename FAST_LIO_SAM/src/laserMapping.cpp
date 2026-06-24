// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <filesystem>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/crop_box.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_conversions/pcl_conversions.h>

// gstam
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

// gnss
#include "GNSS_Processing.hpp"
#include "sensor_msgs/NavSatFix.h"

// slam service
#include "fast_lio_sam/nav_function.h"

// save data in kitti format
#include <sstream>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>

// using namespace gtsam;

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool runtime_pos_log = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0}; //残差，点到面距离平方和
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0;
bool point_selected_surf[100000] = {0}; // 是否为平面特征点
bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<vector<int>> pointSearchInd_surf;
vector<BoxPointType> cub_needrm; // ikd-tree中，地图需要移除的包围盒序列
vector<PointVector> Nearest_Points;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
deque<double> time_buffer;               // 记录lidar时间
deque<PointCloudXYZI::Ptr> lidar_buffer; //记录特征提取或间隔采样后的lidar（特征）数据
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());  //畸变纠正后降采样的单帧点云，lidar系
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI()); //畸变纠正后降采样的单帧点云，w系
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1)); //特征点在地图中对应点的，局部平面参数,w系
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1)); //对应点法相量？
PointCloudXYZI::Ptr _featsArray;                                  // ikd-tree中，map需要移除的点云
PointCloudXYZI::Ptr current_ground_cloud(new PointCloudXYZI());    // 当前帧地面点云 (lidar系)
PointCloudXYZI::Ptr current_noground_cloud(new PointCloudXYZI());  // 当前帧非地面点云 (lidar系)

pcl::VoxelGrid<PointType> downSizeFilterSurf; //单帧内降采样使用voxel grid
pcl::VoxelGrid<PointType> downSizeFilterMap;  //未使用

KD_TREE ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d); // T lidar to imu (imu = r * lidar + t)
M3D Lidar_R_wrt_IMU(Eye3d);  // R lidar to imu (imu = r * lidar + t)

// 新增：传感器到 base_link 的外参
V3D Imu_T_wrt_Baselink(Zero3d);     // IMU 到 base_link 的平移
M3D Imu_R_wrt_Baselink(Eye3d);      // IMU 到 base_link 的旋转
V3D Lidar_T_wrt_Baselink(Zero3d);   // Lidar 到 base_link 的平移
M3D Lidar_R_wrt_Baselink(Eye3d);    // Lidar 到 base_link 的旋转
bool enable_input_transform = false; // 是否启用输入坐标转换

// 地面分割参数
bool enable_ground_seg = false;
float ground_distance = 3.75f;           // XY范围：候选地面点范围(m)
float clip_hight = 0.3f;                 // 地面以上裁剪高度(m)
float ground_distancethreshold = 0.2f;   // RANSAC平面拟合距离阈值(m)
float base_link_hight = 0.5f;            // base_link原点距真实地面高度(m)
float min_distance_f = 0.05f;            // 前方虚拟地面点范围(m)
float min_distance_b = 0.5f;             // 后方虚拟地面点范围(m)
float min_distance_l = 0.25f;            // 左方虚拟地面点范围(m)
float min_distance_r = 0.25f;            // 右方虚拟地面点范围(m)
int   extra_ground_filter_num = 2;       // 降采样间隔

// 地面分割Publisher
ros::Publisher pubGroundCloud;
ros::Publisher pubNoGroundCloud;

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf; // 状态，噪声维度，输入
state_ikfom state_point;
vect3 pos_lid; // world系下lidar坐标

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
ros::Publisher pubLidarOdomAftMapped;
ros::Publisher pubImuOdomAftMapped;
ros::Publisher pubLidarLinkPath;
ros::Publisher pubImuLinkPath;
ros::Publisher pubLidarOptLinkPath;
ros::Publisher pubImuOptLinkPath;
ros::Publisher pubCurrentVelocity;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

nav_msgs::Path lidarLinkPath;
nav_msgs::Path imuLinkPath;
nav_msgs::Path lidarLinkGlobalPath;
nav_msgs::Path imuLinkGlobalPath;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

/*back end*/
vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames; // 历史所有关键帧的角点集合（降采样）
vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;   // 历史所有关键帧的平面点集合（降采样）
vector<pcl::PointCloud<PointType>::Ptr> groundCloudKeyFrames;    // 地面点云关键帧 (lidar系)
vector<pcl::PointCloud<PointType>::Ptr> nogroundCloudKeyFrames;  // 非地面点云关键帧 (lidar系)

pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D(new pcl::PointCloud<PointType>());         // 历史关键帧位姿（位置）
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); // 历史关键帧位姿
pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>());

pcl::PointCloud<PointTypePose>::Ptr fastlio_unoptimized_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); //  存储fastlio 未优化的位姿
pcl::PointCloud<PointTypePose>::Ptr gnss_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); //  gnss 轨迹

// voxel filter paprams
float odometrySurfLeafSize;
float mappingCornerLeafSize;
float mappingSurfLeafSize;

float z_tollerance;
float rotation_tollerance;

// CPU Params
int numberOfCores = 4;
double mappingProcessInterval;

/*loop clousre*/
bool startFlag = true;
bool loopClosureEnableFlag;
float loopClosureFrequency; //   回环检测频率
int surroundingKeyframeSize;
float historyKeyframeSearchRadius;   // 回环检测 radius kdtree搜索半径
float historyKeyframeSearchTimeDiff; //  帧间时间阈值
int historyKeyframeSearchNum;        //   回环时多少个keyframe拼成submap
float historyKeyframeFitnessScore;   // icp 匹配阈值
bool potentialLoopFlag = false;

ros::Publisher pubHistoryKeyFrames; //  发布 loop history keyframe submap
ros::Publisher pubIcpKeyFrames;
ros::Publisher pubRecentKeyFrames;
ros::Publisher pubRecentKeyFrame;
ros::Publisher pubCloudRegisteredRaw;
ros::Publisher pubLoopConstraintEdge;

bool aLoopIsClosed = false;
map<int, int> loopIndexContainer; // from new to old
vector<pair<int, int>> loopIndexQueue;
vector<gtsam::Pose3> loopPoseQueue;
vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
deque<std_msgs::Float64MultiArray> loopInfoVec;

nav_msgs::Path globalPath;

// 局部关键帧构建的map点云，对应kdtree，用于scan-to-map找相邻点
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses(new pcl::KdTreeFLANN<PointType>());

// 降采样
pcl::VoxelGrid<PointType> downSizeFilterCorner;
// pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterICP;
pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

float transformTobeMapped[6]; //  当前帧的位姿(world系下)

std::mutex mtx;
std::mutex mtxLoopInfo;

// Surrounding map
float surroundingkeyframeAddingDistThreshold;  //  判断是否为关键帧的距离阈值
float surroundingkeyframeAddingAngleThreshold; //  判断是否为关键帧的角度阈值
float surroundingKeyframeDensity;
float surroundingKeyframeSearchRadius;

// gtsam
gtsam::NonlinearFactorGraph gtSAMgraph;
gtsam::Values initialEstimate;
gtsam::Values optimizedEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;
Eigen::MatrixXd poseCovariance;

ros::Publisher pubLaserCloudSurround;
ros::Publisher pubOptimizedGlobalMap ;           //   发布最后优化的地图

bool    recontructKdTree = false;
int updateKdtreeCount = 0 ;        //  每100次更新一次
bool visulize_IkdtreeMap = false;            //  visual iktree submap

// gnss
double last_timestamp_gnss = -1.0 ;
deque<nav_msgs::Odometry> gnss_buffer;
geometry_msgs::PoseStamped msg_gnss_pose;
string gnss_topic ;
bool useImuHeadingInitialization;   
bool useGpsElevation;             //  是否使用gps高层优化
float gpsCovThreshold;          //   gps方向角和高度差的协方差阈值
float gpsCovThreshold_Z;        //   gps Z(高程)方向协方差阈值
float poseCovThreshold;       //  位姿协方差阈值  from isam2

M3D Gnss_R_wrt_Lidar(Eye3d) ;         // gnss  与 imu 的外参
V3D Gnss_T_wrt_Lidar(Zero3d);
bool gnss_inited = false ;                        //  是否完成gnss初始化
bool gnss_extrinsic_calibrated = false;           //  是否完成GNSS-LiDAR外参标定
std::vector<std::pair<double, Eigen::Vector3d>> raw_gnss_enu_queue;  //  原始GNSS ENU位置(时间戳, 位置)
int gnss_calib_min_match;                         //  最少匹配点对数
double gnss_calib_max_time_diff;                  //  最大时间匹配容差
shared_ptr<GnssProcess> p_gnss(new GnssProcess());
GnssProcess gnss_data;
ros::Publisher pubGnssPath ;
nav_msgs::Path gps_path ;
vector<double>       extrinT_Gnss2Lidar(3, 0.0);
vector<double>       extrinR_Gnss2Lidar(9, 0.0);


// global map visualization radius
float globalMapVisualizationSearchRadius;
float globalMapVisualizationPoseDensity;
float globalMapVisualizationLeafSize;

// slam service
ros::ServiceServer slam_service;
ros::Publisher pub_slam_state;
bool show_globalmap_flag = false;
int slam_mode = 0; // 0: mapping, 1: localization (对齐 digitaltwins-x-nav)
int last_saved_keyframe_index = -1;  // 增量保存追踪

// ---- Localization 初始化状态机 (对齐 digitaltwins-x-nav) ----
int initializedFlag = 0;                   // 0:等待初始位姿, 1:ICP匹配中, 2:初始化完成
bool hand_init_state = false;              // 是否通过 /initialpose 手动初始化
bool auto_localization_enableflag = false; // 启用GNSS自动初始化
float initframe_FitnessScore = 0.1;        // ICP匹配分数阈值
float localization_search_radius = 5.0;    // GNSS/手动初始化时的搜索半径(m)
float localization_localmap_leafsize = 0.5;// 局部地图降采样leaf size
float localization_icp_correspondence_dist = 100.0; // ICP对应点最大距离
int   localization_icp_max_iters = 100;    // ICP最大迭代次数
float safe_speed = 15.0;                   // localization模式下最大允许速度(m/s)
int   localization_flag_vel = 0;           // 初始化完成后置1, Run()中重置速度基准

// GNSS-关键帧关联 (用于auto-localization)
struct GNSSKeyframeLink {
    double gnss_x, gnss_y, gnss_z;
    int keyframe_index;
};
std::vector<GNSSKeyframeLink> gnss_keyframe_links;
pcl::KdTreeFLANN<PointType>::Ptr kdtreeGNSSKeyPoses(new pcl::KdTreeFLANN<PointType>());
pcl::PointCloud<PointType>::Ptr gnssCloudKeyPoses3D(new pcl::PointCloud<PointType>());

// 局部地图 (用于ICP初始化)
pcl::PointCloud<PointType>::Ptr localizationLocalMap(new pcl::PointCloud<PointType>());
int locMapCenterKfIdx = -1;               // 局部地图中心关键帧索引

// 速度安全检测基准
Eigen::Vector3d last_safe_position = Eigen::Vector3d::Zero();
double last_safety_check_time = -1.0;


// 获取ros namespace
std::string rosNamespace;

// Frame IDs (可配置的坐标系名称)
std::string odometryFrame;
std::string base_linkFrame;
std::string lidar_linkFrame;
std::string imu_linkFrame;
std::string mapFrame;

/**
 * 更新里程计轨迹
 */
void updatePath(const PointTypePose &pose_in)
{
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);

    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose.position.x =  pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z =  pose_in.z;
    tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalPath.poses.push_back(pose_stamped);
}

/**
 * 对点云cloudIn进行变换transformIn，返回结果点云， 修改liosam, 考虑到外参的表示
 */
pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);
    
   // 注意：lio_sam 中的姿态用的euler表示，而fastlio存的姿态角是旋转矢量。而 pcl::getTransformation是将euler_angle 转换到rotation_matrix 不合适，注释
  // Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    Eigen::Isometry3d T_b_lidar(state_point.offset_R_L_I  );       //  获取  body2lidar  外参
    T_b_lidar.pretranslate(state_point.offset_T_L_I);        

    Eigen::Affine3f T_w_b_ = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    Eigen::Isometry3d T_w_b ;          //   world2body  
    T_w_b.matrix() = T_w_b_.matrix().cast<double>();

    Eigen::Isometry3d  T_w_lidar  =  T_w_b * T_b_lidar  ;           //  T_w_lidar  转换矩阵

    Eigen::Isometry3d transCur = T_w_lidar;        

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
        cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
        cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
}

/**
 * 位姿格式变换
 */
gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                        gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
}

/**
 * 位姿格式变换
 */
gtsam::Pose3 trans2gtsamPose(float transformIn[])
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                        gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}

/**
 * Eigen格式的位姿变换
 */
Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
{
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

/**
 * Eigen格式的位姿变换
 */
Eigen::Affine3f trans2Affine3f(float transformIn[])
{
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
}

/**
 * 位姿格式变换
 */
PointTypePose trans2PointTypePose(float transformIn[])
{
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw = transformIn[2];
    return thisPose6D;
}

/**
 * 发布thisCloud，返回thisCloud对应msg格式
 */
sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

/**
 * 点到坐标系原点距离
 */
float pointDistance(PointType p)
{
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

/**
 * 两点之间距离
 */
float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

/**
 * 初始化
 */
void allocateMemory()
{
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

    laserCloudOri.reset(new pcl::PointCloud<PointType>());

    kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

    for (int i = 0; i < 6; ++i)
    {
        transformTobeMapped[i] = 0;
    }
}

//  eulerAngle 2 Quaterniond
Eigen::Quaterniond  EulerToQuat(float roll_, float pitch_, float yaw_)
{
    Eigen::Quaterniond q ;            //   四元数 q 和 -q 是相等的
    Eigen::AngleAxisd roll(double(roll_), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch(double(pitch_), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw(double(yaw_), Eigen::Vector3d::UnitZ());
    q = yaw * pitch * roll ;
    q.normalize();
    return q ;
}

// 将更新的pose赋值到 transformTobeMapped
void getCurPose(state_ikfom cur_state)
{
    //  欧拉角是没有群的性质，所以从SO3还是一般的rotation matrix 转换过来的结果一样
    Eigen::Vector3d eulerAngle = cur_state.rot.matrix().eulerAngles(2,1,0);        //  yaw pitch roll  单位：弧度
    // V3D eulerAngle  =  SO3ToEuler(cur_state.rot)/57.3 ;     //   fastlio 自带  roll pitch yaw  单位: 度，旋转顺序 zyx

    // transformTobeMapped[0] = eulerAngle(0);                //  roll     使用 SO3ToEuler 方法时，顺序是 rpy
    // transformTobeMapped[1] = eulerAngle(1);                //  pitch
    // transformTobeMapped[2] = eulerAngle(2);                //  yaw
    
    transformTobeMapped[0] = eulerAngle(2);                //  roll  使用 eulerAngles(2,1,0) 方法时，顺序是 ypr
    transformTobeMapped[1] = eulerAngle(1);                //  pitch
    transformTobeMapped[2] = eulerAngle(0);                //  yaw
    transformTobeMapped[3] = cur_state.pos(0);          //  x
    transformTobeMapped[4] = cur_state.pos(1);          //   y
    transformTobeMapped[5] = cur_state.pos(2);          // z
}

/**
 * rviz展示闭环边
 */
void visualizeLoopClosure()
{
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳

    if (loopIndexContainer.empty())
        return;

    visualization_msgs::MarkerArray markerArray;
    // 闭环顶点
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = odometryFrame;
    markerNode.header.stamp = timeLaserInfoStamp;
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3;
    markerNode.scale.y = 0.3;
    markerNode.scale.z = 0.3;
    markerNode.color.r = 0;
    markerNode.color.g = 0.8;
    markerNode.color.b = 1;
    markerNode.color.a = 1;
    // 闭环边
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = odometryFrame;
    markerEdge.header.stamp = timeLaserInfoStamp;
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9;
    markerEdge.color.g = 0.9;
    markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    // 遍历闭环
    for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
    {
        int key_cur = it->first;
        int key_pre = it->second;
        geometry_msgs::Point p;
        p.x = copy_cloudKeyPoses6D->points[key_cur].x;
        p.y = copy_cloudKeyPoses6D->points[key_cur].y;
        p.z = copy_cloudKeyPoses6D->points[key_cur].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = copy_cloudKeyPoses6D->points[key_pre].x;
        p.y = copy_cloudKeyPoses6D->points[key_pre].y;
        p.z = copy_cloudKeyPoses6D->points[key_pre].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pubLoopConstraintEdge.publish(markerArray);
}

/**
 * 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
 */
bool saveFrame()
{
    if (cloudKeyPoses3D->points.empty())
        return true;

    // 前一帧位姿
    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
    // 当前帧位姿
    Eigen::Affine3f transFinal = trans2Affine3f(transformTobeMapped);
    // Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
    //                                                     transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
                    
    // 位姿变换增量
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw); //  获取上一帧 相对 当前帧的 位姿

    // 旋转和平移量都较小，当前帧不设为关键帧
    if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
        abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
        abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
        sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
        return false;
    return true;
}

/**
 * 添加激光里程计因子
 */
void addOdomFactor()
{
    if (cloudKeyPoses3D->points.empty())
    {
        // 第一帧初始化先验因子
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) <<1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished()); // rad*rad, meter*meter   // indoor 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12    //  1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8
        gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
        // 变量节点设置初始值
        initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
    }
    else
    {
        // 添加激光里程计因子
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back()); /// pre
        gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);                   // cur
        // 参数：前一帧id，当前帧id，前一帧与当前帧的位姿变换（作为观测值），噪声协方差
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
        // 变量节点设置初始值
        initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
    }
}

/**
 * 添加闭环因子
 */
void addLoopFactor()
{
    if (loopIndexQueue.empty())
        return;

    // 闭环队列
    for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
    {
        // 闭环边对应两帧的索引
        int indexFrom = loopIndexQueue[i].first; //   cur
        int indexTo = loopIndexQueue[i].second;  //    pre
        // 闭环边的位姿变换
        gtsam::Pose3 poseBetween = loopPoseQueue[i];
        gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();
    aLoopIsClosed = true;
}


/**
 * Umeyama 轨迹对齐 (Z轴约束): 仅求解绕Z轴的旋转(yaw) + 3D平移
 * dst = R_z * src + t,  其中 R_z 只有 yaw 分量
 */
bool umeyamaAlignment(const std::vector<Eigen::Vector3d>& src,
                      const std::vector<Eigen::Vector3d>& dst,
                      Eigen::Matrix3d& R, Eigen::Vector3d& t)
{
    if (src.size() < 3 || src.size() != dst.size())
        return false;

    int n = src.size();

    // 计算 2D 质心 (XY 平面)
    Eigen::Vector2d src_centroid_2d = Eigen::Vector2d::Zero();
    Eigen::Vector2d dst_centroid_2d = Eigen::Vector2d::Zero();
    double src_z_sum = 0.0, dst_z_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        src_centroid_2d += src[i].head<2>();
        dst_centroid_2d += dst[i].head<2>();
        src_z_sum += src[i].z();
        dst_z_sum += dst[i].z();
    }
    src_centroid_2d /= n;
    dst_centroid_2d /= n;

    // 构建 2x2 协方差矩阵 H (仅 XY 平面)
    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
    for (int i = 0; i < n; ++i) {
        Eigen::Vector2d src_d = src[i].head<2>() - src_centroid_2d;
        Eigen::Vector2d dst_d = dst[i].head<2>() - dst_centroid_2d;
        H += src_d * dst_d.transpose();
    }

    // SVD 分解 (2x2)
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d U = svd.matrixU();
    Eigen::Matrix2d V = svd.matrixV();

    // 计算 2D 旋转矩阵 (带反射修正)
    Eigen::Matrix2d R_2d = V * U.transpose();
    if (R_2d.determinant() < 0) {
        Eigen::Matrix2d V_corrected = V;
        V_corrected.col(1) *= -1.0;
        R_2d = V_corrected * U.transpose();
    }

    // 由 R_2d 构造 R_z (绕Z轴旋转)
    R = Eigen::Matrix3d::Identity();
    R.block<2,2>(0,0) = R_2d;

    // 平移: XY 由 2D 对齐, Z 独立 (仅平移不耦合旋转)
    t.head<2>() = dst_centroid_2d - R_2d * src_centroid_2d;
    t.z() = dst_z_sum / n - src_z_sum / n;

    return true;
}

/**
 * 使用 Umeyama 对齐 GNSS ENU 轨迹与 LiDAR SLAM 轨迹, 标定外参
 */
bool tryCalibrateExtrinsic()
{
    ROS_WARN("Use Umeyama to Align GNSS ENU and Lidar SLAM");
    if (fastlio_unoptimized_cloudKeyPoses6D->points.empty() || raw_gnss_enu_queue.empty())
        return false;

    // 按时间戳匹配 GNSS 和 LiDAR 位置
    std::vector<Eigen::Vector3d> lidar_positions;
    std::vector<Eigen::Vector3d> gnss_positions;

    for (const auto& gnss_pt : raw_gnss_enu_queue) {
        double gnss_time = gnss_pt.first;
        double min_diff = gnss_calib_max_time_diff;
        int best_idx = -1;

        for (size_t i = 0; i < fastlio_unoptimized_cloudKeyPoses6D->points.size(); ++i) {
            double diff = std::abs(fastlio_unoptimized_cloudKeyPoses6D->points[i].time - gnss_time);
            if (diff < min_diff) {
                min_diff = diff;
                best_idx = i;
            }
        }

        if (best_idx >= 0) {
            const auto& lidar_pt = fastlio_unoptimized_cloudKeyPoses6D->points[best_idx];
            lidar_positions.push_back(Eigen::Vector3d(lidar_pt.x, lidar_pt.y, lidar_pt.z));
            gnss_positions.push_back(gnss_pt.second);
        }
    }

    if (lidar_positions.size() < (size_t)gnss_calib_min_match) {
        ROS_INFO("GNSS calibration: %zu / %d matched pairs, waiting for more...",
                 lidar_positions.size(), gnss_calib_min_match);
        return false;
    }

    // Umeyama 对齐: P_enu = R_align * P_lidar + t_align
    Eigen::Matrix3d R_align;
    Eigen::Vector3d t_align;
    if (!umeyamaAlignment(lidar_positions, gnss_positions, R_align, t_align)) {
        ROS_WARN("Umeyama alignment failed");
        return false;
    }

    // 转换为外参: P_lidar = Gnss_R_wrt_Lidar * P_enu + Gnss_T_wrt_Lidar
    // R_align 将 LiDAR -> ENU, 所以 Gnss_R_wrt_Lidar = R_align^T
    // t_align 将 LiDAR 原点表示在 ENU 中, 所以 Gnss_T_wrt_Lidar = -R_align^T * t_align
    Gnss_R_wrt_Lidar = R_align.transpose();
    Gnss_T_wrt_Lidar = -R_align.transpose() * t_align;

    gnss_extrinsic_calibrated = true;

    ROS_INFO("GNSS-LiDAR extrinsic calibrated via Umeyama (%zu pairs):", lidar_positions.size());
    ROS_INFO("  R_gnss2lidar:\n%f %f %f\n%f %f %f\n%f %f %f",
             Gnss_R_wrt_Lidar(0,0), Gnss_R_wrt_Lidar(0,1), Gnss_R_wrt_Lidar(0,2),
             Gnss_R_wrt_Lidar(1,0), Gnss_R_wrt_Lidar(1,1), Gnss_R_wrt_Lidar(1,2),
             Gnss_R_wrt_Lidar(2,0), Gnss_R_wrt_Lidar(2,1), Gnss_R_wrt_Lidar(2,2));
    ROS_INFO("  t_gnss2lidar: %f %f %f", Gnss_T_wrt_Lidar(0), Gnss_T_wrt_Lidar(1), Gnss_T_wrt_Lidar(2));

    raw_gnss_enu_queue.clear();  // 释放不再需要的数据
    return true;
}

/**
 * 添加GPS因子
*/
void addGPSFactor()
{
    if (gnss_buffer.empty())
        return;
    // 如果没有关键帧，或者首尾关键帧距离小于5m，不添加gps因子
    if (cloudKeyPoses3D->points.empty())
        return;
    else
    {
        if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
            return;
    }
    // 位姿协方差很小，没必要加入GPS数据进行校正
    // if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
    //     return;
    static PointType lastGPSPoint;      // 最新的gps数据
    while (!gnss_buffer.empty())
    {
        // 删除当前帧0.2s之前的里程计
        if (gnss_buffer.front().header.stamp.toSec() < lidar_end_time - 0.1)
        {
            gnss_buffer.pop_front();
        }
        // 超过当前帧0.2s之后，退出
        else if (gnss_buffer.front().header.stamp.toSec() > lidar_end_time + 0.1)
        {
            break;
        }
        else
        {
            nav_msgs::Odometry thisGPS = gnss_buffer.front();
            gnss_buffer.pop_front();
            // GPS噪声协方差太大，不能用
            float noise_x = thisGPS.pose.covariance[0];         //  x 方向的协方差
            float noise_y = thisGPS.pose.covariance[7];
            float noise_z = thisGPS.pose.covariance[14];      //   z(高层)方向的协方差
            if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                continue;
            // GPS里程计位置
            float gps_x = thisGPS.pose.pose.position.x;
            float gps_y = thisGPS.pose.pose.position.y;
            float gps_z = thisGPS.pose.pose.position.z;
            if (!useGpsElevation || noise_z > gpsCovThreshold_Z) //  是否使用gps的高度
            {
                gps_z = transformTobeMapped[5];
                noise_z = 0.01;
            }

            // (0,0,0)无效数据
            if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                continue;
            
            // 应用外参变换: 将 GNSS ENU 坐标转到 LiDAR 系
            if (gnss_extrinsic_calibrated) {
                Eigen::Vector3d P_enu(gps_x, gps_y, gps_z);
                Eigen::Vector3d P_lidar = Gnss_R_wrt_Lidar * P_enu + Gnss_T_wrt_Lidar;
                gps_x = P_lidar.x();
                gps_y = P_lidar.y();
                gps_z = P_lidar.z();
            
                // 每隔5m添加一个GPS里程计
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;
                // 添加GPS因子
                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);
                aLoopIsClosed = true;
                ROS_INFO("GPS Factor Added");
                break;
            }
            else{
                break;
            }
        }
    }
}

void saveKeyFramesAndFactor()
{
    //  计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
    if (saveFrame() == false)
        return;
    // 激光里程计因子(from fast-lio),  输入的是frame_relative pose  帧间位姿(body 系下)
    addOdomFactor();
    // GPS因子 (UTM -> WGS84)
    // 若未标定外参，尝试通过 Umeyama 对齐 GPS 与 LiDAR 轨迹
    if (!gnss_extrinsic_calibrated) {
        tryCalibrateExtrinsic();
    }
    addGPSFactor();
    // 闭环因子 (rs-loop-detect)  基于欧氏距离的检测
    addLoopFactor();
    // 执行优化
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    if (aLoopIsClosed == true) // 有回环因子，多update几次
    {
        isam->update();
        isam->update();
        // isam->update();
        // isam->update();
        // isam->update();
    }
    // update之后要清空一下保存的因子图，注：历史数据不会清掉，ISAM保存起来了
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    PointType thisPose3D;
    PointTypePose thisPose6D;
    gtsam::Pose3 latestEstimate;

    // 优化结果
    isamCurrentEstimate = isam->calculateBestEstimate();
    // 当前帧位姿结果
    latestEstimate = isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size() - 1);

    // cloudKeyPoses3D加入当前帧位置
    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    // 索引
    thisPose3D.intensity = cloudKeyPoses3D->size(); //  使用intensity作为该帧点云的index
    cloudKeyPoses3D->push_back(thisPose3D);         //  新关键帧帧放入队列中

    // cloudKeyPoses6D加入当前帧位姿
    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity;
    thisPose6D.roll = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw = latestEstimate.rotation().yaw();
    thisPose6D.time = lidar_end_time;
    cloudKeyPoses6D->push_back(thisPose6D);

    // 位姿协方差
    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

    // ESKF状态和方差  更新
    state_ikfom state_updated = kf.get_x(); //  获取cur_pose (还没修正)
    Eigen::Vector3d pos(latestEstimate.translation().x(), latestEstimate.translation().y(), latestEstimate.translation().z());
    Eigen::Quaterniond q = EulerToQuat(latestEstimate.rotation().roll(), latestEstimate.rotation().pitch(), latestEstimate.rotation().yaw());

    //  更新状态量
    state_updated.pos = pos;
    state_updated.rot =  q;
    state_point = state_updated; // 对state_point进行更新，state_point可视化用到
    // if(aLoopIsClosed == true )
    kf.change_x(state_updated);  //  对cur_pose 进行isam2优化后的修正

    // TODO:  P的修正有待考察，按照yanliangwang的做法，修改了p，会跑飞
    // esekfom::esekf<state_ikfom, 12, input_ikfom>::cov P_updated = kf.get_P(); // 获取当前的状态估计的协方差矩阵
    // P_updated.setIdentity();
    // P_updated(6, 6) = P_updated(7, 7) = P_updated(8, 8) = 0.00001;
    // P_updated(9, 9) = P_updated(10, 10) = P_updated(11, 11) = 0.00001;
    // P_updated(15, 15) = P_updated(16, 16) = P_updated(17, 17) = 0.0001;
    // P_updated(18, 18) = P_updated(19, 19) = P_updated(20, 20) = 0.001;
    // P_updated(21, 21) = P_updated(22, 22) = 0.00001;
    // kf.change_P(P_updated);

    // 当前帧激光角点、平面点，降采样集合
    // pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
    // pcl::copyPointCloud(*feats_undistort,  *thisCornerKeyFrame);
    pcl::copyPointCloud(*feats_undistort, *thisSurfKeyFrame); // 存储关键帧,没有降采样的点云

    // 保存特征点降采样集合
    // cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);

    // 保存当前帧地面/非地面点云关键帧
    if (enable_ground_seg)
    {
        pcl::PointCloud<PointType>::Ptr thisGroundKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisNoGroundKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*current_ground_cloud, *thisGroundKeyFrame);
        pcl::copyPointCloud(*current_noground_cloud, *thisNoGroundKeyFrame);
        groundCloudKeyFrames.push_back(thisGroundKeyFrame);
        nogroundCloudKeyFrames.push_back(thisNoGroundKeyFrame);
    }

    updatePath(thisPose6D); //  可视化update后的path
}

void recontructIKdTree(){
    if(recontructKdTree  &&  updateKdtreeCount >  0){
        /*** if path is too large, the rvis will crash ***/
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMapPoses(new pcl::KdTreeFLANN<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kdtree查找最近一帧关键帧相邻的关键帧集合
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        mtx.lock();
        kdtreeGlobalMapPoses->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMapPoses->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            subMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);     //  subMap的pose集合
        // 降采样
        pcl::VoxelGrid<PointType> downSizeFilterSubMapKeyPoses;
        downSizeFilterSubMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterSubMapKeyPoses.setInputCloud(subMapKeyPoses);
        downSizeFilterSubMapKeyPoses.filter(*subMapKeyPosesDS);         //  subMap poses  downsample
        // 提取局部相邻关键帧对应的特征点云
        for (int i = 0; i < (int)subMapKeyPosesDS->size(); ++i)
        {
            // 距离过大
            if (pointDistance(subMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                    continue;
            int thisKeyInd = (int)subMapKeyPosesDS->points[i].intensity;
            // *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *subMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]); //  fast_lio only use  surfCloud
        }
        // 降采样，发布
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(subMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*subMapKeyFramesDS);

        std::cout << "subMapKeyFramesDS sizes  =  "   << subMapKeyFramesDS->points.size()  << std::endl;
        
        ikdtree.reconstruct(subMapKeyFramesDS->points);
        updateKdtreeCount = 0;
        ROS_INFO("Reconstructed  ikdtree ");
        int featsFromMapNum = ikdtree.validnum();
        kdtree_size_st = ikdtree.size();
        std::cout << "featsFromMapNum  =  "   << featsFromMapNum   <<  "\t" << " kdtree_size_st   =  "  <<  kdtree_size_st  << std::endl;
    }
        updateKdtreeCount ++ ; 
}

// Forward declaration for helper used in correctPoses()
inline void transformBaseToSensor(
    const Eigen::Vector3d& t_odom_base,
    const Eigen::Quaterniond& q_odom_base,
    const M3D& R_sensor_base,
    const V3D& T_sensor_base,
    Eigen::Vector3d& t_out,
    Eigen::Quaterniond& q_out);

/**
 * 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
 */
void correctPoses()
{
    if (cloudKeyPoses3D->points.empty())
        return;

    if (aLoopIsClosed == true)
    {
        // 清空里程计轨迹
        globalPath.poses.clear();
        if (enable_input_transform)
        {
            lidarLinkGlobalPath.poses.clear();
            imuLinkGlobalPath.poses.clear();
        }
        // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses; ++i)
        {
            cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
            cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
            cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();

            cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
            cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
            cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
            cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

            // 更新里程计轨迹
            updatePath(cloudKeyPoses6D->points[i]);

            // Build sensor-frame optimized paths
            if (enable_input_transform)
            {
                geometry_msgs::PoseStamped pose_stamped;
                pose_stamped.header.stamp = ros::Time().fromSec(cloudKeyPoses6D->points[i].time);
                pose_stamped.header.frame_id = odometryFrame;
                pose_stamped.pose.position.x = cloudKeyPoses6D->points[i].x;
                pose_stamped.pose.position.y = cloudKeyPoses6D->points[i].y;
                pose_stamped.pose.position.z = cloudKeyPoses6D->points[i].z;
                tf::Quaternion q_base = tf::createQuaternionFromRPY(
                    cloudKeyPoses6D->points[i].roll,
                    cloudKeyPoses6D->points[i].pitch,
                    cloudKeyPoses6D->points[i].yaw);

                Eigen::Vector3d t_base(pose_stamped.pose.position.x,
                                       pose_stamped.pose.position.y,
                                       pose_stamped.pose.position.z);
                Eigen::Quaterniond q_odom_base(q_base.w(), q_base.x(), q_base.y(), q_base.z());

                // lidar_link optimized pose
                {
                    geometry_msgs::PoseStamped lidar_pose = pose_stamped;
                    Eigen::Vector3d t_lidar;
                    Eigen::Quaterniond q_lidar;
                    transformBaseToSensor(t_base, q_odom_base,
                                          Lidar_R_wrt_Baselink, Lidar_T_wrt_Baselink,
                                          t_lidar, q_lidar);
                    lidar_pose.pose.position.x = t_lidar.x();
                    lidar_pose.pose.position.y = t_lidar.y();
                    lidar_pose.pose.position.z = t_lidar.z();
                    lidar_pose.pose.orientation.x = q_lidar.x();
                    lidar_pose.pose.orientation.y = q_lidar.y();
                    lidar_pose.pose.orientation.z = q_lidar.z();
                    lidar_pose.pose.orientation.w = q_lidar.w();
                    lidarLinkGlobalPath.poses.push_back(lidar_pose);
                }

                // imu_link optimized pose
                {
                    geometry_msgs::PoseStamped imu_pose = pose_stamped;
                    Eigen::Vector3d t_imu;
                    Eigen::Quaterniond q_imu;
                    transformBaseToSensor(t_base, q_odom_base,
                                          Imu_R_wrt_Baselink, Imu_T_wrt_Baselink,
                                          t_imu, q_imu);
                    imu_pose.pose.position.x = t_imu.x();
                    imu_pose.pose.position.y = t_imu.y();
                    imu_pose.pose.position.z = t_imu.z();
                    imu_pose.pose.orientation.x = q_imu.x();
                    imu_pose.pose.orientation.y = q_imu.y();
                    imu_pose.pose.orientation.z = q_imu.z();
                    imu_pose.pose.orientation.w = q_imu.w();
                    imuLinkGlobalPath.poses.push_back(imu_pose);
                }
            }
        }
        // 清空局部map， reconstruct  ikdtree submap
        recontructIKdTree();
        ROS_INFO("ISMA2 Update");
        aLoopIsClosed = false;
    }
}

//回环检测三大要素
// 1.设置最小时间差，太近没必要
// 2.控制回环的频率，避免频繁检测，每检测一次，就做一次等待
// 3.根据当前最小距离重新计算等待时间
bool detectLoopClosureDistance(int *latestID, int *closestID)
{
    // 当前关键帧帧
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1; //  当前关键帧索引
    int loopKeyPre = -1;

    // 当前帧已经添加过闭环对应关系，不再继续添加
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;
    // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
    std::vector<int> pointSearchIndLoop;                        //  候选关键帧索引
    std::vector<float> pointSearchSqDisLoop;                    //  候选关键帧距离
    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D); //  历史帧构建kdtree
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
    // 在候选关键帧集合中，找到与当前帧时间相隔较远的帧，设为候选匹配帧
    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
        int id = pointSearchIndLoop[i];
        if (abs(copy_cloudKeyPoses6D->points[id].time - lidar_end_time) > historyKeyframeSearchTimeDiff)
        {
            loopKeyPre = id;
            break;
        }
    }
    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return false;
    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    ROS_INFO("Find loop clousre frame ");
    return true;
}

/**
 * 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合，降采样
 */
void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum)
{
    // 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses6D->size();
    auto surfcloud_keyframes_size = surfCloudKeyFrames.size() ;
    for (int i = -searchNum; i <= searchNum; ++i)
    {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= cloudSize)
            continue;

        if (keyNear < 0 || keyNear >= surfcloud_keyframes_size)
            continue;

        // *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
        // 注意：cloudKeyPoses6D 存储的是 T_w_b , 而点云是lidar系下的，构建icp的submap时，需要通过外参数T_b_lidar 转换 , 参考pointBodyToWorld 的转换
        *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]); //  fast-lio 没有进行特征提取，默认点云就是surf
    }

    if (nearKeyframes->empty())
        return;

    // 降采样
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
}

void performLoopClosure()
{
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳

    if (cloudKeyPoses3D->points.empty() == true)
    {
        return;
    }

    mtx.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    mtx.unlock();

    // 当前关键帧索引，候选闭环匹配帧索引
    int loopKeyCur;
    int loopKeyPre;
    // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
    if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
    {
        return;
    }

    // 提取
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>()); //  cue keyframe
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>()); //   history keyframe submap
    {
        // 提取当前关键帧特征点集合，降采样
        loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0); //  将cur keyframe 转换到world系下
        // 提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
        loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum); //  选取historyKeyframeSearchNum个keyframe拼成submap
        // 如果特征点较少，返回
        // if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
        //     return;
        // 发布闭环匹配关键帧局部map
        if (pubHistoryKeyFrames.getNumSubscribers() != 0)
            publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
    }

    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
    icp.setMaximumIterations(20);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // scan-to-map，调用icp匹配
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    // 未收敛，或者匹配不够好
    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
        return;

    std::cout << "icp  success  " << std::endl;

    // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
    if (pubIcpKeyFrames.getNumSubscribers() != 0)
    {
        pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
        publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
    }

    // 闭环优化得到的当前关键帧与闭环关键帧之间的位姿变换
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();

    // 闭环优化前当前帧位姿
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    // 闭环优化后当前帧位姿
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw); //  获取上一帧 相对 当前帧的 位姿
    gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
    // 闭环匹配帧的位姿
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    float noiseScore = icp.getFitnessScore() ; //  loop_clousre  noise from icp
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    gtsam::noiseModel::Diagonal::shared_ptr constraintNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
    std::cout << "loopNoiseQueue   =   " << noiseScore << std::endl;

    // 添加闭环因子需要的数据
    mtx.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    mtx.unlock();

    loopIndexContainer[loopKeyCur] = loopKeyPre; //   使用hash map 存储回环对
}

//回环检测线程
void loopClosureThread()
{
    if (loopClosureEnableFlag == false)
    {
        std::cout << "loopClosureEnableFlag   ==  false " << endl;
        return;
    }

    ros::Rate rate(loopClosureFrequency); //   回环频率
    while (ros::ok() && startFlag)
    {
        rate.sleep();
        performLoopClosure();   //  回环检测
        visualizeLoopClosure(); // rviz展示闭环边
    }
}

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                            // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2));    // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // omega
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2));    // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // Acc
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));       // Bias_g
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));       // Bias_a
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a
    fprintf(fp, "\r\n");
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

//按当前body(lidar)的状态，将局部点转换到世界系下
void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    // world <-- imu <-- lidar
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po)        //  lidar2world
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void test_RGBpointBodyToWorld(PointType const *const pi, PointType *const po,Eigen::Vector3d pos, Eigen::Matrix3d rotation)        //  lidar2world
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(rotation * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    for (int i = 0; i < points_history.size(); i++)
        _featsArray->push_back(points_history[i]);
}

//根据lidar的FoV分割场景
BoxPointType LocalMap_Points; // ikd-tree中,局部地图的包围盒角点
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear(); // 清空需要移除的区域
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world); // X轴分界点转换到w系下
    V3D pos_LiD = pos_lid;                               // global系lidar位置

    //初始化局部地图包围盒角点，以为w系下lidar位置为中心
    if (!Localmap_Initialized)
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    float dist_to_map_edge[3][2]; //各个方向与局部地图边界的距离
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);

        //与某个方向上的边界距离太小，标记需要移除need_move
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }

    //不需要移除则直接返回
    if (!need_move)
        return;

    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points; // 新的局部地图角点
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        //与包围盒最小值角点距离
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints); // 移除较远包围盒
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

// ---- Localization 初始化函数前向声明 ----
void InitialPoseCallBack(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg);
void buildLocalMapFromKeyframe(int center_idx);
bool gnssAutoInit();
void icpLocalizationInit(pcl::PointCloud<PointType>::Ptr scan_lidar_frame);
void velocitySafetyCheck();

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mtx_buffer.lock();
    scan_count++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);

    // 新增：如果启用输入坐标转换，转换点云到 base_link
    if (enable_input_transform)
    {
        PointCloudXYZI::Ptr transformed_ptr(new PointCloudXYZI());
        transformed_ptr->reserve(ptr->size());

        for (size_t i = 0; i < ptr->size(); i++)
        {
            PointType p_in = ptr->points[i];
            PointType p_out;

            V3D p_lidar(p_in.x, p_in.y, p_in.z);
            V3D p_baselink = Lidar_R_wrt_Baselink * p_lidar + Lidar_T_wrt_Baselink;

            p_out.x = p_baselink.x();
            p_out.y = p_baselink.y();
            p_out.z = p_baselink.z();
            p_out.intensity = p_in.intensity;
            p_out.curvature = p_in.curvature;

            transformed_ptr->push_back(p_out);
        }

        ptr = transformed_ptr;
    }

    // 保存 lidar 系原始点云 (用于 localization ICP 初始化)
    PointCloudXYZI::Ptr ptr_lidar_frame(new PointCloudXYZI());
    pcl::copyPointCloud(*ptr, *ptr_lidar_frame);

    // ---- Localization 初始化 ----
    if (slam_mode == 1 && initializedFlag != 2)
    {
        if (initializedFlag == 0)
        {
            std_msgs::String sm;
            sm.data = "3";
            pub_slam_state.publish(sm);

            if (auto_localization_enableflag && !hand_init_state)
                gnssAutoInit();
        }

        if (initializedFlag == 1)
            icpLocalizationInit(ptr_lidar_frame);
    }

    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// ============================================================================
// Localization 初始化函数 (对齐 digitaltwins-x-nav)
// ============================================================================

/**
 * /initialpose 回调 — 手动设置初始位姿
 */
void InitialPoseCallBack(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
{
    if (slam_mode != 1 || initializedFlag != 0) return;

    double ix = msg->pose.pose.position.x;
    double iy = msg->pose.pose.position.y;
    double iz = msg->pose.pose.position.z;
    tf::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                     msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    double ir, ip, iyaw; tf::Matrix3x3(q).getRPY(ir, ip, iyaw);

    if (cloudKeyPoses3D->points.empty())
    {
        ROS_WARN("[localization] No keyframes loaded, cannot set initial pose from /initialpose");
        return;
    }

    PointType sp; sp.x = ix; sp.y = iy; sp.z = iz;
    kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);
    std::vector<int> idx; std::vector<float> dist;
    if (kdtreeSurroundingKeyPoses->radiusSearch(sp, localization_search_radius, idx, dist) == 0)
    {
        ROS_WARN("[localization] No keyframe within %.1fm of initial pose (%.1f, %.1f, %.1f)",
                 localization_search_radius, ix, iy, iz);
        return;
    }
    int best = idx[std::min_element(dist.begin(), dist.end()) - dist.begin()];
    ROS_INFO("[localization] /initialpose: nearest KF=%d (dist=%.2fm), building local map", best, sqrt(dist[best]));

    buildLocalMapFromKeyframe(best);

    // 设置 EKF 初始状态
    state_point.pos = V3D(cloudKeyPoses6D->points[best].x, cloudKeyPoses6D->points[best].y, cloudKeyPoses6D->points[best].z);
    state_point.rot = EulerToQuat(cloudKeyPoses6D->points[best].roll, cloudKeyPoses6D->points[best].pitch, cloudKeyPoses6D->points[best].yaw);
    state_point.vel.setZero();
    kf.change_x(state_point);

    hand_init_state = true;
    initializedFlag = 1;

    ROS_INFO("[localization] /initialpose accepted, entering ICP matching phase");
}

/**
 * 以 center_idx 为中心构建局部地图 (用于ICP匹配)
 */
void buildLocalMapFromKeyframe(int center_idx)
{
    localizationLocalMap->clear();
    locMapCenterKfIdx = center_idx;

    int w = 10;  // ±10 关键帧窗口
    int num_kfs = (int)surfCloudKeyFrames.size();
    for (int i = center_idx - w; i <= center_idx + w; i++)
    {
        if (i < 0 || i >= num_kfs) continue;
        *localizationLocalMap += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
    }

    pcl::PointCloud<PointType>::Ptr ds(new pcl::PointCloud<PointType>());
    pcl::VoxelGrid<PointType> vf;
    vf.setLeafSize(localization_localmap_leafsize, localization_localmap_leafsize, localization_localmap_leafsize);
    vf.setInputCloud(localizationLocalMap);
    vf.filter(*ds);
    *localizationLocalMap = *ds;

    ROS_INFO("[localization] Built local map: %zu pts around KF %d [range %d..%d]",
             localizationLocalMap->size(), center_idx,
             std::max(0, center_idx - w), std::min(num_kfs - 1, center_idx + w));
}

/**
 * GNSS 自动初始化: 用当前 GNSS 位置在预存的关键帧-GNSS关联中搜索
 */
bool gnssAutoInit()
{
    if (gnss_buffer.empty() || gnssCloudKeyPoses3D->points.empty()) return false;
    if (initializedFlag != 0) return false;

    // 取最近 lidar时间 ±0.5s 内的 GNSS 数据
    nav_msgs::Odometry latest;
    bool found = false;
    while (!gnss_buffer.empty())
    {
        double dt = gnss_buffer.front().header.stamp.toSec() - lidar_end_time;
        if (dt < -0.5) { gnss_buffer.pop_front(); continue; }
        if (dt > 0.5)  break;
        latest = gnss_buffer.front();
        gnss_buffer.pop_front();
        found = true;
    }
    if (!found || abs(latest.pose.pose.position.x) < 1e-6) return false;

    double gx = latest.pose.pose.position.x;
    double gy = latest.pose.pose.position.y;
    double gz = latest.pose.pose.position.z;

    PointType sp; sp.x = gx; sp.y = gy; sp.z = gz;
    kdtreeGNSSKeyPoses->setInputCloud(gnssCloudKeyPoses3D);
    std::vector<int> idx; std::vector<float> dist;
    if (kdtreeGNSSKeyPoses->radiusSearch(sp, localization_search_radius, idx, dist) == 0)
    {
        ROS_WARN("[localization] No GNSS-aligned keyframe within %.1fm of (%.1f, %.1f)",
                 localization_search_radius, gx, gy);
        return false;
    }
    int best = idx[std::min_element(dist.begin(), dist.end()) - dist.begin()];
    int kf_idx = (int)gnssCloudKeyPoses3D->points[best].intensity;
    ROS_INFO("[localization] GNSS auto-init: matched KF %d (dist=%.2fm)", kf_idx, sqrt(dist[best]));

    buildLocalMapFromKeyframe(kf_idx);

    state_point.pos = V3D(cloudKeyPoses6D->points[kf_idx].x, cloudKeyPoses6D->points[kf_idx].y, cloudKeyPoses6D->points[kf_idx].z);
    state_point.rot = EulerToQuat(cloudKeyPoses6D->points[kf_idx].roll, cloudKeyPoses6D->points[kf_idx].pitch, cloudKeyPoses6D->points[kf_idx].yaw);
    state_point.vel.setZero();
    kf.change_x(state_point);

    initializedFlag = 1;
    return true;
}

/**
 * ICP 匹配: 当前扫描 vs 局部地图
 */
void icpLocalizationInit(pcl::PointCloud<PointType>::Ptr scan_lidar_frame)
{
    if (localizationLocalMap->empty()) return;

    // 降采样当前扫描
    pcl::PointCloud<PointType>::Ptr scanDS(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(scan_lidar_frame);
    downSizeFilterICP.filter(*scanDS);

    // 转换到 world 系
    pcl::PointCloud<PointType>::Ptr scanW(new pcl::PointCloud<PointType>());
    scanW->resize(scanDS->size());
    for (size_t i = 0; i < scanDS->size(); i++)
        pointBodyToWorld(&scanDS->points[i], &scanW->points[i]);

    // ICP 匹配
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(localization_icp_correspondence_dist);
    icp.setMaximumIterations(localization_icp_max_iters);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);
    icp.setInputSource(scanW);
    icp.setInputTarget(localizationLocalMap);
    pcl::PointCloud<PointType>::Ptr unused(new pcl::PointCloud<PointType>());
    icp.align(*unused);

    float score = icp.getFitnessScore();

    if (icp.hasConverged() && score < initframe_FitnessScore)
    {
        // ICP 成功 — 修正 EKF 状态
        Eigen::Affine3f T_corr(icp.getFinalTransformation());
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(T_corr, x, y, z, roll, pitch, yaw);

        state_point.pos += V3D(x, y, z);
        Eigen::Quaterniond qc(Eigen::AngleAxisd((double)yaw, V3D::UnitZ()) *
                              Eigen::AngleAxisd((double)pitch, V3D::UnitY()) *
                              Eigen::AngleAxisd((double)roll, V3D::UnitX()));
        state_point.rot = state_point.rot * qc;
        state_point.vel.setZero();
        kf.change_x(state_point);

        initializedFlag = 2;
        localization_flag_vel = 1;
        last_safe_position = state_point.pos;
        last_safety_check_time = lidar_end_time;

        std_msgs::String sm;
        sm.data = "5#" + std::to_string(score);
        pub_slam_state.publish(sm);

        ROS_INFO("[localization] ICP init SUCCESS: score=%.6f < threshold=%.6f", score, initframe_FitnessScore);
    }
    else
    {
        std_msgs::String sm;
        char buf[64];
        snprintf(buf, sizeof(buf), "4#%.4f", score);
        sm.data = buf;
        pub_slam_state.publish(sm);

        ROS_INFO("[localization] ICP attempt: score=%.6f (threshold=%.6f), retrying...",
                 score, initframe_FitnessScore);
    }
}

/**
 * 速度安全检测 (localization 模式专用)
 */
void velocitySafetyCheck()
{
    if (slam_mode != 1 || localization_flag_vel != 1) return;

    double dt = lidar_end_time - last_safety_check_time;
    if (dt <= 0.0 || dt > 1.0)
    {
        last_safety_check_time = lidar_end_time;
        last_safe_position = state_point.pos;
        return;
    }

    double speed = (state_point.pos - last_safe_position).norm() / dt;

    if (speed > safe_speed)
    {
        state_point.pos = last_safe_position;
        state_point.vel.setZero();
        kf.change_x(state_point);

        std_msgs::String sm;
        sm.data = "-1#" + std::to_string(speed);
        pub_slam_state.publish(sm);

        ROS_WARN("[localization] Speed safety violation: %.2f > %.2f m/s, resetting EKF", speed, safe_speed);
    }
    else
    {
        last_safe_position = state_point.pos;
        last_safety_check_time = lidar_end_time;
    }
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false; // 标记是否已经进行了时间补偿
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu; //????
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());

    // 特征提取或间隔采样
    p_pre->process(msg, ptr);

    // 新增：如果启用输入坐标转换，转换点云到 base_link
    if (enable_input_transform)
    {
        PointCloudXYZI::Ptr transformed_ptr(new PointCloudXYZI());
        transformed_ptr->reserve(ptr->size());

        for (size_t i = 0; i < ptr->size(); i++)
        {
            PointType p_in = ptr->points[i];
            PointType p_out;

            V3D p_lidar(p_in.x, p_in.y, p_in.z);
            V3D p_baselink = Lidar_R_wrt_Baselink * p_lidar + Lidar_T_wrt_Baselink;

            p_out.x = p_baselink.x();
            p_out.y = p_baselink.y();
            p_out.z = p_baselink.z();
            p_out.intensity = p_in.intensity;
            p_out.curvature = p_in.curvature;

            transformed_ptr->push_back(p_out);
        }

        ptr = transformed_ptr;
    }

    // 保存 lidar 系原始点云 (用于 localization ICP 初始化)
    PointCloudXYZI::Ptr ptr_lidar_frame(new PointCloudXYZI());
    pcl::copyPointCloud(*ptr, *ptr_lidar_frame);

    // ---- Localization 初始化 ----
    if (slam_mode == 1 && initializedFlag != 2)
    {
        if (initializedFlag == 0)
        {
            std_msgs::String sm;
            sm.data = "3";
            pub_slam_state.publish(sm);

            if (auto_localization_enableflag && !hand_init_state)
                gnssAutoInit();
        }

        if (initializedFlag == 1)
            icpLocalizationInit(ptr_lidar_frame);
    }

    lidar_buffer.push_back(ptr); //储存处理后的lidar特征
    time_buffer.push_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    // lidar 和 imu时间差过大，且开启 时间同步, 纠正当前输入imu的时间
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        // 对输入imu时间，纠正为 时间差 + 原始时间
        msg->header.stamp =
            ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    // 新增：如果启用输入坐标转换，转换 IMU 数据到 base_link
    if (enable_input_transform)
    {
        // 转换线性加速度
        V3D acc(msg->linear_acceleration.x,
                msg->linear_acceleration.y,
                msg->linear_acceleration.z);
        acc = Imu_R_wrt_Baselink * acc;
        msg->linear_acceleration.x = acc.x();
        msg->linear_acceleration.y = acc.y();
        msg->linear_acceleration.z = acc.z();

        // 转换角速度
        V3D gyro(msg->angular_velocity.x,
                 msg->angular_velocity.y,
                 msg->angular_velocity.z);
        gyro = Imu_R_wrt_Baselink * gyro;
        msg->angular_velocity.x = gyro.x();
        msg->angular_velocity.y = gyro.y();
        msg->angular_velocity.z = gyro.z();

        // 转换姿态（四元数）
        Eigen::Quaterniond q(msg->orientation.w, msg->orientation.x,
                             msg->orientation.y, msg->orientation.z);
        Eigen::Quaterniond q_rot(Imu_R_wrt_Baselink);
        q = q_rot * q;
        msg->orientation.w = q.w();
        msg->orientation.x = q.x();
        msg->orientation.y = q.y();
        msg->orientation.z = q.z();
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp; // update imu time

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void gnss_cbk(const sensor_msgs::NavSatFixConstPtr& msg_in)
{
    //  ROS_INFO("GNSS DATA IN ");
    double timestamp = msg_in->header.stamp.toSec();

    mtx_buffer.lock();

    // 没有进行时间纠正
    if (timestamp < last_timestamp_gnss)
    {
        ROS_WARN("gnss loop back, clear buffer");
        gnss_buffer.clear();
    }

    last_timestamp_gnss = timestamp;

    // convert ROS NavSatFix to GeographicLib compatible GNSS message:
    gnss_data.time = msg_in->header.stamp.toSec();
    gnss_data.status = msg_in->status.status;
    gnss_data.service = msg_in->status.service;
    gnss_data.pose_cov[0] = msg_in->position_covariance[0];
    gnss_data.pose_cov[1] = msg_in->position_covariance[4];
    gnss_data.pose_cov[2] = msg_in->position_covariance[8];

    mtx_buffer.unlock();
   
    if(!gnss_inited){           //  初始化位置
        gnss_data.InitOriginPosition(msg_in->latitude, msg_in->longitude, msg_in->altitude) ; 
        gnss_inited = true ;
    }else{                               //   初始化完成
        gnss_data.UpdateXYZ(msg_in->latitude, msg_in->longitude, msg_in->altitude) ;             //  WGS84 -> ENU  ???  调试结果好像是 NED 北东地
        nav_msgs::Odometry gnss_data_enu ;
        // add new message to buffer:
        gnss_data_enu.header.stamp = ros::Time().fromSec(gnss_data.time);
        gnss_data_enu.pose.pose.position.x =  gnss_data.local_E ;  //gnss_data.local_E ;   北
        gnss_data_enu.pose.pose.position.y =  gnss_data.local_N ;  //gnss_data.local_N;    东
        gnss_data_enu.pose.pose.position.z =  gnss_data.local_U;  //  地

        gnss_data_enu.pose.pose.orientation.x =  geoQuat.x ;                //  gnss 的姿态不可观，所以姿态只用于可视化，取自imu
        gnss_data_enu.pose.pose.orientation.y =  geoQuat.y;
        gnss_data_enu.pose.pose.orientation.z =  geoQuat.z;
        gnss_data_enu.pose.pose.orientation.w =  geoQuat.w;

        gnss_data_enu.pose.covariance[0] = gnss_data.pose_cov[0] ;
        gnss_data_enu.pose.covariance[7] = gnss_data.pose_cov[1] ;
        gnss_data_enu.pose.covariance[14] = gnss_data.pose_cov[2] ;

        gnss_buffer.push_back(gnss_data_enu);
        
        // 存储原始 GNSS ENU 位置用于外参标定
        if (!gnss_extrinsic_calibrated) {
            raw_gnss_enu_queue.push_back(std::make_pair(gnss_data.time,
                Eigen::Vector3d(gnss_data.local_E, gnss_data.local_N, gnss_data.local_U)));
        }

        // visial gnss path in rviz:
        msg_gnss_pose.header.frame_id = odometryFrame;
        msg_gnss_pose.header.stamp = ros::Time().fromSec(gnss_data.time);
        // Eigen::Vector3d gnss_pose_ (gnss_data.local_E, gnss_data.local_N, - gnss_data.local_U); 
        // Eigen::Vector3d gnss_pose_ (gnss_data.local_N, gnss_data.local_E, - gnss_data.local_U); 
        Eigen::Matrix4d gnss_pose = Eigen::Matrix4d::Identity();

        gnss_pose(0,3) = gnss_data.local_E ;
        gnss_pose(1,3) = gnss_data.local_N ;
        gnss_pose(2,3) = gnss_data.local_U ;

        Eigen::Isometry3d gnss_to_lidar(Gnss_R_wrt_Lidar) ;
        gnss_to_lidar.pretranslate(Gnss_T_wrt_Lidar);
        gnss_pose  =  gnss_to_lidar  *  gnss_pose ;                    //  gnss 转到 lidar 系下

        msg_gnss_pose.pose.position.x = gnss_pose(0,3) ;  
        msg_gnss_pose.pose.position.y = gnss_pose(1,3) ;
        msg_gnss_pose.pose.position.z = gnss_pose(2,3) ;

        gps_path.poses.push_back(msg_gnss_pose);

        //  save_gnss path
        PointTypePose thisPose6D;  
        thisPose6D.x = msg_gnss_pose.pose.position.x ;
        thisPose6D.y = msg_gnss_pose.pose.position.y ;
        thisPose6D.z = msg_gnss_pose.pose.position.z ;
        thisPose6D.intensity = 0;
        thisPose6D.roll =0;
        thisPose6D.pitch = 0;
        thisPose6D.yaw = 0;
        thisPose6D.time = lidar_end_time;
        gnss_cloudKeyPoses6D->push_back(thisPose6D);   
    }


}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();         // lidar指针指向最旧的lidar数据
        meas.lidar_beg_time = time_buffer.front(); //记录最早时间

        //更新结束时刻的时间
        if (meas.lidar->points.size() <= 1) // time too little 时间太短，点数不足
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // 记录lidar结束时间为 起始时间 + 单帧扫描时间
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime) //最后一个点的时间 小于 单帧扫描时间的一半
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // 记录lidar结束时间为 起始时间 + 单帧扫描时间
        }
        else
        {
            scan_num++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); //结束时间设置为 起始时间 + 最后一个点的时间（相对）
            // 动态更新每帧lidar数据平均扫描时间
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec(); // 最旧IMU时间
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) //记录imu数据，imu时间小于当前帧lidar结束时间
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)
            break;
        meas.imu.push_back(imu_buffer.front()); //记录当前lidar帧内的imu数据到meas.imu
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;            //需要加入到ikd-tree中的点云
    PointVector PointNoNeedDownsample; //加入ikd-tree时，不需要降采样的点云
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);

    //根据点与所在包围盒中心点的距离，分类是否需要降采样
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point); //当前点与box中心的距离

            //判断最近点在x、y、z三个方向上，与中心的距离，判断是否加入时需要降采样
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]); //若三个方向距离都大于地图珊格半轴长，无需降采样
                continue;
            }

            //判断当前点的 NUM_MATCH_POINTS 个邻近点与包围盒中心的范围
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(points_near[readd_i], mid_point) < dist) // 如果邻近点到中心的距离 小于 当前点到中心的距离，则不需要添加当前点
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true); //加入点时需要降采样
    ikdtree.Add_Points(PointNoNeedDownsample, false);      //加入点时不需要降采样
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
void publish_frame_world(const ros::Publisher &pubLaserCloudFull)         //    将稠密点云从 imu convert to  world
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = odometryFrame;
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

}

void publish_frame_body(const ros::Publisher &pubLaserCloudFull_body)          //   发布body系(imu)下的点云
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = base_linkFrame;
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher &pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld(
        new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i],
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = odometryFrame;
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = odometryFrame;
    pubLaserCloudMap.publish(laserCloudMap);
}

template <typename T>
void set_posestamp(T &out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
}

/**
 * Transform a base_link pose to a sensor_link pose using sensor-to-base extrinsics.
 * T_odom_sensor = T_odom_base * T_base_sensor
 */
inline void transformBaseToSensor(
    const Eigen::Vector3d& t_odom_base,
    const Eigen::Quaterniond& q_odom_base,
    const M3D& R_sensor_base,
    const V3D& T_sensor_base,
    Eigen::Vector3d& t_out,
    Eigen::Quaterniond& q_out)
{
    q_out = q_odom_base * Eigen::Quaterniond(R_sensor_base);
    t_out = q_odom_base * T_sensor_base + t_odom_base;
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = odometryFrame;
    odomAftMapped.child_frame_id = base_linkFrame;
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time); // ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, odometryFrame, base_linkFrame));

    // 如果启用了 input_transform，根据 odomAftMapped 发布 lidar_link 和 imu_link 的 odometry
    if (enable_input_transform)
    {
        // T_odom_to_base: 从 odomAftMapped 获取
        Eigen::Quaterniond q_odom_base(
            odomAftMapped.pose.pose.orientation.w,
            odomAftMapped.pose.pose.orientation.x,
            odomAftMapped.pose.pose.orientation.y,
            odomAftMapped.pose.pose.orientation.z);
        Eigen::Vector3d t_odom_base(
            odomAftMapped.pose.pose.position.x,
            odomAftMapped.pose.pose.position.y,
            odomAftMapped.pose.pose.position.z);

        // T_base_lidar: Lidar_R_wrt_Baselink/Lidar_T_wrt_Baselink 本身即为 base_link 帧下 LiDAR 的位姿
        // T_odom_to_lidar = T_odom_to_base * T_base_lidar
        Eigen::Quaterniond q_base_lidar(Lidar_R_wrt_Baselink);
        Eigen::Quaterniond q_odom_lidar = q_odom_base * q_base_lidar;
        Eigen::Vector3d t_odom_lidar = q_odom_base * Lidar_T_wrt_Baselink + t_odom_base;

        nav_msgs::Odometry lidar_odom = odomAftMapped;
        lidar_odom.header.frame_id = odometryFrame;
        lidar_odom.child_frame_id = lidar_linkFrame;
        lidar_odom.pose.pose.position.x = t_odom_lidar.x();
        lidar_odom.pose.pose.position.y = t_odom_lidar.y();
        lidar_odom.pose.pose.position.z = t_odom_lidar.z();
        lidar_odom.pose.pose.orientation.x = q_odom_lidar.x();
        lidar_odom.pose.pose.orientation.y = q_odom_lidar.y();
        lidar_odom.pose.pose.orientation.z = q_odom_lidar.z();
        lidar_odom.pose.pose.orientation.w = q_odom_lidar.w();
        pubLidarOdomAftMapped.publish(lidar_odom);

        // T_base_imu: Imu_R_wrt_Baselink/Imu_T_wrt_Baselink 本身即为 base_link 帧下 IMU 的位姿
        // T_odom_to_imu = T_odom_to_base * T_base_imu
        Eigen::Quaterniond q_base_imu(Imu_R_wrt_Baselink);
        Eigen::Quaterniond q_odom_imu = q_odom_base * q_base_imu;
        Eigen::Vector3d t_odom_imu = q_odom_base * Imu_T_wrt_Baselink + t_odom_base;

        nav_msgs::Odometry imu_odom = odomAftMapped;
        imu_odom.header.frame_id = odometryFrame;
        imu_odom.child_frame_id = imu_linkFrame;
        imu_odom.pose.pose.position.x = t_odom_imu.x();
        imu_odom.pose.pose.position.y = t_odom_imu.y();
        imu_odom.pose.pose.position.z = t_odom_imu.z();
        imu_odom.pose.pose.orientation.x = q_odom_imu.x();
        imu_odom.pose.pose.orientation.y = q_odom_imu.y();
        imu_odom.pose.pose.orientation.z = q_odom_imu.z();
        imu_odom.pose.pose.orientation.w = q_odom_imu.w();
        pubImuOdomAftMapped.publish(imu_odom);
    }
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = odometryFrame;

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0)
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
        
        //  save  unoptimized pose
         V3D rot_ang(Log( state_point.rot.toRotationMatrix())); //   旋转向量
        PointTypePose thisPose6D;  
        thisPose6D.x = msg_body_pose.pose.position.x ;
        thisPose6D.y = msg_body_pose.pose.position.y ;
        thisPose6D.z = msg_body_pose.pose.position.z ;
        thisPose6D.roll = rot_ang(0) ;
        thisPose6D.pitch = rot_ang(1) ;
        thisPose6D.yaw = rot_ang(2) ;
        thisPose6D.time = lidar_end_time;
        fastlio_unoptimized_cloudKeyPoses6D->push_back(thisPose6D);

        // Publish sensor-frame unoptimized paths when input_transform is enabled
        if (enable_input_transform)
        {
            Eigen::Vector3d t_base(msg_body_pose.pose.position.x,
                                   msg_body_pose.pose.position.y,
                                   msg_body_pose.pose.position.z);
            Eigen::Quaterniond q_base(msg_body_pose.pose.orientation.w,
                                      msg_body_pose.pose.orientation.x,
                                      msg_body_pose.pose.orientation.y,
                                      msg_body_pose.pose.orientation.z);

            // lidar_link unoptimized path
            {
                geometry_msgs::PoseStamped lidar_pose;
                lidar_pose.header = msg_body_pose.header;
                Eigen::Vector3d t_lidar;
                Eigen::Quaterniond q_lidar;
                transformBaseToSensor(t_base, q_base, Lidar_R_wrt_Baselink, Lidar_T_wrt_Baselink, t_lidar, q_lidar);
                lidar_pose.pose.position.x = t_lidar.x();
                lidar_pose.pose.position.y = t_lidar.y();
                lidar_pose.pose.position.z = t_lidar.z();
                lidar_pose.pose.orientation.x = q_lidar.x();
                lidar_pose.pose.orientation.y = q_lidar.y();
                lidar_pose.pose.orientation.z = q_lidar.z();
                lidar_pose.pose.orientation.w = q_lidar.w();
                lidarLinkPath.poses.push_back(lidar_pose);
                pubLidarLinkPath.publish(lidarLinkPath);
            }

            // imu_link unoptimized path
            {
                geometry_msgs::PoseStamped imu_pose;
                imu_pose.header = msg_body_pose.header;
                Eigen::Vector3d t_imu;
                Eigen::Quaterniond q_imu;
                transformBaseToSensor(t_base, q_base, Imu_R_wrt_Baselink, Imu_T_wrt_Baselink, t_imu, q_imu);
                imu_pose.pose.position.x = t_imu.x();
                imu_pose.pose.position.y = t_imu.y();
                imu_pose.pose.position.z = t_imu.z();
                imu_pose.pose.orientation.x = q_imu.x();
                imu_pose.pose.orientation.y = q_imu.y();
                imu_pose.pose.orientation.z = q_imu.z();
                imu_pose.pose.orientation.w = q_imu.w();
                imuLinkPath.poses.push_back(imu_pose);
                pubImuLinkPath.publish(imuLinkPath);
            }
        }
    }
}

void publish_path_update(const ros::Publisher pubPath, const ros::Publisher pubLidarOptLinkPath_,
                        const ros::Publisher pubImuOptLinkPath_)
{
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳
    /*** if path is too large, the rvis will crash ***/
    static int kkk = 0;
    kkk++;
    if (kkk % 10 == 0)
    {
        // Publish /x_nav/mapping_path (only when there are subscribers)
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }

        // Publish sensor-frame optimized paths when input_transform is enabled
        if (enable_input_transform)
        {
            if (pubLidarOptLinkPath_.getNumSubscribers() != 0)
            {
                lidarLinkGlobalPath.header.stamp = timeLaserInfoStamp;
                lidarLinkGlobalPath.header.frame_id = odometryFrame;
                pubLidarOptLinkPath_.publish(lidarLinkGlobalPath);
            }

            if (pubImuOptLinkPath_.getNumSubscribers() != 0)
            {
                imuLinkGlobalPath.header.stamp = timeLaserInfoStamp;
                imuLinkGlobalPath.header.frame_id = odometryFrame;
                pubImuOptLinkPath_.publish(imuLinkGlobalPath);
            }
        }
        kkk = 0;
    }
}

//  发布gnss 轨迹
void publish_gnss_path(const ros::Publisher pubPath)
{
    gps_path.header.stamp = ros::Time().fromSec(lidar_end_time);
    gps_path.header.frame_id = odometryFrame;

    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        pubPath.publish(gps_path);
    }
}

/**
 * Publish current velocity as Twist message.
 * Linear velocity: x, y from state_point.vel; z always 0.
 * Angular velocity: only z (yaw rate) from bias-corrected gyro; x, y always 0.
 */
void publish_current_velocity(const ros::Publisher &pubVel)
{
    geometry_msgs::Twist twist;

    // 线速度：仅发布 x 和 y，z 始终为 0
    twist.linear.x = state_point.vel(0);
    twist.linear.y = state_point.vel(1);
    twist.linear.z = 0.0;

    // 角速度：仅发布 z (yaw rate)，x 和 y 始终为 0
    twist.angular.x = 0.0;
    twist.angular.y = 0.0;
    twist.angular.z = 0.0;

    if (!Measures.imu.empty())
    {
        auto last_imu = Measures.imu.back();
        twist.angular.z = last_imu->angular_velocity.z - state_point.bg(2);
    }

    pubVel.publish(twist);
}

/**
 * 发布局部关键帧map的特征点云
 */
void publishGlobalMap()
{
    /*** if path is too large, the rvis will crash ***/
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time);
    if (pubLaserCloudSurround.getNumSubscribers() == 0)
        return;

    if (cloudKeyPoses3D->points.empty() == true)
        return;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
    ;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

    // kdtree查找最近一帧关键帧相邻的关键帧集合
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    mtx.lock();
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
        globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
    // 降采样
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;
    downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
    // 提取局部相邻关键帧对应的特征点云
    for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i)
    {
        // 距离过大
        if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
        int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
        // *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
        *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]); //  fast_lio only use  surfCloud
    }
    // 降采样，发布
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
    publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
}

//构造H矩阵
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;

/** closest surface search and residual computation **/
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++) //判断每个点的对应邻域是否符合平面点的假设
    {
        PointType &point_body = feats_down_body->points[i];   // lidar系下坐标
        PointType &point_world = feats_down_world->points[i]; // lidar数据点在world系下坐标

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);                     // lidar系下坐标
        V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos); // w系下坐标
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)        //  如果收敛了
        {
            /** Find the closest surfaces in the map **/
            // world系下从ikdtree找5个最近点用于平面拟合
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            //最近点数大于NUM_MATCH_POINTS，且最大距离小于等于5,point_selected_surf设置为true
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                                                                                : true;
        }

        //不符合平面特征
        if (!point_selected_surf[i])
            continue;

        VF(4)  pabcd;          //  plane 参数  a b c d
        point_selected_surf[i] = false; //二次筛选平面点
        //拟合局部平面，返回：是否有内点大于距离阈值
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            // plane distance
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm()); //筛选条件 1 - 0.9 * （点到平面距离 / 点到lidar原点距离）

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2; //以intensity记录点到面残差
                res_last[i] = abs(pd2);             // 残差，距离
            }
        }
    }

    effct_feat_num = 0; //有效匹配点数

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i]; // body系 平面特征点
            corr_normvect->points[effct_feat_num] = normvec->points[i];         // world系 平面参数
            total_residual += res_last[i];                                      // 残差和
            effct_feat_num++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num; // 残差均值 （距离）
    match_time += omp_get_wtime() - match_start;
    double solve_start_ = omp_get_wtime();

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //定义H维度
    ekfom_data.h.resize(effct_feat_num);                 //有效方程个数

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p = laserCloudOri->points[i]; // lidar系 平面特征点
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I; // 当前状态imu系下 点坐标
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this); // 当前状态imu系下 点坐标反对称矩阵

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z); //对应局部法相量, world系下

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() * norm_vec); // 将对应局部法相量旋转到imu系下 corr_normal_I
        V3D A(point_crossmat * C);           //残差对角度求导系数 P(IMU)^ [R(imu <-- w) * normal_w]
        //添加数据到矩阵
        if (extrinsic_est_en)
        {
            // B = lidar_p^ R(L <-- I) * corr_normal_I
            // B = lidar_p^ R(L <-- I) * R(I <-- W) * normal_W
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); // s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

/// @brief 地面点云分割：将当前帧点云分割为地面和非地面点云并发布
void groundSegmentation()
{
    if (feats_undistort->empty())
        return;

    PointCloudXYZI::Ptr cloud_base(new PointCloudXYZI());
    int cloudSize = feats_undistort->size();

    // Step 1: 坐标变换到 base_link 系
    // 若 enable_input_transform 为 true，点云已在 base_link 系（callback中已变换），直接使用
    // 否则使用 Lidar -> base_link 外参进行变换
    if (enable_input_transform)
    {
        cloud_base = feats_undistort;
    }
    else
    {
        cloud_base->resize(cloudSize);
        Eigen::Affine3f T_base_lidar = Eigen::Affine3f::Identity();
        T_base_lidar.matrix().block<3,3>(0,0) = Lidar_R_wrt_Baselink.cast<float>();
        T_base_lidar.matrix().block<3,1>(0,3) = Lidar_T_wrt_Baselink.cast<float>();
        pcl::transformPointCloud(*feats_undistort, *cloud_base, T_base_lidar);
    }

    // Step 2: 构建 seg_buf（候选地面点）和 up_buf（低处非地面点）
    PointCloudXYZI::Ptr seg_buf(new PointCloudXYZI());
    PointCloudXYZI::Ptr up_buf(new PointCloudXYZI());

    for (int i = 0; i < cloud_base->size(); i++)
    {
        // 降采样
        if (i % extra_ground_filter_num == 0)
            continue;

        const auto &p = cloud_base->points[i];

        // 高于 clip_hight 的点跳过
        if (p.z > clip_hight)
            continue;

        // 超出 XY 地面范围的点跳过
        if (p.x > ground_distance || p.x < -ground_distance)
            continue;
        if (p.y > ground_distance || p.y < -ground_distance)
            continue;

        // z 在 (0, clip_hight) 范围的点归入低位障碍物
        if (p.z > 0 && p.z < clip_hight)
        {
            up_buf->push_back(p);
            continue;
        }
        else
        {
            seg_buf->push_back(p);
        }
    }

    // Step 3: 在车辆周围生成虚拟地面点 (z = -base_link_hight)
    for (float x_ = -min_distance_b; x_ < min_distance_f; x_ += 0.04f)
    {
        for (float y_ = -min_distance_r; y_ < min_distance_l; y_ += 0.04f)
        {
            PointType buf_;
            buf_.x = x_;
            buf_.y = y_;
            buf_.z = -base_link_hight;
            seg_buf->push_back(buf_);
        }
    }

    // Step 4: 计算候选地面点法向量
    pcl::NormalEstimationOMP<PointType, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
    n.setNumberOfThreads(numberOfCores);
    n.setInputCloud(seg_buf);
    n.setSearchMethod(tree);
    n.setRadiusSearch(0.6);
    n.compute(*normals);

    // Step 5: RANSAC 平面分割
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::SACSegmentationFromNormals<PointType, pcl::Normal> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight(0.8);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(5);
    seg.setDistanceThreshold(ground_distancethreshold);
    seg.setInputCloud(seg_buf);
    seg.setInputNormals(normals);
    seg.segment(*inliers, *coefficients);

    // Step 6: 提取地面和非地面点云
    PointCloudXYZI::Ptr ground(new PointCloudXYZI());
    PointCloudXYZI::Ptr no_ground(new PointCloudXYZI());
    ground->resize(cloudSize);
    no_ground->resize(cloudSize);

    pcl::ExtractIndices<PointType> extract;
    extract.setInputCloud(seg_buf);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*ground);
    extract.setNegative(true);
    extract.filter(*no_ground);

    // 合并低位障碍物到非地面点云
    *no_ground += *up_buf;

    // Step 7: 空点云防护 (添加 sentinel 点避免 ros 发布错误)
    // if (ground->size() == 0)
    // {
    //     PointType buf_;
    //     buf_.x = 0; buf_.y = 0; buf_.z = 10;
    //     ground->push_back(buf_);
    // }
    // if (no_ground->size() == 0)
    // {
    //     PointType buf_;
    //     buf_.x = 0; buf_.y = 0; buf_.z = 10;
    //     no_ground->push_back(buf_);
    // }

    // Step 8: 存储到全局变量 (供关键帧保存使用)
    // 地面分割在 base_link 系下进行, 需转换到与 surf 点云相同的坐标系
    // - enable_input_transform=false: surf 点云在 lidar 系, 需将 ground 转回 lidar 系
    // - enable_input_transform=true:  surf 点云已在 base_link 系, 无需转换
    if (enable_input_transform)
    {
        *current_ground_cloud = *ground;
        *current_noground_cloud = *no_ground;
    }
    else
    {
        Eigen::Affine3f T_lidar_base = Eigen::Affine3f::Identity();
        T_lidar_base.matrix().block<3,3>(0,0) = Lidar_R_wrt_Baselink.cast<float>().transpose();
        T_lidar_base.matrix().block<3,1>(0,3) = -Lidar_R_wrt_Baselink.cast<float>().transpose() * Lidar_T_wrt_Baselink.cast<float>();

        PointCloudXYZI::Ptr ground_lidar(new PointCloudXYZI());
        PointCloudXYZI::Ptr no_ground_lidar(new PointCloudXYZI());
        pcl::transformPointCloud(*ground, *ground_lidar, T_lidar_base);
        pcl::transformPointCloud(*no_ground, *no_ground_lidar, T_lidar_base);
        *current_ground_cloud = *ground_lidar;
        *current_noground_cloud = *no_ground_lidar;
    }

    // Step 9: 始终发布地面和非地面点云 (base_link 系供可视化)
    ros::Time stamp = ros::Time().fromSec(lidar_end_time);

    sensor_msgs::PointCloud2 groundMsg;
    pcl::toROSMsg(*ground, groundMsg);
    groundMsg.header.stamp = stamp;
    groundMsg.header.frame_id = base_linkFrame;
    pubGroundCloud.publish(groundMsg);

    sensor_msgs::PointCloud2 noGroundMsg;
    pcl::toROSMsg(*no_ground, noGroundMsg);
    noGroundMsg.header.stamp = stamp;
    noGroundMsg.header.frame_id = base_linkFrame;
    pubNoGroundCloud.publish(noGroundMsg);
}

/**
 * 全量保存地图到目录
 * 目录结构: surf/<i>.pcd, ground/<i>.pcd, obs/<i>.pcd, poses_*.txt, global_map*.pcd
 */
bool savemap_to_dir(std::string save_dir)
{
    if (cloudKeyPoses3D->points.empty() || surfCloudKeyFrames.empty())
    {
        ROS_WARN("save_map: no keyframes to save");
        return false;
    }

    // 删除旧目录, 重建子目录
    std::string cmd_rm = "exec rm -r " + save_dir;
    system(cmd_rm.c_str());
    mkdir(save_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir((save_dir + "/surf").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir((save_dir + "/ground").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir((save_dir + "/obs").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    int num_keyframes = (int)surfCloudKeyFrames.size();
    ROS_INFO("Saving %d keyframes to %s ...", num_keyframes, save_dir.c_str());

    // 打开位姿文件
    std::ofstream pose_optimized, pose_unoptimized, pose_gnss;
    pose_optimized.open(save_dir + "/poses_gps_align.txt", std::ios::out);
    pose_unoptimized.open(save_dir + "/poses_gps_no_align.txt", std::ios::out);
    pose_gnss.open(save_dir + "/poses_gps_raw.txt", std::ios::out);
    pose_optimized << std::fixed << std::setprecision(9);
    pose_unoptimized << std::fixed << std::setprecision(9);
    pose_gnss << std::fixed << std::setprecision(9);

    // 全局地图拼接用
    pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalGroundCloud(new pcl::PointCloud<PointType>());

    for (int i = 0; i < num_keyframes; i++)
    {
        std::string idx = std::to_string(i);

        // 保存 surf 点云 (lidar 系)
        pcl::io::savePCDFileBinary(save_dir + "/surf/" + idx + ".pcd", *surfCloudKeyFrames[i]);

        // 保存 ground/obs 点云 (如有且非空)
        if (i < (int)groundCloudKeyFrames.size() && groundCloudKeyFrames[i] != nullptr && !groundCloudKeyFrames[i]->empty())
            pcl::io::savePCDFileBinary(save_dir + "/ground/" + idx + ".pcd", *groundCloudKeyFrames[i]);
        if (i < (int)nogroundCloudKeyFrames.size() && nogroundCloudKeyFrames[i] != nullptr && !nogroundCloudKeyFrames[i]->empty())
            pcl::io::savePCDFileBinary(save_dir + "/obs/" + idx + ".pcd", *nogroundCloudKeyFrames[i]);

        // 写入优化位姿 (time x y z qx qy qz qw)
        if (i < (int)cloudKeyPoses6D->size())
        {
            auto &p = cloudKeyPoses6D->points[i];
            Eigen::Quaterniond q = EulerToQuat(p.roll, p.pitch, p.yaw);
            pose_optimized << p.time << " "
                           << p.x << " " << p.y << " " << p.z << " "
                           << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        }

        // 拼接全局地图 (转换到世界系)
        if (i < (int)cloudKeyPoses6D->size())
            *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
    }

    // 写入未优化位姿 (fastlio)
    for (int i = 0; i < (int)fastlio_unoptimized_cloudKeyPoses6D->size(); i++)
    {
        auto &p = fastlio_unoptimized_cloudKeyPoses6D->points[i];
        Eigen::Quaterniond q = EulerToQuat(p.roll, p.pitch, p.yaw);
        pose_unoptimized << p.time << " "
                         << p.x << " " << p.y << " " << p.z << " "
                         << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }

    // 写入 GNSS 位姿
    for (int i = 0; i < (int)gnss_cloudKeyPoses6D->size(); i++)
    {
        auto &p = gnss_cloudKeyPoses6D->points[i];
        Eigen::Quaterniond q = EulerToQuat(p.roll, p.pitch, p.yaw);
        pose_gnss << p.time << " "
                  << p.x << " " << p.y << " " << p.z << " "
                  << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }

    pose_optimized.close();
    pose_unoptimized.close();
    pose_gnss.close();

    // 构建并保存全局地面地图 (如有ground数据)
    if (!groundCloudKeyFrames.empty())
    {
        for (int i = 0; i < (int)groundCloudKeyFrames.size() && i < (int)cloudKeyPoses6D->size(); i++)
        {
            if (groundCloudKeyFrames[i] != nullptr && !groundCloudKeyFrames[i]->empty())
                *globalGroundCloud += *transformPointCloud(groundCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
        }
    }

    // 降采样并保存全局地图
    pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
    downSizeFilterSurf.setInputCloud(globalSurfCloud);
    downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterSurf.filter(*globalSurfCloudDS);

    pcl::io::savePCDFileBinary(save_dir + "/global_map.pcd", *globalSurfCloud);
    pcl::io::savePCDFileBinary(save_dir + "/global_map_downsize.pcd", *globalSurfCloudDS);

    if (!globalGroundCloud->empty())
    {
        pcl::PointCloud<PointType>::Ptr globalGroundCloudDS(new pcl::PointCloud<PointType>());
        downSizeFilterSurf.setInputCloud(globalGroundCloud);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurf.filter(*globalGroundCloudDS);
        pcl::io::savePCDFileBinary(save_dir + "/global_ground_map.pcd", *globalGroundCloud);
        pcl::io::savePCDFileBinary(save_dir + "/global_ground_map_downsize.pcd", *globalGroundCloudDS);
    }

    // 发布全局地图
    ros::Time stamp = ros::Time().fromSec(lidar_end_time);
    publishCloud(&pubOptimizedGlobalMap, globalSurfCloudDS, stamp, odometryFrame);

    ROS_INFO("save_map completed: %d keyframes saved to %s", num_keyframes, save_dir.c_str());
    return true;
}

/**
 * 增量保存地图到目录
 * 仅保存新增的关键帧, 已存在的文件通过size比对决定是否覆盖
 */
bool savemap_to_dir_incremental(std::string save_dir)
{
    if (cloudKeyPoses3D->points.empty() || surfCloudKeyFrames.empty())
    {
        ROS_WARN("increase_map: no keyframes to save");
        return false;
    }

    int num_keyframes = (int)surfCloudKeyFrames.size();

    // 确保目录存在
    mkdir(save_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir((save_dir + "/surf").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir((save_dir + "/ground").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir((save_dir + "/obs").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    // 统计已有文件数量
    int existing_files = 0;
    while (true)
    {
        struct stat buffer;
        std::string fname = save_dir + "/surf/" + std::to_string(existing_files) + ".pcd";
        if (stat(fname.c_str(), &buffer) == 0)
            existing_files++;
        else
            break;
    }

    ROS_INFO("increase_map: %d existing, %d total keyframes", existing_files, num_keyframes);

    // 保存新增的关键帧逐帧 PCD
    for (int i = 0; i < num_keyframes; i++)
    {
        std::string idx = std::to_string(i);

        // surf
        {
            std::string path = save_dir + "/surf/" + idx + ".pcd";
            if (i < existing_files)
            {
                // 已存在: 写入临时文件后比较size, 变化则覆盖
                std::string tmp_path = path + ".tmp";
                pcl::io::savePCDFileBinary(tmp_path, *surfCloudKeyFrames[i]);
                struct stat st_orig, st_tmp;
                bool overwrite = (stat(path.c_str(), &st_orig) != 0) ||
                                 (stat(tmp_path.c_str(), &st_tmp) != 0) ||
                                 (st_orig.st_size != st_tmp.st_size);
                if (overwrite)
                {
                    rename(tmp_path.c_str(), path.c_str());
                }
                else
                {
                    remove(tmp_path.c_str());
                }
            }
            else
            {
                pcl::io::savePCDFileBinary(path, *surfCloudKeyFrames[i]);
            }
        }

        // ground (仅保存非空点云, 跳过已存在的帧)
        if (i < (int)groundCloudKeyFrames.size() && groundCloudKeyFrames[i] != nullptr && !groundCloudKeyFrames[i]->empty())
        {
            std::string path = save_dir + "/ground/" + idx + ".pcd";
            if (i < existing_files)
            {
                std::string tmp_path = path + ".tmp";
                pcl::io::savePCDFileBinary(tmp_path, *groundCloudKeyFrames[i]);
                struct stat st_orig, st_tmp;
                if (stat(path.c_str(), &st_orig) != 0 || stat(tmp_path.c_str(), &st_tmp) != 0 ||
                    st_orig.st_size != st_tmp.st_size)
                    rename(tmp_path.c_str(), path.c_str());
                else
                    remove(tmp_path.c_str());
            }
            else
            {
                pcl::io::savePCDFileBinary(path, *groundCloudKeyFrames[i]);
            }
        }

        // obs (仅保存非空点云, 跳过已存在的帧)
        if (i < (int)nogroundCloudKeyFrames.size() && nogroundCloudKeyFrames[i] != nullptr && !nogroundCloudKeyFrames[i]->empty())
        {
            std::string path = save_dir + "/obs/" + idx + ".pcd";
            if (i < existing_files)
            {
                std::string tmp_path = path + ".tmp";
                pcl::io::savePCDFileBinary(tmp_path, *nogroundCloudKeyFrames[i]);
                struct stat st_orig, st_tmp;
                if (stat(path.c_str(), &st_orig) != 0 || stat(tmp_path.c_str(), &st_tmp) != 0 ||
                    st_orig.st_size != st_tmp.st_size)
                    rename(tmp_path.c_str(), path.c_str());
                else
                    remove(tmp_path.c_str());
            }
            else
            {
                pcl::io::savePCDFileBinary(path, *nogroundCloudKeyFrames[i]);
            }
        }
    }

    // 重写位姿文件 (始终保持最新)
    {
        std::ofstream pose_optimized, pose_unoptimized, pose_gnss;
        pose_optimized.open(save_dir + "/poses_gps_align.txt", std::ios::out);
        pose_unoptimized.open(save_dir + "/poses_gps_no_align.txt", std::ios::out);
        pose_gnss.open(save_dir + "/poses_gps_raw.txt", std::ios::out);
        pose_optimized << std::fixed << std::setprecision(9);
        pose_unoptimized << std::fixed << std::setprecision(9);
        pose_gnss << std::fixed << std::setprecision(9);

        for (int i = 0; i < (int)cloudKeyPoses6D->size(); i++)
        {
            auto &p = cloudKeyPoses6D->points[i];
            Eigen::Quaterniond q = EulerToQuat(p.roll, p.pitch, p.yaw);
            pose_optimized << p.time << " " << p.x << " " << p.y << " " << p.z << " "
                           << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        }
        for (int i = 0; i < (int)fastlio_unoptimized_cloudKeyPoses6D->size(); i++)
        {
            auto &p = fastlio_unoptimized_cloudKeyPoses6D->points[i];
            Eigen::Quaterniond q = EulerToQuat(p.roll, p.pitch, p.yaw);
            pose_unoptimized << p.time << " " << p.x << " " << p.y << " " << p.z << " "
                             << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        }
        for (int i = 0; i < (int)gnss_cloudKeyPoses6D->size(); i++)
        {
            auto &p = gnss_cloudKeyPoses6D->points[i];
            Eigen::Quaterniond q = EulerToQuat(p.roll, p.pitch, p.yaw);
            pose_gnss << p.time << " " << p.x << " " << p.y << " " << p.z << " "
                      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        }
        pose_optimized.close();
        pose_unoptimized.close();
        pose_gnss.close();
    }

    // 重建全局地图
    {
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalGroundCloud(new pcl::PointCloud<PointType>());

        for (int i = 0; i < (int)surfCloudKeyFrames.size() && i < (int)cloudKeyPoses6D->size(); i++)
            *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);

        if (!groundCloudKeyFrames.empty())
        {
            for (int i = 0; i < (int)groundCloudKeyFrames.size() && i < (int)cloudKeyPoses6D->size(); i++)
                if (groundCloudKeyFrames[i] != nullptr && !groundCloudKeyFrames[i]->empty())
                    *globalGroundCloud += *transformPointCloud(groundCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
        }

        pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurf.filter(*globalSurfCloudDS);

        pcl::io::savePCDFileBinary(save_dir + "/global_map.pcd", *globalSurfCloud);
        pcl::io::savePCDFileBinary(save_dir + "/global_map_downsize.pcd", *globalSurfCloudDS);

        if (!globalGroundCloud->empty())
        {
            pcl::PointCloud<PointType>::Ptr globalGroundCloudDS(new pcl::PointCloud<PointType>());
            downSizeFilterSurf.setInputCloud(globalGroundCloud);
            downSizeFilterSurf.filter(*globalGroundCloudDS);
            pcl::io::savePCDFileBinary(save_dir + "/global_ground_map.pcd", *globalGroundCloud);
            pcl::io::savePCDFileBinary(save_dir + "/global_ground_map_downsize.pcd", *globalGroundCloudDS);
        }

        ros::Time stamp = ros::Time().fromSec(lidar_end_time);
        publishCloud(&pubOptimizedGlobalMap, globalSurfCloudDS, stamp, odometryFrame);
    }

    last_saved_keyframe_index = num_keyframes - 1;
    ROS_INFO("increase_map completed: %d keyframes", num_keyframes);
    return true;
}

/**
 * 设置 SLAM 模式 (对齐 digitaltwins-x-nav 启动逻辑)
 */
void Set_mapping()
{
    slam_mode = 0;
}

void Set_localization()
{
    slam_mode = 1;
    initializedFlag = 0;
    hand_init_state = false;
    localization_flag_vel = 0;
}

/**
 * 从目录加载地图 (localization 模式)
 * 读取 poses_gps_align.txt 和逐帧 PCD, 重建 ISAM2 因子图和 ikd-Tree
 */
bool load_map(std::string load_dir)
{
    if (slam_mode != 1)
    {
        ROS_ERROR("load_map: only available in localization mode (slam_mode=1), current slam_mode=%d", slam_mode);
        return false;
    }

    // 1. 读取位姿文件
    std::string pose_file = load_dir + "/poses_gps_align.txt";
    std::ifstream ifs(pose_file);
    if (!ifs.is_open())
    {
        ROS_ERROR("load_map: cannot open %s", pose_file.c_str());
        return false;
    }

    struct PoseStored
    {
        double time;
        double x, y, z;
        double qx, qy, qz, qw;
    };
    std::vector<PoseStored> loaded_poses;
    std::string line;
    while (std::getline(ifs, line))
    {
        if (line.empty()) continue;
        std::istringstream iss(line);
        PoseStored p;
        if (iss >> p.time >> p.x >> p.y >> p.z >> p.qx >> p.qy >> p.qz >> p.qw)
            loaded_poses.push_back(p);
    }
    ifs.close();

    if (loaded_poses.empty())
    {
        ROS_ERROR("load_map: no poses found in %s", pose_file.c_str());
        return false;
    }

    ROS_INFO("load_map: loaded %zu poses from %s", loaded_poses.size(), pose_file.c_str());

    // 2. 清除现有数据
    cloudKeyPoses3D->clear();
    cloudKeyPoses6D->clear();
    fastlio_unoptimized_cloudKeyPoses6D->clear();
    surfCloudKeyFrames.clear();
    groundCloudKeyFrames.clear();
    nogroundCloudKeyFrames.clear();
    globalPath.poses.clear();

    // 3. 加载逐帧点云
    int loaded_count = 0;
    for (int i = 0; i < (int)loaded_poses.size(); i++)
    {
        std::string surf_path = load_dir + "/surf/" + std::to_string(i) + ".pcd";

        pcl::PointCloud<PointType>::Ptr surf_cloud(new pcl::PointCloud<PointType>());
        if (pcl::io::loadPCDFile(surf_path, *surf_cloud) == -1)
        {
            ROS_WARN("load_map: failed to load %s, stopping", surf_path.c_str());
            break;
        }

        // // 加载 ground/obs (可选)
        // pcl::PointCloud<PointType>::Ptr ground_cloud(new pcl::PointCloud<PointType>());
        // pcl::PointCloud<PointType>::Ptr noground_cloud(new pcl::PointCloud<PointType>());

        // std::string ground_path = load_dir + "/ground/" + std::to_string(i) + ".pcd";
        // std::string obs_path = load_dir + "/obs/" + std::to_string(i) + ".pcd";
        // struct stat st;
        // if (stat(ground_path.c_str(), &st) == 0)
        //     pcl::io::loadPCDFile(ground_path, *ground_cloud);
        // if (stat(obs_path.c_str(), &st) == 0)
        //     pcl::io::loadPCDFile(obs_path, *noground_cloud);

        // 添加关键帧位姿
        PointType thisPose3D;
        thisPose3D.x = loaded_poses[i].x;
        thisPose3D.y = loaded_poses[i].y;
        thisPose3D.z = loaded_poses[i].z;
        thisPose3D.intensity = i;
        cloudKeyPoses3D->push_back(thisPose3D);

        PointTypePose thisPose6D;
        thisPose6D.x = loaded_poses[i].x;
        thisPose6D.y = loaded_poses[i].y;
        thisPose6D.z = loaded_poses[i].z;
        thisPose6D.intensity = i;
        thisPose6D.time = loaded_poses[i].time;
        Eigen::Quaterniond q(loaded_poses[i].qw, loaded_poses[i].qx, loaded_poses[i].qy, loaded_poses[i].qz);
        Eigen::Vector3d rpy = q.toRotationMatrix().eulerAngles(2, 1, 0); // yaw, pitch, roll
        thisPose6D.roll = rpy(2);
        thisPose6D.pitch = rpy(1);
        thisPose6D.yaw = rpy(0);
        cloudKeyPoses6D->push_back(thisPose6D);
        fastlio_unoptimized_cloudKeyPoses6D->push_back(thisPose6D);

        // 添加关键帧点云
        surfCloudKeyFrames.push_back(surf_cloud);
        // if (!ground_cloud->empty() || !noground_cloud->empty())
        // {
        //     groundCloudKeyFrames.push_back(ground_cloud);
        //     nogroundCloudKeyFrames.push_back(noground_cloud);
        // }

        // 更新可视化路径
        updatePath(thisPose6D);

        loaded_count++;
    }

    ROS_INFO("load_map: loaded %d keyframes", loaded_count);

    // 3.5 加载 GNSS 关键帧关联数据 (用于 auto-localization)
    gnss_keyframe_links.clear();
    gnssCloudKeyPoses3D->clear();
    {
        std::string gf = load_dir + "/poses_gps_raw.txt";
        std::ifstream ifs_gnss(gf);
        if (ifs_gnss.is_open())
        {
            std::string gl;
            while (std::getline(ifs_gnss, gl))
            {
                if (gl.empty()) continue;
                std::istringstream gss(gl);
                double gt, gx, gy, gz, gqx, gqy, gqz, gqw;
                if (gss >> gt >> gx >> gy >> gz >> gqx >> gqy >> gqz >> gqw)
                {
                    int nk = -1; double md = 1.0;
                    for (int k = 0; k < loaded_count; k++)
                    {
                        double d = std::abs(cloudKeyPoses6D->points[k].time - gt);
                        if (d < md) { md = d; nk = k; }
                    }
                    if (nk >= 0)
                    {
                        PointType gp;
                        gp.x = gx; gp.y = gy; gp.z = gz; gp.intensity = nk;
                        gnssCloudKeyPoses3D->push_back(gp);
                        gnss_keyframe_links.push_back({gx, gy, gz, nk});
                    }
                }
            }
            ifs_gnss.close();
            if (!gnssCloudKeyPoses3D->empty())
                kdtreeGNSSKeyPoses->setInputCloud(gnssCloudKeyPoses3D);
            ROS_INFO("load_map: %zu GNSS-keyframe links loaded for auto-localization",
                     gnssCloudKeyPoses3D->points.size());
        }
        else
        {
            ROS_WARN("load_map: no %s, auto-localization disabled", gf.c_str());
        }
    }

    // 4. 重置 ISAM2 并重建因子图
    delete isam;
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new gtsam::ISAM2(parameters);
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    double load_map_noise_rpy = 1e-9;
    double load_map_noise_xyz = 1e-9;

    for (int i = 0; i < loaded_count; i++)
    {
        if (i == 0)
        {
            gtsam::noiseModel::Diagonal::shared_ptr priorNoise =
                gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6)
                    << load_map_noise_rpy, load_map_noise_rpy, load_map_noise_rpy,
                       load_map_noise_xyz, load_map_noise_xyz, load_map_noise_xyz).finished());
            gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, pclPointTogtsamPose3(cloudKeyPoses6D->points[0]), priorNoise));
            initialEstimate.insert(0, pclPointTogtsamPose3(cloudKeyPoses6D->points[0]));
        }
        else
        {
            gtsam::noiseModel::Diagonal::shared_ptr odomNoise =
                gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6)
                    << load_map_noise_rpy, load_map_noise_rpy, load_map_noise_rpy,
                       load_map_noise_xyz, load_map_noise_xyz, load_map_noise_xyz).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points[i - 1]);
            gtsam::Pose3 poseTo = pclPointTogtsamPose3(cloudKeyPoses6D->points[i]);
            gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(i - 1, i, poseFrom.between(poseTo), odomNoise));
            initialEstimate.insert(i, poseTo);
        }
    }

    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    isam->update();
    isam->update();
    isam->update();
    isam->update();

    isamCurrentEstimate = isam->calculateBestEstimate();
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    // 5. 重建 ikd-Tree（模拟 mapping 模式增量构建过程）
    // Step 5a: 设置降采样参数，与 mapping 模式 line 4169 一致
    ikdtree.set_downsample_param(filter_size_map_min);

    // Step 5b: 创建逐帧体素降采样过滤器
    // 模拟每帧 downSizeFilterSurf.filter()（line 4159-4160）
    // 在 lidar 系下降采样（与 mapping 模式一致），再转换到 world 系
    pcl::VoxelGrid<PointType> frame_voxel_filter;
    frame_voxel_filter.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

    // 用于可视化的全局地图点云（发布原始未过滤点云转换到 world 系）
    pcl::PointCloud<PointType>::Ptr ikdtree_points(new pcl::PointCloud<PointType>());

    bool first_frame = true;
    int total_added = 0;

    for (int i = 0; i < loaded_count; i++)
    {
        // 收集全局可视化点云（原始点云转换到 world 系）
        *ikdtree_points += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);

        // Step 5c: 在 lidar 系下体素降采样（模拟 mapping 模式的 downSizeFilterSurf）
        pcl::PointCloud<PointType>::Ptr frame_filtered(new pcl::PointCloud<PointType>());
        frame_voxel_filter.setInputCloud(surfCloudKeyFrames[i]);
        frame_voxel_filter.filter(*frame_filtered);

        if (frame_filtered->empty()) continue;

        // Step 5d: 转换到 world 系
        pcl::PointCloud<PointType>::Ptr frame_world =
            transformPointCloud(frame_filtered, &cloudKeyPoses6D->points[i]);

        // Step 5e: 增量构建 ikd-Tree
        if (first_frame)
        {
            // 第一帧用 Build() 初始化（与 mapping 模式 line 4176 一致）
            if (frame_world->size() > 5)
            {
                ikdtree.Build(frame_world->points);
                first_frame = false;
            }
        }
        else
        {
            // 后续帧用 Add_Points + 降采样（与 mapping 模式 map_incremental line 2243 一致）
            total_added += ikdtree.Add_Points(frame_world->points, true);
        }
    }

    if (first_frame && !ikdtree_points->empty())
    {
        // 回退：如果没有任何一帧满足 Build 阈值条件，使用全量 Build
        ROS_WARN("load_map: no single frame met Build threshold (>5 pts), using filtered global Build");
        pcl::PointCloud<PointType>::Ptr filtered_all(new pcl::PointCloud<PointType>());
        frame_voxel_filter.setInputCloud(ikdtree_points);
        frame_voxel_filter.filter(*filtered_all);
        ikdtree.reconstruct(filtered_all->points);
    }

    ROS_INFO("load_map: ikd-Tree reconstructed — %d keyframes, %d points added via Add_Points",
             loaded_count, total_added);

    // 6. 发布加载后的路径和状态
    ros::Time stamp = ros::Time().fromSec(lidar_end_time);
    PointVector().swap(ikdtree.PCL_Storage);
    ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
    featsFromMap->clear();
    featsFromMap->points = ikdtree.PCL_Storage;
    publishCloud(&pubOptimizedGlobalMap, featsFromMap, stamp, odometryFrame);

    std_msgs::String state_msg;
    state_msg.data = "0";
    pub_slam_state.publish(state_msg);

    ROS_INFO("load_map completed: %d keyframes loaded from %s", loaded_count, load_dir.c_str());
    return true;
}

/**
 * 统一 slam service 回调
 * 通过 req.cmd 字段分发命令
 */
bool CallbackFrom_slam_service(fast_lio_sam::nav_functionRequest& req, fast_lio_sam::nav_functionResponse& res)
{
    ROS_INFO("[slam_service] cmd='%s'", req.cmd.c_str());

    if (req.cmd.empty())
    {
        ROS_ERROR("slam_service: cmd is empty");
        res.success = 0;
        res.return_msgs = "cmd is empty";
        return true;
    }

    // ---- "get_map" ----
    if (req.cmd == "get_map")
    {
        if (cloudKeyPoses3D->points.empty())
        {
            res.success = 0;
            res.return_msgs = "get_map fail: no keyframes";
            return true;
        }
        show_globalmap_flag = true;
        res.success = 1;
        res.return_msgs = "get_map triggered";
        return true;
    }

    // ---- "save_map" ----
    if (req.cmd == "save_map")
    {
        std::string save_dir = req.set_value_string.empty()
            ? std::string(std::getenv("HOME")) + "/fast_lio_sam_map"
            : req.set_value_string;
        bool ok = savemap_to_dir(save_dir);
        res.success = ok ? 1 : 0;
        if (!ok)
            res.return_msgs = "save_map failed: keyframes=" + std::to_string(surfCloudKeyFrames.size());
        else
            res.return_msgs = "save_map success: " + save_dir;
        return true;
    }

    // ---- "increase_map" ----
    if (req.cmd == "increase_map")
    {
        std::string save_dir = req.set_value_string.empty()
            ? std::string(std::getenv("HOME")) + "/fast_lio_sam_map"
            : req.set_value_string;
        bool ok = savemap_to_dir_incremental(save_dir);
        res.success = ok ? 1 : 0;
        if (!ok)
            res.return_msgs = "increase_map failed: keyframes=" + std::to_string(surfCloudKeyFrames.size());
        else
            res.return_msgs = "increase_map success: " + save_dir;
        return true;
    }

    // ---- "load_map" ----
    if (req.cmd == "load_map")
    {
        std::string load_dir = req.set_value_string;
        if (load_dir.empty())
        {
            res.success = 0;
            res.return_msgs = "load_map: set_value_string (directory) is required";
            return true;
        }
        if (slam_mode != 1)
        {
            res.success = 0;
            res.return_msgs = "load_map: must be in localization mode (slam_mode=1), current mode is mapping";
            return true;
        }
        bool ok = load_map(load_dir);
        res.success = ok ? 1 : 0;
        res.return_msgs = ok ? "load_map success: " + load_dir : "load_map failed";
        return true;
    }

    // ---- "change_loop_search_radius" ----
    if (req.cmd == "change_loop_search_radius")
    {
        if (req.set_value_float32 == 0.0f)
        {
            res.success = 0;
            res.return_msgs = "cannot set loop_search_radius to zero";
            return true;
        }
        historyKeyframeSearchRadius = req.set_value_float32;
        res.success = 1;
        res.return_msgs = "loop_search_radius changed to " + std::to_string(req.set_value_float32);
        return true;
    }

    // ---- "change_keyframe_distance" ----
    if (req.cmd == "change_keyframe_distance")
    {
        if (req.set_value_float32 == 0.0f)
        {
            res.success = 0;
            res.return_msgs = "cannot set keyframe_distance to zero";
            return true;
        }
        surroundingkeyframeAddingDistThreshold = req.set_value_float32;
        res.success = 1;
        res.return_msgs = "keyframe_distance changed to " + std::to_string(req.set_value_float32);
        return true;
    }

    // ---- "change_keyframe_angle" ----
    if (req.cmd == "change_keyframe_angle")
    {
        if (req.set_value_float32 == 0.0f)
        {
            res.success = 0;
            res.return_msgs = "cannot set keyframe_angle to zero";
            return true;
        }
        surroundingkeyframeAddingAngleThreshold = req.set_value_float32;
        res.success = 1;
        res.return_msgs = "keyframe_angle changed to " + std::to_string(req.set_value_float32);
        return true;
    }

    // ---- "change_localmap_radius" ----
    if (req.cmd == "change_localmap_radius")
    {
        if (req.set_value_float32 == 0.0f)
        {
            res.success = 0;
            res.return_msgs = "cannot set localmap_radius to zero";
            return true;
        }
        surroundingKeyframeSearchRadius = req.set_value_float32;
        res.success = 1;
        res.return_msgs = "localmap_radius changed to " + std::to_string(req.set_value_float32);
        return true;
    }

    // ---- "change_localmap_density" ----
    if (req.cmd == "change_localmap_density")
    {
        if (req.set_value_float32 == 0.0f)
        {
            res.success = 0;
            res.return_msgs = "cannot set localmap_density to zero";
            return true;
        }
        surroundingKeyframeDensity = req.set_value_float32;
        res.success = 1;
        res.return_msgs = "localmap_density changed to " + std::to_string(req.set_value_float32);
        return true;
    }

    // ---- "set_mapping" ----
    if (req.cmd == "set_mapping")
    {
        Set_mapping();
        res.success = 1;
        res.return_msgs = "Switched to mapping mode (slam_mode=0)";
        return true;
    }

    // ---- "set_localization" ----
    if (req.cmd == "set_localization")
    {
        Set_localization();
        if (!req.set_value_string.empty())
        {
            bool ok = load_map(req.set_value_string);
            if (!ok)
            {
                res.success = 0;
                res.return_msgs = "set_localization: load_map failed for " + req.set_value_string;
                return true;
            }
            res.return_msgs = "Switched to localization mode (slam_mode=1), map loaded: " + req.set_value_string;
        }
        else
        {
            res.return_msgs = "Switched to localization mode (slam_mode=1), no map loaded (use load_map after)";
        }
        res.success = 1;
        return true;
    }

    // 未知命令
    res.success = 0;
    res.return_msgs = "unknown cmd: " + req.cmd;
    ROS_WARN("slam_service: unknown cmd '%s'", req.cmd.c_str());
    return true;
}

int main(int argc, char **argv)
{
    // ---- 命令行参数解析 (对齐 digitaltwins-x-nav x_slam.cc 启动逻辑) ----
    if (argc != 2 && argc != 3 && argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <mapping|localization> [map_dir1] [map_dir2]" << std::endl;
        return 1;
    }

    std::string run_mode = argv[1];
    if (run_mode != "mapping" && run_mode != "localization")
    {
        std::cerr << "Invalid mode: " << run_mode << ". Must be 'mapping' or 'localization'." << std::endl;
        return 1;
    }

    int load_map_num = argc - 2;
    std::string map_dir1, map_dir2;

    if (run_mode == "localization")
    {
        if (argc != 3 && argc != 4)
        {
            std::cerr << "Localization mode requires 1 or 2 map directories." << std::endl;
            return 1;
        }

        // 验证地图目录存在且非空 (对齐 x_slam.cc:33-72)
        for (int i = 2; i < argc; i++)
        {
            std::string dir = argv[i];
            bool isEmpty = true;
            try
            {
                for (const auto& entry : std::filesystem::directory_iterator(dir))
                {
                    if (std::filesystem::is_regular_file(entry.path()))
                    {
                        isEmpty = false;
                        break;
                    }
                }
            }
            catch (const std::filesystem::filesystem_error& e)
            {
                std::cerr << "Failed to access directory: " << dir << " - " << e.what() << std::endl;
                return 1;
            }
            if (isEmpty)
            {
                std::cerr << "Map directory is empty: " << dir << std::endl;
                return 1;
            }
        }

        map_dir1 = argv[2];
        if (load_map_num == 2)
            map_dir2 = argv[3];
    }

    // 原有初始化代码
    // allocateMemory();
    for (int i = 0; i < 6; ++i)
    {
        transformTobeMapped[i] = 0;
    }

    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh,private_nh("~");
    
    nh.param<std::string>("tf_prefix", rosNamespace, "");
    if(rosNamespace != ""){
        rosNamespace = rosNamespace + "/";
    }

    // Frame ID 参数配置
    nh.param<std::string>("odometryFrame", odometryFrame, rosNamespace + "odom");
    nh.param<std::string>("base_linkFrame", base_linkFrame, rosNamespace + "base_link");
    nh.param<std::string>("lidar_linkFrame", lidar_linkFrame, rosNamespace + "lidar");
    nh.param<std::string>("imu_linkFrame", imu_linkFrame, rosNamespace + "imu");
    nh.param<std::string>("mapFrame", mapFrame, rosNamespace + "map");

    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<string>("map_file_path", map_file_path, "");
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5);
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    nh.param<double>("cube_side_length", cube_len, 200);
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
    nh.param<double>("mapping/fov_degree", fov_deg, 180);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);

    // 地面分割参数
    nh.param<bool>("ground_segmentation/enable", enable_ground_seg, false);
    nh.param<float>("ground_segmentation/ground_distance", ground_distance, 3.75f);
    nh.param<float>("ground_segmentation/clip_hight", clip_hight, 0.3f);
    nh.param<float>("ground_segmentation/ground_distancethreshold", ground_distancethreshold, 0.2f);
    nh.param<float>("ground_segmentation/base_link_hight", base_link_hight, 0.5f);
    nh.param<float>("ground_segmentation/min_distance_f", min_distance_f, 0.05f);
    nh.param<float>("ground_segmentation/min_distance_b", min_distance_b, 0.5f);
    nh.param<float>("ground_segmentation/min_distance_l", min_distance_l, 0.25f);
    nh.param<float>("ground_segmentation/min_distance_r", min_distance_r, 0.25f);
    nh.param<int>("ground_segmentation/extra_ground_filter_num", extra_ground_filter_num, 2);

    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    cout << "p_pre->lidar_type " << p_pre->lidar_type << endl;

    // 新增：读取输入转换外参
    nh.param<bool>("input_transform/enable", enable_input_transform, false);
    vector<double> imu_to_baselink_T_vec, imu_to_baselink_R_vec;
    vector<double> lidar_to_baselink_T_vec, lidar_to_baselink_R_vec;
    nh.param<vector<double>>("input_transform/imu_to_baselink_T", imu_to_baselink_T_vec, vector<double>());
    nh.param<vector<double>>("input_transform/imu_to_baselink_R", imu_to_baselink_R_vec, vector<double>());
    nh.param<vector<double>>("input_transform/lidar_to_baselink_T", lidar_to_baselink_T_vec, vector<double>());
    nh.param<vector<double>>("input_transform/lidar_to_baselink_R", lidar_to_baselink_R_vec, vector<double>());

    // 初始化转换矩阵
    if (enable_input_transform)
    {
        if (imu_to_baselink_T_vec.size() == 3)
        {
            Imu_T_wrt_Baselink << VEC_FROM_ARRAY(imu_to_baselink_T_vec);
        }
        if (imu_to_baselink_R_vec.size() == 9)
        {
            Imu_R_wrt_Baselink << MAT_FROM_ARRAY(imu_to_baselink_R_vec);
        }
        if (lidar_to_baselink_T_vec.size() == 3)
        {
            Lidar_T_wrt_Baselink << VEC_FROM_ARRAY(lidar_to_baselink_T_vec);
        }
        if (lidar_to_baselink_R_vec.size() == 9)
        {
            Lidar_R_wrt_Baselink << MAT_FROM_ARRAY(lidar_to_baselink_R_vec);
        }

        ROS_INFO("Input transform enabled:");
        ROS_INFO("  IMU->Baselink T: [%.3f, %.3f, %.3f]",
                 Imu_T_wrt_Baselink.x(), Imu_T_wrt_Baselink.y(), Imu_T_wrt_Baselink.z());
        ROS_INFO("  Lidar->Baselink T: [%.3f, %.3f, %.3f]",
                 Lidar_T_wrt_Baselink.x(), Lidar_T_wrt_Baselink.y(), Lidar_T_wrt_Baselink.z());
    }

    nh.param<float>("odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
    nh.param<float>("mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
    nh.param<float>("mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

    nh.param<float>("z_tollerance", z_tollerance, FLT_MAX);
    nh.param<float>("rotation_tollerance", rotation_tollerance, FLT_MAX);

    nh.param<int>("numberOfCores", numberOfCores, 2);
    nh.param<double>("mappingProcessInterval", mappingProcessInterval, 0.15);

    // save keyframes
    nh.param<float>("surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 20.0);
    nh.param<float>("surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
    nh.param<float>("surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
    nh.param<float>("surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

    // loop clousre
    nh.param<bool>("loopClosureEnableFlag", loopClosureEnableFlag, false);
    nh.param<float>("loopClosureFrequency", loopClosureFrequency, 1.0);
    nh.param<int>("surroundingKeyframeSize", surroundingKeyframeSize, 50);
    nh.param<float>("historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
    nh.param<float>("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
    nh.param<int>("historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
    nh.param<float>("historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

    // gnss
    nh.param<string>("common/gnss_topic", gnss_topic,"gps/fix");
    nh.param<vector<double>>("mapping/extrinR_Gnss2Lidar", extrinR_Gnss2Lidar, vector<double>());
    nh.param<vector<double>>("mapping/extrinT_Gnss2Lidar", extrinT_Gnss2Lidar, vector<double>());
    nh.param<bool>("useImuHeadingInitialization", useImuHeadingInitialization, false);
    nh.param<bool>("useGpsElevation", useGpsElevation, false);
    nh.param<float>("gpsCovThreshold", gpsCovThreshold, 2.0);
    nh.param<float>("gpsCovThreshold_Z", gpsCovThreshold_Z, 200.0);
    nh.param<float>("poseCovThreshold", poseCovThreshold, 25.0);
    nh.param<int>("gnss_calib_min_match", gnss_calib_min_match, 50);
    nh.param<double>("gnss_calib_max_time_diff", gnss_calib_max_time_diff, 0.5);

    // ---- Localization 初始化参数 (对齐 digitaltwins-x-nav) ----
    nh.param<bool>("localization/auto_localization_enable", auto_localization_enableflag, false);
    nh.param<float>("localization/initframe_FitnessScore", initframe_FitnessScore, 0.1);
    nh.param<float>("localization/search_radius", localization_search_radius, 5.0);
    nh.param<float>("localization/localmap_leafsize", localization_localmap_leafsize, 0.5);
    nh.param<float>("localization/icp_correspondence_dist", localization_icp_correspondence_dist, 100.0);
    nh.param<int>("localization/icp_max_iters", localization_icp_max_iters, 100);
    nh.param<float>("localization/safe_speed", safe_speed, 15.0);

    // Visualization
    nh.param<float>("globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
    nh.param<float>("globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
    nh.param<float>("globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

    // visual ikdtree map
    nh.param<bool>("visulize_IkdtreeMap", visulize_IkdtreeMap, false);

    // reconstruct ikdtree
    nh.param<bool>("recontructKdTree", recontructKdTree, false);

    downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    // downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

    // ISAM2参数
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new gtsam::ISAM2(parameters);

    path.header.stamp = ros::Time::now();
    path.header.frame_id = odometryFrame;

    lidarLinkPath.header.frame_id = odometryFrame;
    imuLinkPath.header.frame_id = odometryFrame;
    lidarLinkGlobalPath.header.frame_id = odometryFrame;
    imuLinkGlobalPath.header.frame_id = odometryFrame;

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;

    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG)*0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf)); //重复？
    memset(res_last, -1000.0f, sizeof(res_last));

    //设置imu和lidar外参和imu参数等
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov)); // 加速度协方差
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    //设置gnss外参数
    Gnss_T_wrt_Lidar<<VEC_FROM_ARRAY(extrinT_Gnss2Lidar);
    Gnss_R_wrt_Lidar<<MAT_FROM_ARRAY(extrinR_Gnss2Lidar);

    double epsi[23] = {0.001};
    fill(epsi, epsi + 23, 0.001);
    ///初始化，其中h_share_model定义了·平面搜索和残差计算
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(), "w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~" << ROOT_DIR << " file opened" << endl;
    else
        cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/x_nav/current_pointcloud", 100000);        //  world系下稠密点云 /x_nav/current_pointcloud
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("cloud_registered_body", 100000);      //  body系下稠密点云
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("cloud_effected", 100000);         //  no used
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/x_nav/global_map", 100000);                    //  no used
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/base_link/odom", 100000);
    // lidar_link 和 imu_link 的 odometry
    pubLidarOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/lidar_link/odom", 100000);
    pubImuOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/imu_link/odom", 100000);

    // sensor-frame paths and velocity
    pubLidarLinkPath    = nh.advertise<nav_msgs::Path>("/x_nav/lidar_link_path", 100000);
    pubImuLinkPath      = nh.advertise<nav_msgs::Path>("/x_nav/imu_link_path", 100000);
    pubLidarOptLinkPath = nh.advertise<nav_msgs::Path>("/x_nav/lidar_opt_link_path", 100000);
    pubImuOptLinkPath   = nh.advertise<nav_msgs::Path>("/x_nav/imu_opt_link_path", 100000);
    pubCurrentVelocity  = nh.advertise<geometry_msgs::Twist>("/x_nav/slam/current_velocity", 100000);

    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/x_nav/localization_path", 1e00000);

    ros::Publisher pubPathUpdate = nh.advertise<nav_msgs::Path>("/x_nav/mapping_path", 100000);                   //  isam更新后的path
    pubGnssPath = nh.advertise<nav_msgs::Path>("gnss_path", 100000);
    pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/x_nav/local_map", 1); // 发布局部关键帧map的特征点云
    pubOptimizedGlobalMap = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/map_global_optimized", 1); // 发布局部关键帧map的特征点云

    // 地面分割Publisher
    pubGroundCloud   = nh.advertise<sensor_msgs::PointCloud2>("/x_nav/current_pointcloud_ground", 100);
    pubNoGroundCloud = nh.advertise<sensor_msgs::PointCloud2>("/x_nav/current_pointcloud_noground", 100);

    // loop clousre
    // 发布闭环匹配关键帧局部map
    pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/icp_loop_closure_history_cloud", 1);
    // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
    pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
    // 发布闭环边，rviz中表现为闭环帧之间的连线
    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/x_nav/loop_closure_constraints", 1);


    // 发布静态 TF
    static tf2_ros::StaticTransformBroadcaster static_br;
    geometry_msgs::TransformStamped static_transformStamped;

    // map -> odom (identity: 平移和四元数均为0)
    static_transformStamped.header.stamp = ros::Time::now();
    static_transformStamped.header.frame_id = mapFrame;
    static_transformStamped.child_frame_id = odometryFrame;
    static_transformStamped.transform.translation.x = 0;
    static_transformStamped.transform.translation.y = 0;
    static_transformStamped.transform.translation.z = 0;
    static_transformStamped.transform.rotation.x = 0;
    static_transformStamped.transform.rotation.y = 0;
    static_transformStamped.transform.rotation.z = 0;
    static_transformStamped.transform.rotation.w = 1;
    static_br.sendTransform(static_transformStamped);

    // 如果启用 input_transform，发布 base_link 到 lidar_link 和 imu_link 的静态 TF
    if (enable_input_transform)
    {
        // base_link -> lidar_link
        geometry_msgs::TransformStamped base_to_lidar_stamped;
        base_to_lidar_stamped.header.stamp = ros::Time::now();
        base_to_lidar_stamped.header.frame_id = base_linkFrame;
        base_to_lidar_stamped.child_frame_id = lidar_linkFrame;
        base_to_lidar_stamped.transform.translation.x = Lidar_T_wrt_Baselink.x();
        base_to_lidar_stamped.transform.translation.y = Lidar_T_wrt_Baselink.y();
        base_to_lidar_stamped.transform.translation.z = Lidar_T_wrt_Baselink.z();
        Eigen::Quaterniond q_lidar_base(Lidar_R_wrt_Baselink);
        base_to_lidar_stamped.transform.rotation.x = q_lidar_base.x();
        base_to_lidar_stamped.transform.rotation.y = q_lidar_base.y();
        base_to_lidar_stamped.transform.rotation.z = q_lidar_base.z();
        base_to_lidar_stamped.transform.rotation.w = q_lidar_base.w();
        static_br.sendTransform(base_to_lidar_stamped);

        // base_link -> imu_link
        geometry_msgs::TransformStamped base_to_imu_stamped;
        base_to_imu_stamped.header.stamp = ros::Time::now();
        base_to_imu_stamped.header.frame_id = base_linkFrame;
        base_to_imu_stamped.child_frame_id = imu_linkFrame;
        base_to_imu_stamped.transform.translation.x = Imu_T_wrt_Baselink.x();
        base_to_imu_stamped.transform.translation.y = Imu_T_wrt_Baselink.y();
        base_to_imu_stamped.transform.translation.z = Imu_T_wrt_Baselink.z();
        Eigen::Quaterniond q_imu_base(Imu_R_wrt_Baselink);
        base_to_imu_stamped.transform.rotation.x = q_imu_base.x();
        base_to_imu_stamped.transform.rotation.y = q_imu_base.y();
        base_to_imu_stamped.transform.rotation.z = q_imu_base.z();
        base_to_imu_stamped.transform.rotation.w = q_imu_base.w();
        static_br.sendTransform(base_to_imu_stamped);
    }

    // gnss
    ros::Subscriber sub_gnss = nh.subscribe(gnss_topic, 200000, gnss_cbk);

    // initialpose 订阅 (localization 模式手动初始化)
    ros::Subscriber sub_initialpose = nh.subscribe("/initialpose", 10, InitialPoseCallBack);

    // slam service: 统一的 slam service 接口
    slam_service = nh.advertiseService("/x_nav/slam/service", &CallbackFrom_slam_service);
    pub_slam_state = nh.advertise<std_msgs::String>("/x_nav/slam/state", 10);

    // ---- 设置 SLAM 模式 (对齐 digitaltwins-x-nav x_slam.cc:82-91) ----
    if (run_mode == "localization")
    {
        Set_localization();
        if (load_map_num == 1)
            load_map(map_dir1);
        else if (load_map_num == 2)
            load_map(map_dir2);
    }
    else  // mapping
    {
        Set_mapping();
    }

    // 回环检测线程
    std::thread loopthread(&loopClosureThread);

    //------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit)
            break;
        ros::spinOnce();

        // localization 模式下周期性发布等待状态
        if (slam_mode == 1 && initializedFlag == 0)
        {
            static ros::Time last_state_pub(0);
            if ((ros::Time::now() - last_state_pub).toSec() > 0.5)
            {
                std_msgs::String sm;
                sm.data = "3";
                pub_slam_state.publish(sm);
                last_state_pub = ros::Time::now();
            }
        }

        /// 在Measure内，储存当前lidar数据及lidar扫描时间内对应的imu数据序列
        if (sync_packages(Measures))
        {
            //第一帧lidar数据
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time; //记录第一帧绝对时间
                p_imu->first_lidar_time = first_lidar_time; //记录第一帧绝对时间
                flg_first_scan = false;
                continue;
            }

            double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time = 0;
            t0 = omp_get_wtime();

            //根据imu数据序列和lidar数据，向前传播纠正点云的畸变, 此前已经完成间隔采样或特征提取
            // feats_undistort 为畸变纠正之后的点云,lidar系
            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();                                               // 前向传播后body的状态预测值
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; // global系 lidar位置

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            // 检查当前lidar数据时间，与最早lidar数据时间是否足够
            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

            /*** Segment the map in lidar FOV (mapping mode only) ***/
            if (slam_mode == 0)
                lasermap_fov_segment(); // 根据lidar在W系下的位置，重新确定局部地图的包围盒角点，移除远端的点

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size(); //当前帧降采样后点数

            /*** initialize the map kdtree ***/
            if (ikdtree.Root_Node == nullptr)
            {
                if (feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for (int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i])); // point转到world系下
                    }
                    // world系下对当前帧降采样后的点云，初始化lkd-tree
                    ikdtree.Build(feats_down_world->points);
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();

            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            // lidar --> imu
            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                     << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << endl;

            if (visulize_IkdtreeMap) // If you need to see map point, change to "if(1)"
            {
                PointVector().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
                publish_map(pubLaserCloudMap);
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();

            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time); //预测、更新
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; // world系下lidar坐标
            geoQuat.x = state_point.rot.coeffs()[0];                                // world系下当前imu的姿态四元数
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            getCurPose(state_point); //   更新transformTobeMapped
            /*** Ground Segmentation (必须在 saveKeyFramesAndFactor 之前，确保关键帧保存时 ground 数据可用) ***/
            if (enable_ground_seg)
            {
                groundSegmentation();
            }

            /*back end*/
            // mapping 模式: 保存关键帧 + 因子图优化 + 增长ikd-Tree
            // localization 模式: 仅使用EKF跟踪，不保存关键帧、不增长地图
            // if (slam_mode == 0)
            // {
                saveKeyFramesAndFactor();
                correctPoses();
                t3 = omp_get_wtime();
                map_incremental();
                t5 = omp_get_wtime();
            // }
            // else
            // {
            //     t3 = omp_get_wtime();
            //     t5 = omp_get_wtime();
            // }

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);
            publish_current_velocity(pubCurrentVelocity);

            // ---- Localization 速度安全检测 ----
            if (slam_mode == 1 && initializedFlag == 2)
                velocitySafetyCheck();
            /******* Publish points *******/
            if (path_en){
                publish_path(pubPath);
                publish_gnss_path(pubGnssPath);                        //   发布gnss轨迹
                publish_path_update(pubPathUpdate, pubLidarOptLinkPath, pubImuOptLinkPath);             //   发布经过isam2优化后的路径
                static int jjj = 0;
                jjj++;
                if (jjj % 100 == 0)
                {
                    publishGlobalMap();             //  发布局部点云特征地图
                }
                // 响应 get_map service 请求
                if (show_globalmap_flag)
                {
                    publishGlobalMap();
                    show_globalmap_flag = false;
                }
            }
            if (scan_pub_en)
                publish_frame_world(pubLaserCloudFull);        //   发布world系下的点云
            if (scan_pub_en && scan_body_pub_en)
                publish_frame_body(pubLaserCloudFull_body);         //  发布imu系下的点云


            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1) / frame_num + (kdtree_incremental_time) / frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time + solve_H_time) / frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n", t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu, aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                         << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << " " << feats_undistort->points.size() << endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(), "w");
        fprintf(fp2, "time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0; i < time_log_counter; i++)
        {
            fprintf(fp2, "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n", T1[i], s_plot[i], int(s_plot2[i]), s_plot3[i], s_plot4[i], int(s_plot5[i]), s_plot6[i], int(s_plot7[i]), int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    startFlag = false;
    loopthread.join(); //  分离线程

    return 0;
}
