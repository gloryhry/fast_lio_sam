#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

std::string world_frame_id;
tf::TransformListener*  tfListener;
std::string odom_frame_id, lidar_frame_id, slam_topic;
tf::StampedTransform trans_result;
bool init= false;
tf::TransformBroadcaster *tf_br;

void callback(const nav_msgs::OdometryConstPtr slam_msg)
{
    init = true;

    // TF
    tf::Quaternion slam_quaternion;
    tf::quaternionMsgToTF(slam_msg->pose.pose.orientation, slam_quaternion);
    tf::Point slam_translate;
    tf::pointMsgToTF(slam_msg->pose.pose.position, slam_translate);
    tf::Transform slam_T;
    slam_T.setRotation(slam_quaternion);
    slam_T.setOrigin(slam_translate);

    // get odom to livox
    tf::StampedTransform odom2lidar_transform;
    try 
    {
        tfListener->waitForTransform(odom_frame_id,lidar_frame_id, slam_msg->header.stamp, ros::Duration(0.5));
        tfListener->lookupTransform(odom_frame_id, lidar_frame_id, slam_msg->header.stamp, odom2lidar_transform);
    } 
    catch(tf::TransformException &exception) {
        ROS_WARN_STREAM(exception.what());         
    }
    tf::Transform odom_T;
    odom_T.setRotation(odom2lidar_transform.getRotation());
    odom_T.setOrigin(odom2lidar_transform.getOrigin());
    tf::Transform fix_T;
    fix_T = slam_T * odom_T.inverse();

    trans_result = tf::StampedTransform(fix_T, slam_msg->header.stamp, world_frame_id, odom_frame_id);
    trans_result.stamp_ = slam_msg->header.stamp;
    tf_br->sendTransform(trans_result);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "fix_odom");
    ros::NodeHandle nh, private_nh("~");

    tfListener = new tf::TransformListener;
    tf_br = new tf::TransformBroadcaster;

    private_nh.param<std::string>("odom_frame_id", odom_frame_id, "odom");
    private_nh.param<std::string>("lidar_frame_id",lidar_frame_id, "rslidar");
    private_nh.param<std::string>("slam_odom", slam_topic, "Odometry");
    private_nh.param<std::string>("world_frame_id", world_frame_id, "world");
    ROS_INFO_STREAM("odom frame id: " << odom_frame_id);
    ROS_INFO_STREAM("world frame id: " << world_frame_id);

    ros::Subscriber slam_odom_sub = nh.subscribe(slam_topic,10,&callback,ros::TransportHints().tcpNoDelay());
    ros::spin();
    // ros::Rate rate(50);
    // while(ros::ok())
    // {
    //     ros::spinOnce();
    //     if(init)
    //     {
    //         trans_result.stamp_ = ros::Time::now();
    //         tf_br->sendTransform(trans_result);
    //     }
    //     rate.sleep();
    // }
    return 0;
}