#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <ros/package.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include "ros_libfacedetection/FaceObject.h"
#include "ros_libfacedetection/Euler.h"
#include "facedetection/facedetectcnn.h"

#define DEG_TO_RAD (M_PI/180.0)

cv_bridge::CvImagePtr cv_ptr;
ros_libfacedetection::FaceObject faceMsg;
ros_libfacedetection::Euler eulerMsg;
ros::Publisher face_pub, pose_pub;

std::string node_name = "";
bool display_output = true;
bool estimate_pose = true;
int detector_width  = 160;
int detector_height = 120;
double prob_threshold = 0.75;
int* detector_results = NULL;
unsigned char* pBuffer = NULL;

std::vector<cv::Point3d> model_points;
std::vector<cv::Point2d> image_points;

using namespace sensor_msgs;
using namespace message_filters;

void callback(const ImageConstPtr& msg, const CameraInfoConstPtr& cam_info)
{
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    cv::Mat det_input;

    cv::resize(cv_ptr->image, det_input, cv::Size(detector_width, detector_height));
    detector_results = facedetect_cnn(pBuffer, (unsigned char*)(det_input.ptr(0)), detector_width, detector_height, (int)det_input.step);

    for(int i = 0; i < (detector_results ? *detector_results : 0); i++)
    {
        short * p = ((short*)(detector_results+1))+142*i;
        int confidence = p[0];
        int x = p[1];
        int y = p[2];
        int w = p[3];
        int h = p[4];
        int shift = w * 0.1;
        x = (x - shift) < 0 ? 0: x - shift;
        y = (y - shift) < 0 ? 0: y - shift;
        w = w + shift * 2;
        h = h + shift * 2;

        x = int(x  * 1.0 / detector_width * cam_info->width);
        y = int(y  * 1.0 / detector_height * cam_info->height);
        w = int(w * 1.0 / detector_width * cam_info->width);
        h = int(h * 1.0 / detector_height * cam_info->height);
        w = (w > cam_info->width) ? cam_info->width : w;
        h = (h > cam_info->height) ? cam_info->height : h;

        char sScore[256];
        snprintf(sScore, 256, "%d", confidence);

        if(confidence > prob_threshold){

            faceMsg.header.seq++;
            faceMsg.header.frame_id = cam_info->header.frame_id;
            faceMsg.header.stamp = ros::Time::now();
            faceMsg.probability = confidence;
            faceMsg.boundingbox.position.x = x;
            faceMsg.boundingbox.position.y = y;
            faceMsg.boundingbox.size.x = w;
            faceMsg.boundingbox.size.y = h;

            for (uint8_t i=0; i<5; i++){
              faceMsg.landmark[i].x = cam_info->width*(p[5+i*2])/detector_width;
              faceMsg.landmark[i].y = cam_info->height*(p[5 + i*2 + 1])/detector_height;
              if (estimate_pose) image_points.push_back(cv::Point(faceMsg.landmark[i].x,faceMsg.landmark[i].y));
            }
            face_pub.publish(faceMsg);

            if (estimate_pose){
              // TODO: read distortion model from camera_info
              double focal_length = cv_ptr->image.cols;
              cv::Point2d center = cv::Point2d(cv_ptr->image.cols/2,cv_ptr->image.rows/2);
              cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0,
                                       focal_length, center.y, 0, 0, 1);
              cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type);
              //
              cv::Mat rotation_vector;
              cv::Mat translation_vector;
              cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                          rotation_vector, translation_vector, false, cv::SOLVEPNP_EPNP );
              image_points.clear();
              cv::Mat rotation_matrix; // 3 x 3 R
              cv::Rodrigues(rotation_vector, rotation_matrix);
              cv::Mat pose_matrix = cv::Mat(3, 4, CV_64FC1); // 3 x 4 R | T
              cv::hconcat(rotation_matrix, translation_vector, pose_matrix);
              cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
              cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
              cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);
              cv::Mat euler_angles = cv::Mat(3, 1, CV_64FC1);
              cv::decomposeProjectionMatrix(pose_matrix, out_intrinsics, out_rotation, out_translation,
                                            cv::noArray(), cv::noArray(), cv::noArray(), euler_angles);
              eulerMsg.roll  = euler_angles.at<double>(2)*DEG_TO_RAD;
              eulerMsg.pitch = euler_angles.at<double>(0)*DEG_TO_RAD;
              eulerMsg.yaw   = euler_angles.at<double>(1)*DEG_TO_RAD;
              pose_pub.publish(eulerMsg);
            }

            if (display_output){
              cv::Mat result_image = cv_ptr->image.clone();

              if(x + w >= cam_info->width) w = cam_info->width - x;
              if(y + h >= cam_info->height) h = cam_info->height - y;

              cv::putText(result_image, sScore, cv::Point(x, y-8), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                          cv::Scalar(0, 0, 255), 1);
              cv::rectangle(result_image, cv::Rect(x, y, w, h),
                          cv::Scalar(0, 0, 255), 1);

              for (uint8_t i=0; i<5; i++){
                cv::putText(result_image, std::to_string(i).c_str(),
                            cv::Point(faceMsg.landmark[i].x,faceMsg.landmark[i].y),
                            cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(0, 0, 255), 1);
              }

              cv::imshow(node_name, result_image);
              cv::waitKey(1);
            }
        }
    }
  }
  catch (cv_bridge::Exception& e) {
    ROS_ERROR("CV bridge exception: %s", e.what());
    return;
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "face_detection_node");
  ros::NodeHandle nhLocal("~");
  ros::NodeHandle nh;
  node_name = ros::this_node::getName();

  nhLocal.param("probability_threshold", prob_threshold, 0.75);
  nhLocal.param("detector_width", detector_width, 160);
  nhLocal.param("detector_height", detector_height, 120);
  nhLocal.param("display_output", display_output, true);
  nhLocal.param("estimate_pose", estimate_pose, true);

  face_pub = nh.advertise<ros_libfacedetection::FaceObject>(node_name+"/faces", 10);

  if (estimate_pose){
    pose_pub = nh.advertise<ros_libfacedetection::Euler>(node_name+"/euler", 10);
  }

  message_filters::Subscriber<Image> image_sub(nh, "/camera/image_raw", 1);
  message_filters::Subscriber<CameraInfo> info_sub(nh, "/camera/camera_info", 1);
  TimeSynchronizer<Image, CameraInfo> sync(image_sub, info_sub, 10);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  pBuffer = (unsigned char *)malloc(0x20000);
  if(!pBuffer) {
    ROS_ERROR("Can not allocate buffer.");
    return -1;
  }

  // Satya Mallick's facial landmark model:
  model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // 0 - Right eye right corner
  model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // 1 - Left eye left corner
  model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // 2 - Nose tip
  model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));       // 3 - Right mouth corner
  model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // 4 - Left Mouth corner

  while (ros::ok()) {
    ros::spinOnce();
  }

  return 0;
}
