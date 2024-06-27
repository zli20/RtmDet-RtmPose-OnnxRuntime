#include <iostream>


#include "opencv2/opencv.hpp"

#include "rtmpose_utils.h"
#include "rtmpose_onnxruntime.h"
#include "rtmdet_onnxruntime.h"
#include "rtmpose_tracker_onnxruntime.h"

std::vector<std::pair<int, int>> coco_17_joint_links = {
	{0,1},{0,2},{1,3},{2,4},{5,7},{7,9},{6,8},{8,10},{5,6},{5,11},{6,12},{11,12},{11,13},{13,15},{12,14},{14,16}
};

int main()
{
	std::string rtm_detnano_onnx_path = "";
	std::string rtm_pose_onnx_path = "";

	rtm_detnano_onnx_path = "./models/rtmdet_nano.onnx";
	rtm_pose_onnx_path = "./models/rtmpose-m-mpii.onnx";

	std::string img_path = "./images/z.jpg";
	// 加载图像

	//
	RTMPoseTrackerOnnxruntime rtmpose_tracker_onnxruntime(rtm_detnano_onnx_path, rtm_pose_onnx_path);

	cv::VideoCapture video_reader(1);
	int frame_num = 0;
	DetectBox detect_box;
	while (video_reader.isOpened())
	{
		cv::Mat frame;
		video_reader >> frame;

		//std::string img_path = "images/z.jpg";
		//frame = cv::imread(img_path, cv::IMREAD_COLOR);

		if (frame.empty())
			break;
		auto start_time = std::chrono::high_resolution_clock::now();
		std::pair<DetectBox, std::vector<PosePoint>> inference_box= rtmpose_tracker_onnxruntime.Inference(frame);

		auto end_time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
		std::cout << "Inference time: " << duration.count() << " ms" << std::endl;


		DetectBox detect_box = inference_box.first;
		std::vector<PosePoint> pose_result = inference_box.second;

		cv::rectangle(
			frame,
			cv::Point(detect_box.left, detect_box.top),
			cv::Point(detect_box.right, detect_box.bottom),
			cv::Scalar{ 255, 0, 0 },
			2);

		for (int i = 0; i < pose_result.size(); ++i)
		{
			if (pose_result[i].score > 0.6 &&
				pose_result[i].x >= 0 && pose_result[i].x < frame.cols &&
				pose_result[i].y >= 0 && pose_result[i].y < frame.rows) {
				cv::circle(frame, cv::Point(pose_result[i].x, pose_result[i].y), 1, cv::Scalar(0, 0, 255), 5, cv::LINE_AA);
			}


		}

		//for (int i = 0; i < coco_17_joint_links.size(); ++i)
		//{
		//	std::pair<int, int> joint_links = coco_17_joint_links[i];
		//	if (pose_result[joint_links.first].score > 0.6 &&
		//		pose_result[joint_links.second].score > 0.6 &&
		//		pose_result[joint_links.first].x >= 0 && pose_result[joint_links.first].x < frame.cols &&
		//		pose_result[joint_links.first].y >= 0 && pose_result[joint_links.first].y < frame.rows &&
		//		pose_result[joint_links.second].x >= 0 && pose_result[joint_links.second].x < frame.cols &&
		//		pose_result[joint_links.second].y >= 0 && pose_result[joint_links.second].y < frame.rows) {
		//		cv::line(
		//			frame,
		//			cv::Point(pose_result[joint_links.first].x, pose_result[joint_links.first].y),
		//			cv::Point(pose_result[joint_links.second].x, pose_result[joint_links.second].y),
		//			cv::Scalar{ 0, 255, 0 },
		//			2,
		//			cv::LINE_AA);
		//	}
		//}

		imshow("RTMPose", frame);
		cv::waitKey(1);
	}

	video_reader.release();
	cv::destroyAllWindows();

	return 0;
}
