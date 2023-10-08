#pragma once
#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <time.h>
#include "net.h"
#include "layer.h"
#include <float.h>
#include <stdio.h>
#include <fstream>

#define MAX_STRIDE 64

static ncnn::Net yolov5;
static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

#define mins(a,b) (((a)<(b)) ? (a):(b))
#define maxs(a,b) (((a)>(b)) ? (a):(b))

struct Object
{
	cv::Rect_<float> rect;
	int label;
	float prob;
};


//static const char* class_names[] = {
//		"helmet", "no - helmet", "no - vest", "person", "reflective - vest"
//};

static const char* class_names[] = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush"
};

class detectNet
{
public:
	detectNet();
	~detectNet();
	int init1(std::string net_bin, std::string net_para);

	Object get_one_Object(cv::Mat& img);

	int get_one_label(Object objects);

	void get_video_result(std::string videoPath, std::string saveVideoPath, int is_save);

	int detect(cv::Mat img, std::vector<Object>& objects);

	void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects);

	void send_json(cv::Mat img, std::string label, std::string level);

private:

	static inline float intersection_area(const Object& a, const Object& b);

	static void qsort_descent_inplace(std::vector<Object>& faceobjects);

	static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);

	static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false);

	static inline float sigmoid(float x);

	void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects);
};




