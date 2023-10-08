
#include "layer.h"
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>
#include <iostream>

//#include "./DeepAppearanceDescriptor/deepsort.h"
#include "KalmanFilter/tracker.h"
#include "dataType.h"
#include "detect.h"

#include <math.h>

//ncnn::Net base_net;

//int feature_dim = 512;

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat& frame, const  std::vector<Object>& outs, DETECTIONS& d);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

// Get the names of the output layers
std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net);

void get_detections(DETECTBOX box, float confidence, DETECTIONS& d);

int init_cls_model(std::string params_path, std::string bins_path);

bool getRectsFeature(const cv::Mat& img, DETECTIONS& d);

//void deepsort_draw(const cv::Mat& imgs, std::vector<Object> outs_obj, const int nn_budget, const float max_cosine_distance);
