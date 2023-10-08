
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
#include "feature_sort.h"


//Deep SORT parameter
const int nn_budget = 100;
const float max_cosine_distance = 0.2;


int main()
{
    tracker mytracker(max_cosine_distance, nn_budget);
    // Open a video file or an image file or a camera stream.
    std::string str, outputFile;
    cv::VideoCapture cap;
    cv::VideoWriter video;
    cv::Mat frame, blob;
    //DeepSort deepsort;
    const char* imagepath = "E:\\ubuntu\\2\\12.mp4";



    std::string params = "D:\\deepsort\\ShuffleNet_deepsort\\pth2onnx\\best.param";
    std::string bins = "D:\\deepsort\\ShuffleNet_deepsort\\pth2onnx\\best.bin";

    init_cls_model(params, bins);

    //base_net.load_param("D:\\deepsort\\ShuffleNet_deepsort\\pth2onnx\\best.param");
    //base_net.load_model("D:\\deepsort\\ShuffleNet_deepsort\\pth2onnx\\best.bin");

    outputFile = "./deep_sort2.avi";
    // Open the video file      anquanmao.mp4   human.flv    ship.mp4
    bool readRes = cap.open(imagepath);  //ship.mp4
    std::cout << "read: " << readRes << std::endl;

    // Get the video writer initialized to save the output video
    bool writeRes = video.open(outputFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 12.0,
        cv::Size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))));
    std::cout << "write: " << writeRes << std::endl;
    // Create a window
    static const  std::string kWinName = "Multiple Object Tracking";
    namedWindow(kWinName, cv::WINDOW_NORMAL);
    int count = 0;
    std::vector<Object> outs;

    std::string detect_bins = "./models/yolov5s.bin";
    std::string detect_paras = "./models/yolov5s.param";
    void* p_algorithmss;
    detectNet* tmp_detect;
    tmp_detect = (detectNet*)(p_algorithmss);
    tmp_detect->init1(detect_bins, detect_paras);
    while (readRes)
    {
        // get frame from the video
        cap >> frame;

        // Stop the program if reached end of video
        if (frame.empty())
        {
            std::cout << "Done processing !!!" << std::endl;
            std::cout << "Output file is stored as " << outputFile << std::endl;
            cv::waitKey(3000);
            break;
        }

        //detect_yolov5(frame, outs);
        tmp_detect->detect(frame, outs);


        DETECTIONS detections;
        postprocess(frame, outs, detections);

        std::cout << "Detections size:" << detections.size() << std::endl;
        if (getRectsFeature(frame, detections))
        {
            mytracker.predict();
            mytracker.update(detections);

            std::vector<RESULT_DATA> result;
            for (Track& track : mytracker.tracks) {
                if (!track.is_confirmed() || track.time_since_update > 1) continue;
                result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
            }
            for (unsigned int k = 0; k < detections.size(); k++)
            {
                DETECTBOX tmpbox = detections[k].tlwh;
                cv::Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
                //                cv::rectangle(frame, rect, cv::Scalar(0,0,255), 4);
                                // cvScalar�Ĵ���˳����B-G-R��CV_RGB�Ĵ���˳����R-G-B

                for (unsigned int k = 0; k < result.size(); k++)
                {
                    DETECTBOX tmp = result[k].second;
                    cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
                    rectangle(frame, rect, cv::Scalar(255, 255, 0), 2);

                    std::string label = cv::format("%d", result[k].first);
                    cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
                }
            }
        }

        // Write the frame with the detection boxes
        cv::Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        video.write(detectedFrame);

        count++;
        std::cout << "frame count: " << count << std::endl;

        imshow(kWinName, frame);
        cv::waitKey(3);  //��ʱ20ms
    }

    cap.release();
    video.release();


    return 0;
}



