#include "feature_sort.h"
int feature_dim = 512;
ncnn::Net base_net;

int init_cls_model(std::string params_path, std::string bins_path) {
    const char* param_path_ = params_path.c_str();
    const char* bin_path_ = bins_path.c_str();
    {
        int ret = base_net.load_param(param_path_);

        if (ret != 0)
        {
            std::cout << "ret= " << ret << std::endl;
            fprintf(stderr, "YoloV5Ncnn, load_param failed");
            return -301;
        }
    }
    // init bin
    {
        int ret = base_net.load_model(bin_path_);
        if (ret != 0)
        {
            fprintf(stderr, "YoloV5Ncnn, load_model failed");
            return -301;
        }
    }
    return 0;
}

bool getRectsFeature(const cv::Mat& img, DETECTIONS& d) {
    std::vector<cv::Mat> mats;
    for (DETECTION_ROW& dbox : d) {
        cv::Rect rc = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
            int(dbox.tlwh(2)), int(dbox.tlwh(3)));
        rc.x -= (rc.height * 0.5 - rc.width) * 0.5;
        rc.width = rc.height * 0.5;
        rc.x = (rc.x >= 0 ? rc.x : 0);
        rc.y = (rc.y >= 0 ? rc.y : 0);
        rc.width = (rc.x + rc.width <= img.cols ? rc.width : (img.cols - rc.x));
        rc.height = (rc.y + rc.height <= img.rows ? rc.height : (img.rows - rc.y));

        cv::Mat mattmp = img(rc).clone();
        cv::resize(mattmp, mattmp, cv::Size(64, 128));
        mats.push_back(mattmp);
    }
    int count = mats.size();

    float norm[3] = { 0.229, 0.224, 0.225 };
    float mean[3] = { 0.485, 0.456, 0.406 };
    for (int i = 0; i < count; i++)
    {
        ncnn::Mat in_net = ncnn::Mat::from_pixels(mats[i].data, ncnn::Mat::PIXEL_BGR2RGB, 64, 128);
        in_net.substract_mean_normalize(mean, norm);

        ncnn::Mat out_net;
        ncnn::Extractor ex = base_net.create_extractor();
        ex.set_light_mode(true);
        ex.set_num_threads(4);
        //    if (toUseGPU) {  // 消除提示
        //        ex.set_vulkan_compute(toUseGPU);
        //    }
        ex.input("x", in_net);
        ex.extract("y", out_net);

        cv::Mat tmp(out_net.h, out_net.w, CV_32FC1, (void*)(const float*)out_net.channel(0));

        const float* tp = tmp.ptr<float>(0);
        for (int j = 0; j < feature_dim; j++)
        {
            d[i].feature[j] = tp[j];
        }

    }
    return true;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat& frame, const  std::vector<Object>& outs, DETECTIONS& d)
{
    for (const Object& info : outs)
    {
        //目标检测 代码的可视化
//        drawPred(info.label, info.prob, info.x, info.y,info.w+info.x, info.y+info.h, frame);

        get_detections(DETECTBOX(info.rect.x, info.rect.y, info.rect.width, info.rect.height),
            info.prob, d);
    }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    std::string label = std::to_string(classId);

    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    //top = max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - round(1.5 * labelSize.height)), cv::Point(left + round(1.5 * labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
}

void get_detections(DETECTBOX box, float confidence, DETECTIONS& d)
{
    DETECTION_ROW tmpRow;
    tmpRow.tlwh = box;//DETECTBOX(x, y, w, h);

    tmpRow.confidence = confidence;
    d.push_back(tmpRow);
}



//void deepsort_draw(const cv::Mat& imgs, std::vector<Object> outs_obj, const int nn_budget, const float max_cosine_distance) {
//    tracker mytracker(max_cosine_distance, nn_budget);
//    DETECTIONS detections;
//    cv::Mat image = imgs.clone();
//    postprocess(image, outs_obj, detections);
//    if (getRectsFeature(image, detections))
//    {
//        mytracker.predict();
//        mytracker.update(detections);
//
//        std::vector<RESULT_DATA> result;
//        for (Track& track : mytracker.tracks) {
//            if (!track.is_confirmed() || track.time_since_update > 1) continue;
//            result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
//        }
//        for (unsigned int k = 0; k < detections.size(); k++)
//        {
//            DETECTBOX tmpbox = detections[k].tlwh;
//            cv::Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
//            //                cv::rectangle(frame, rect, cv::Scalar(0,0,255), 4);
//                            // cvScalar的储存顺序是B-G-R，CV_RGB的储存顺序是R-G-B
//
//            for (unsigned int k = 0; k < result.size(); k++)
//            {
//                DETECTBOX tmp = result[k].second;
//                cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
//                rectangle(image, rect, cv::Scalar(255, 255, 0), 2);
//
//                std::string label = cv::format("%d", result[k].first);
//                cv::putText(image, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
//            }
//        }
//    }
//}

