#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void CalcHist(const Mat& src, Mat& hist, int histSize, const float* histRange)
{
    int channels = src.channels();

    if(channels == 1){
        hist = Mat::zeros(histSize, 1, CV_32F);

        for(int i = 0; i < src.rows; i++){
            for(int j = 0; j < src.cols; j++){
                uchar pixel = src.at<uchar>(i, j);
                int bin = pixel * histSize / 256;
                hist.at<float>(bin, 0)++;
            }
        }
    }

    else if(channels == 3){
        hist = Mat::zeros(histSize, 3, CV_32F);

        for(int i = 0; i < src.rows; i++){
            for(int j = 0; j < src.cols; j++){
                Vec3b pixel = src.at<Vec3b>(i, j);
                for(int c = 0; c < 3; c++){
                    int bin = pixel[c] * histSize / 256;
                    hist.at<float>(bin, c)++;
                }
            }
        }
    }
    else{
        cerr << "The number of channels is not supported." << endl;
    }
}

void DrawHistImage(const Mat& hist, Mat& hist_image, int histSize, const cv::Scalar& lineColor) {
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double) hist_w / histSize);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255)); // 흰 배경

    // 정규화 (그래프가 잘 보이도록)
    Mat hist_norm;
    normalize(hist, hist_norm, 0, histImage.rows, NORM_MINMAX);

    for (int i = 1; i < histSize; i++) {
        line(histImage,
             Point(bin_w * (i - 1), hist_h - cvRound(hist_norm.at<float>(i - 1))),
             Point(bin_w * i,       hist_h - cvRound(hist_norm.at<float>(i))),
             lineColor, 2); // 검정색 선
    }

    hist_image = histImage;
}



int main(){
    Mat room_img = imread("room.jpg", IMREAD_COLOR);
    if(room_img.empty()){
        cerr << "Can't call image, Check your directory." << endl;
        return -1;
    }

    Mat img;
    cvtColor(room_img, img, COLOR_BGR2GRAY);

    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat hist;

    // CalcHist(img, hist, histSize, histRange);
    CalcHist(img, hist, histSize, histRange);

    // Mat b_hist = hist.col(0).clone();
    // Mat g_hist = hist.col(1).clone();
    // Mat r_hist = hist.col(2).clone();

    // Mat b_img, g_img, r_img;
    // DrawHistImage(b_hist, b_img, histSize, Scalar(255, 0, 0)); // 이 함수는 grayscale 버전이어야 해
    // DrawHistImage(g_hist, g_img, histSize, Scalar(0, 255, 0));
    // DrawHistImage(r_hist, r_img, histSize, Scalar(0, 0, 255));
    Mat hist_img;
    DrawHistImage(hist, hist_img, histSize, Scalar(0, 0, 0));

    Mat equalized;
    equalizeHist(img, equalized);

    Mat hist_after;
    CalcHist(equalized, hist_after, histSize, histRange);

    Mat hist_after_img;
    DrawHistImage(hist_after, hist_after_img, histSize, Scalar(0, 0, 0));

    imshow("Source Image", img);
    imshow("Equalized Image", equalized);
    imshow("Gray Scale Histogram", hist_img);
    imshow("Equalized Histogram", hist_after_img);
    // imshow("Blue Histogram", b_img);
    // imshow("Green Histogram", g_img);
    // imshow("Red Histogram", r_img);

    waitKey(0);

    return 0;
}