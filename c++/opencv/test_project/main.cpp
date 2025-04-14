#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    cv::Mat img = cv::imread("test.jpg", cv::IMREAD_COLOR);

    if(img.empty())
    {
        std::cerr << "Can't call image, Check your directory." << std::endl;
        return -1;
    }

    cv::namedWindow("Test Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Test Image", img);

    cv::waitKey(0);
    return 0;
}