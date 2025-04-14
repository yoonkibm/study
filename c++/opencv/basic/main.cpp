#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
    Mat img = imread("test.jpg", IMREAD_COLOR);
    if(img.empty())
    {
        cerr << "Can't call image, Check your directory." << endl;
        return -1;
    }

    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);

    resize(gray_img, gray_img, Size(200, 200));
    rectangle(gray_img, Rect(Point(50, 50), Point(150, 150)), Scalar(0, 255, 0), 2, 8, 0);

    for(int y = 0; y < gray_img.rows; y++){
        for(int x = 0; x < gray_img.cols; x++){
            uchar &pixel = gray_img.at<uchar>(y,x);

            pixel = 255 - pixel;
        }
    }
    
    imwrite("gray_test.jpg", gray_img);

    namedWindow("Gray Test Image", WINDOW_AUTOSIZE);
    imshow("Gray Test Image", gray_img);
    waitKey(0);

    return 0;
}