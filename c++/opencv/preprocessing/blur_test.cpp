#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){

    Mat lena_img = imread("lena.jpg", IMREAD_COLOR);
    if(lena_img.empty()){
        cerr << "Can't call image, Check your directory." << endl;
        return -1;
    }

    Mat gray_img;
    cvtColor(lena_img, gray_img, COLOR_BGR2GRAY);

    Mat blur33, blur55, blur99;
    Size size1(3, 3), size2(5, 5), size3(9, 9);
    blur(gray_img, blur33, size1);
    blur(gray_img, blur55, size2);
    blur(gray_img, blur99, size3);

    Mat gblur33, gblur55, gblur99;
    double sigma = 3.0;
    GaussianBlur(gray_img, gblur33, size1, sigma, sigma);
    GaussianBlur(gray_img, gblur55, size2, sigma, sigma);
    GaussianBlur(gray_img, gblur99, size3, sigma, sigma);

    imshow("Gray Image", gray_img);
    imshow("Blur 3 by 3", blur33);
    imshow("Blur 5 by 5", blur55);
    imshow("Blur 9 by 9", blur99);
    imshow("Gaussian Blur 3 by 3", gblur33);
    imshow("Gaussian Blur 5 by 5", gblur55);
    imshow("Gaussian Blur 9 by 9", gblur99);

    waitKey(0);

    return 0;
}