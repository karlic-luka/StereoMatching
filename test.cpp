#include <stdio.h>
#include <opencv2/opencv.hpp>
// using namespace cv;
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    cv::Mat image, image2;
    image = cv::imread(argv[1]);
    image2 = cv::imread(argv[2]);
    if (!image.data || !image2.data)
    {
        printf("No image data \n");
        return -1;
    }
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Display Image2", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image);
    cv::imshow("Display Image2", image2);
    cv::waitKey(0);
    return 0;
}