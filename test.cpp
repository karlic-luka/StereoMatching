#include <stdio.h>
#include <opencv2/opencv.hpp>
// using namespace cv;
using namespace std;

void showImage(const cv::Mat &image, std::string windowName)
{
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, image);
}

void destroyWindows()
{
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int absoluteDifference(const int &number1, const int &number2)
{
    return number1 >= number2 ? (number1 - number2) : (number2 - number1);
}

int ADlookUpTable[256][256]; //trenutno globalna varijabla, ali bit ce privatna varijabla u objektnoj paradigmi za pripadni matching cost

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("usage: DisplayImage.out <Image1_Path> <Image2_Path>\n");
        return -1;
    }
    cv::Mat image1, image2, gray1, gray2;
    image1 = cv::imread(argv[1]);
    image2 = cv::imread(argv[2]);
    cv::cvtColor(image1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);
    assert(gray1.channels() == gray2.channels() == 1);

    if (!image1.data || !image2.data)
    {
        printf("No image data \n");
        return -1;
    }
    //provjeravam radi li AD
    // cout << absoluteDifference(5, 3) << " " << absoluteDifference(3, 5) << endl;
    //----------------------------------LOOKUP TABLICA ZA ABSOLUTE DIFFERENCE------
    for (int row = 0; row < 255; row++)
    {
        for (int column = row; column < 255; column++)
        {
            ADlookUpTable[row][column] = absoluteDifference(row, column);
            ADlookUpTable[column][row] = absoluteDifference(row, column);
        }
    }
    cout << image1.size() << endl; //width x height
    cout << ADlookUpTable[150][234] << endl;
    cout << ADlookUpTable[5][5] << endl;
    cout << ADlookUpTable[255][255] << endl;
    cout << ADlookUpTable[0][0] << endl;
    //---------------------------------------------------------------------------------
    // cv::Mat roi(image1, cv::Range(50, 52), cv::Range(10, 15)); //range of interest
    // cout << "ROI(numpy" << endl;
    // cout << cv::format(roi, cv::Formatter::FMT_NUMPY) << endl;

    //prikazi obje slike
    // showImage(image1, "Image1");
    // showImage(image2, "Image2");
    // showImage(gray1, "Left");
    // showImage(gray2, "Right");
    //cout << gray1.channels() << endl;

    destroyWindows();
}
