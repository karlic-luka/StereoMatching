// #include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
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

// cv::Mat &scanImageAndApplyCost(cv::Mat &image1, cv::Mat &image2, cv::Mat &destination, const uchar *const table)
// {
//     //accept only char type matrices
//     CV_Assert(image1.depth() == CV_8U);
//     //dolazi ce samo grayscale slike
//     //Since we are reading the image from disk and imread uses the create method, we can simply loop over all pixels using simple pointer arithmetic that does not require a multiplication.
//     int numberOfRows = image.rows;
//     int numberOfColumns = image.cols;

//     uchar *p;
//     for (int i = 0; i < numberOfRows; ++i)
//     {
//         p1 = image1.ptr<uchar>(i);
//         p2 = image2.ptr<uchar>(i);
//         for (int j = 0; j < numberOfColumns; ++j)
//         {
//             destination.at<>(i, j, 0) = ADlookUpTable(p1[j], p2[j]);
//         }
//     }
// }
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("usage: DisplayImage.out <Image1_Path> <Image2_Path> d_min d_max\n"); //trenutno cu ih sam napisati u mainu, kasnije cu parsirati preko komandne linije
        return -1;
    }
    cv::Mat image1, image2, gray1, gray2;
    image1 = cv::imread(argv[1]);
    image2 = cv::imread(argv[2]);
    int imageWidth = image1.size[0];
    int imageHeight = image1.size[1];
    assert(image1.size == image2.size); //kasnije jos dodati malo teksta
    int d_min = 100;
    int d_max = 150;

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
    // cout << image1.size() << endl;
    // cout << ADlookUpTable[5][250] << endl;
    // cout << ADlookUpTable[255][255] << endl;
    // cout << ADlookUpTable[0][0] << endl;

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
    // cout << image1.size[0] << endl;
    //inicijalizaciju prvih d_min stupaca OUTPUT mape mogu uciniti preko "view"-a na tu matricu
    //buduci da se zapravo dijeli header
    // cv::Mat initialColumns(image1, cv::Range::all(), cv::Range(0, d_min));
    // cv::Mat initialColumns(d_min, imageHeight, CV_8UC1);
    // cout << initialColumns.size() << endl;
    // cv::Mat initialColumns(gray1, cv::Range::all(), cv::Range(0, d_min));
    // cv::Mat outputMap(imageHeight, imageWidth, CV_8UC1);
    // cv::Rect roi(0, 0, d_min, imageHeight);
    // cv::Mat sourceWindow = gray1(roi);
    // cv::Mat targetWindow = outputMap(roi);
    // sourceWindow.copyTo(targetWindow);
    // showImage(outputMap, "output");
    // cv::copyMakeBorder(initialColumns, outputMap, 0, 0, 0, imageWidth - d_min, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));
    // cout << outputMap.size() << endl;
    // initialColumns = gray1(cv::Range::all(), cv::Range(0, d_min)).clone();
    // cout << "Initial columns: " << initialColumns.size() << endl;
    // showImage(initialColumns, "columns");
    // outputMap(cv::Range::all(), cv::Range(0, d_min)) = gray1(cv::Range::all(), cv::Range(0, d_min));
    // showImage(gray1, "gray1");
    // cv::copyMakeBorder(gray1,grayPadded, dis)
    // cout << "Time to calculate per one disparity hypothese: " << t << endl;

    int dimensions[3] = {imageHeight, imageWidth, d_max - d_min}; //ovdje cu spremati rezultate operacija po disparitetima
    cv::Mat results(3, dimensions, CV_32SC1, cv::Scalar(0));
    // cout << results.at<int>(3, 4, 5) << endl;

    int disparity = 0;
    uchar *p1, *p2;
    for (int i = 0; i < imageWidth; ++i)
    {
        //PAZI: OVO JE REDAK --> Y-KOORDINATA
        p1 = gray1.ptr<uchar>(i);
        p2 = gray2.ptr<uchar>(i);
        // p3 = dimensions.ptr<uchar>(i);
        for (int j = 0; j < imageHeight; ++j)
        {
            //X KOORDINATA
            if (j < disparity)
            {
                results.at<int>(i, j, disparity) = p1[j];
            }
            else
            {
                auto firstIntensity = p1[j];
                auto secondIntensity = p2[j - disparity];
                results.at<int>(i, j, disparity) = ADlookUpTable[(int)firstIntensity][(int)secondIntensity];
            }
        }
    }
    cout << results.at<int>(374, 449, 0) << endl;
    //provjera
    cv::Mat test;
    cv::absdiff(gray1, gray2, test);
    // cout << results.at<int>(3, 4, 0) << endl;
    // cout << test.at<int>(150, 200) << endl;

    // outputMap = results(cv::Range::all, cv::Range::all, 0);
    // uchar *p = gray1.ptr<uchar>(2);
    // cout << (int)p[3] << endl;
    cout << "gray1(450, 375): " << (int)gray1.ptr<uchar>(374)[449] << endl;
    cout << "gray2(450, 375): " << (int)gray2.ptr<uchar>(374)[449] << endl;
    showImage(gray1, "gray1");
    showImage(gray2, "gray2");
    destroyWindows();
}
