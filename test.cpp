// #include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>

// using namespace cv;
using namespace std;
using namespace std::chrono;
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

// int absoluteDifference(const int &number1, const int &number2)
// {
//     return number1 >= number2 ? (number1 - number2) : (number2 - number1);
// }
auto absoluteDifference = [&](int first, int second) { return first >= second ? (first - second) : (second - first); }; //pripaziti na LAMBDU
int ADlookUpTable[256][256];                                                                                            //trenutno globalna varijabla, ali bit ce privatna varijabla u objektnoj paradigmi za pripadni matching cost

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
    int d_min = 0;
    int d_max = 3;
    int window = 3;

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
    auto start = high_resolution_clock::now();
    for (int row = 0; row < 255; row++)
    {
        for (int column = row; column < 255; column++)
        {
            ADlookUpTable[row][column] = absoluteDifference(row, column);
            ADlookUpTable[column][row] = absoluteDifference(row, column);
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    auto tableInit = duration;
    cout << "Look-up-table init: " << duration.count() << " microseconds." << endl;
    //---------------------------------------------------------------------------------
    // cv::Mat roi(image1, cv::Range(50, 52), cv::Range(10, 15)); //range of interest
    // cout << "ROI(numpy" << endl;
    // cout << cv::format(roi, cv::Formatter::FMT_NUMPY) << endl;

    int dimensions[3] = {imageHeight, imageWidth, d_max - d_min}; //ovdje cu spremati rezultate operacija po disparitetima
    // cv::Mat results(3, dimensions, CV_32SC1, cv::Scalar(0));
    // cout << results.at<int>(3, 4, 5) << endl;

    //dio koda koji računa cost za svaki piksel i svaki disparity --> BEZ KUMULATIVNOG TROSKA!
    int count = 0;
    uchar *p1, *p2, *p3;

    // Get starting timepoint
    vector<unique_ptr<cv::Mat>> results;
    start = high_resolution_clock::now();

    for (int disparity = d_min; disparity <= d_max; disparity++)
    {
        unique_ptr<cv::Mat> current(new cv::Mat(imageHeight, imageWidth, CV_8UC1));

        for (int i = 0; i < imageWidth; ++i)
        {
            //PAZI: OVO JE REDAK --> Y-KOORDINATA
            p1 = gray1.ptr<uchar>(i);
            p2 = gray2.ptr<uchar>(i);
            p3 = current->ptr<uchar>(i);
            for (int j = 0; j < imageHeight; ++j)
            {
                //X KOORDINATA
                if (j < disparity)
                {
                    p3[j] = p1[j];
                }
                else
                {
                    auto firstIntensity = p1[j];
                    auto secondIntensity = p2[j - disparity];
                    p3[j] = ADlookUpTable[(int)firstIntensity][(int)secondIntensity];
                    // results.at<int>(i, j, disparity) = absoluteDifference((int)firstIntensity, (int)secondIntensity);
                }
                count++;
            }
        }
        results.push_back(move(current));
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);

    cout << "Average time taken: look-up-table "
         << duration.count() * 1e-6 / (d_max - d_min) << " seconds" << endl;
    cout << "Total time taken (with init): look-up table " << (duration.count() + tableInit.count()) * 1e-6 << " seconds" << endl;

    cout << "count: " << count << " size: " << gray1.rows * gray1.cols * (d_max - d_min + 1) << endl;
    // cout << results.at<int>(374, 449, 2) << endl
    // cout << results.at<int>(3, 4, 0) << endl;
    // cout << test.at<int>(150, 200) << endl;

    // outputMap = results(cv::Range::all, cv::Range::all, 0);
    // uchar *p = gray1.ptr<uchar>(2);
    // cout << (int)p[3] << endl;
    // cout << "gray1(450, 375): " << (int)gray1.ptr<uchar>(374)[449] << endl;
    // cout << "gray2(450, 375): " << (int)gray2.ptr<uchar>(374)[449] << endl;
    // cout << "results(450, 375):" << (int)results.at(0)->ptr<uchar>(374)[449] << endl;

    cout << "gray1(150, 300): " << (int)gray1.ptr<uchar>(150)[300] << endl;
    cout << "gray2(150, 298): " << (int)gray2.ptr<uchar>(150)[298] << endl;
    cout << "results(450, 375):" << (int)results.at(2)->ptr<uchar>(150)[300] << endl;
    // cout << results.size() << endl;
    // cout << "result(450, 375):" << (int)results.at<int>(374, 449, 100) << endl; //(red, stupac, disparity)

    // for(int disparity = d_min; disparity <= d_max; disparity++) {

    // }

    // showImage(gray1, "gray1");
    // showImage(gray2, "gray2");
    // destroyWindows();
}
