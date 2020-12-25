// #include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <getopt.h>
#include <boost/program_options.hpp>

// using namespace cv;
using namespace std;
using namespace std::chrono;
#define MAX_SINGLE_COST 256;
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

int main(int argc, char **argv)
{
    // if (argc != 3)
    // {
    //     printf("usage: DisplayImage.out <Image1_Path> <Image2_Path> d_min d_max\n");
    //     return -1;
    //
    const string keys =
        "{help h usage ? | | print this message}"
        "{left l |<none>| path to left image}"
        "{right r |<none> | path to right image}"
        "{d_min | 0 | minimum disparity to evaluate}"
        "{d_max |255| maximum disparity to evaluate}"
        "{window w | 1 | radius of a square window}"
        "{cost c | AD | Cost function. Only AD and Census available.}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    string left = parser.get<string>("left");
    string right = parser.get<string>("right");
    int d_min = parser.get<int>("d_min");
    int d_max = parser.get<int>("d_max");
    int radius = parser.get<int>("window");
    int window = 2 * radius + 1;
    string cost = parser.get<string>("cost");

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    if (d_min > d_max)
    {
        cout << "You provided minimum disparity larger than maximum disparity. Please try again." << endl;
        return -1;
    }
    if(d_min <0 || d_max <0 || radius < 0) {
        cout << "Some of your parameters were negative integers. Please provide only positives." << endl;
        return -1;
    }
    if(d_max - d_min >= 120) {
        cout << "Wow. That's pretty large range for disparity. Please wait. It shouldn't take more than 30 seconds." << endl;
    }
    // cout << "Parameters: " << left << " " << right << " " << d_min << " " << d_max << " " << radius << " " << window << " " << cost << endl;
    //-----------------------------------------------------------------------------
    
    cv::Mat image1, image2, gray1, gray2;
    image1 = cv::imread(left);
    image2 = cv::imread(right);
    int columns = image1.cols;
    int rows = image1.rows;
    // assert(image1.size == image2.size); //kasnije jos dodati malo teksta
    // int d_min = 0, d_max = 255, radius = 2;

    cv::cvtColor(image1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);
    // assert(gray1.channels() == gray2.channels() == 1);

    if (!image1.data || !image2.data)
    {
        printf("No image data \n");
        return -1;
    }
    //----------------------------------LOOKUP TABLICA ZA ABSOLUTE DIFFERENCE------
    auto start = high_resolution_clock::now();
    for (int row = 0; row < 256; row++)
    {
        for (int column = row; column < 256; column++)
        {
            ADlookUpTable[row][column] = absoluteDifference(row, column);
            ADlookUpTable[column][row] = absoluteDifference(row, column);
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    auto tableInit = duration;
    // cout << "Look-up-table init: " << duration.count() << " microseconds." << endl;
    //---------------------------------------------------------------------------------
    // cv::Mat roi(image1, cv::Range(50, 52), cv::Range(10, 15)); //range of interest
    // cout << "ROI(numpy" << endl;
    // cout << cv::format(roi, cv::Formatter::FMT_NUMPY) << endl;

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
        unique_ptr<cv::Mat> current(new cv::Mat(rows + 2 * radius, columns + 2 * radius, CV_8UC1, cv::Scalar(0))); //radius=padding
        for (int i = 0; i < rows; ++i)
        {
            //PAZI: OVO JE REDAK --> Y-KOORDINATA
            p1 = gray1.ptr<uchar>(i);
            p2 = gray2.ptr<uchar>(i);
            //+radius jer zelim dodati padding na svaku stranu
            p3 = current->ptr<uchar>(i + radius);
            for (int j = 0; j < columns; ++j)
            {
                //X KOORDINATA
                if (j < disparity)
                    p3[j + radius] = p1[j]; //+radius jer zelim dodati padding na svaku stranu
                else
                {
                    auto firstIntensity = p1[j];
                    auto secondIntensity = p2[j - disparity];
                    //+radius jer zelim dodati padding na svaku stranu
                    p3[j + radius] = ADlookUpTable[(int)firstIntensity][(int)secondIntensity];
                    // p3[j] = absoluteDifference((int)firstIntensity, (int)secondIntensity);
                }
                count++;
            }
        }
        results.push_back(move(current));
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);

    // cout << "Average time taken: look-up-table "
    //      << duration.count() * 1e-6 / (d_max - d_min) << " seconds" << endl;
    // cout << "Total time taken (with init): look-up table " << (duration.count() + tableInit.count()) * 1e-6 << " seconds" << endl;

    // cout << "count: " << count << " size: " << gray1.rows * gray1.cols * (d_max - d_min + 1) << endl;
    //-------------------------------------------------------------------------------------------------------------------------------
    // cout << results.at<int>(374, 449, 2) << endl
    // cout << results.at<int>(3, 4, 0) << endl;
    // cout << test.at<int>(150, 200) << endl;

    // outputMap = results(cv::Range::all, cv::Range::all, 0);
    // uchar *p = gray1.ptr<uchar>(2);
    // cout << (int)p[3] << endl;
    // cout << "gray1(450, 375): " << (int)gray1.ptr<uchar>(374)[449] << endl;
    // cout << "gray2(450, 375): " << (int)gray2.ptr<uchar>(374)[449] << endl;
    // cout << "results(450, 375):" << (int)results.at(0)->ptr<uchar>(374)[449] << endl;

    // cout << "gray1(150, 300): " << (int)gray1.ptr<uchar>(150)[300] << endl;
    // cout << "gray2(150, 298): " << (int)gray2.ptr<uchar>(150)[298] << endl;
    // cout << "results(450, 375):" << (int)results.at(2)->ptr<uchar>(150)[300] << endl;
    // cout << "Size: " << results.size() << endl;

    // cout << results.size() << endl;
    // cout << "result(450, 375):" << (int)results.at<int>(374, 449, 100) << endl; //(red, stupac, disparity)
    int disparityRange = d_max - d_min + 1;

    cv::Mat outputMap = cv::Mat::zeros(rows, columns, CV_8UC1);
    uchar *pOutput;
    int minIndex, currentCost, leftCost, rightCost;
    start = high_resolution_clock::now();
    for (int row = radius; row < rows + radius; row++)
    {
        p1 = gray1.ptr<uchar>(row);
        p2 = gray2.ptr<uchar>(row);
        pOutput = outputMap.ptr<uchar>(row - radius);
        for (int column = radius; column < columns + radius - 1; column++)
        {
            int min_cost = window * window * MAX_SINGLE_COST;

            for (int i = 0; i < disparityRange; i++)
            {
                if (column - radius < d_min + i)
                {
                    // pOutput[column - radius] = p1[column - radius];
                    continue;
                }
                cv::Mat leftColumn(*results.at(i), cv::Range(row - radius, row + radius + 1), cv::Range(column - radius, column - radius + 1 + 1)); //+1+1 da se skuzi o cemu se radi
                cv::Mat rightColumn(*results.at(i), cv::Range(row - radius, row + radius + 1), cv::Range(column + radius, column + radius + 1 + 1));
                leftCost = (int)cv::sum(leftColumn)[0];
                rightCost = (int)cv::sum(rightColumn)[0];
                if (column == radius) //bilo bi inace 0, al padding
                {
                    //onda zelim inicijalizirati trenutni prozor i oznaciti lijevi i desni rub --> vec ih imam
                    //lijevi i desni stupac imam vec, samo me zanima sto je izmedu (da ne racunam dvaput)
                    cv::Mat middle(*results.at(i), cv::Range(row - radius, row + radius + 1), cv::Range(radius + 1, radius + window)); //
                    currentCost = (int)cv::sum(middle)[0] + leftCost + rightCost;
                }
                else
                {
                    //inace dodaj cost od stupca koji je sad "dosao" - desni, a na kraju micem lijevi cost
                    currentCost += rightCost;
                }
                if (currentCost < min_cost)
                {
                    min_cost = currentCost;
                    minIndex = i;
                }
                currentCost -= leftCost;
            }
            int disparity = d_min + minIndex;
            pOutput[column - radius] = disparity;
        }
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "Total time taken to generate window cost and estimate output map: " << duration.count() * 1e-6 << " seconds" << endl;

    // showImage(gray1, "gray1");
    // showImage(gray2, "gray2");

    showImage(gray1, "gray1");
    showImage(gray2, "gray2");
    showImage(outputMap, "output");
    destroyWindows();
    return 0;
}
