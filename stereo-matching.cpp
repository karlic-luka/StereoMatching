
#include "stereo-matching.h"

using namespace std;
using namespace std::chrono;
#define MAX_SINGLE_COST 256;

AbsoluteDifferenceCost::AbsoluteDifferenceCost() : Cost()
{
    //lookup table initialization
    for (int row = 0; row < 256; row++)
    {
        for (int column = row; column < 256; column++)
        {
            int result = this->process(row, column);
            this->ADLookUpTable[row][column] = result;
            this->ADLookUpTable[column][row] = result;
        }
    }
}
int AbsoluteDifferenceCost::process(const int &x, const int &y) const
{
    return x >= y ? (x - y) : (y - x);
}
int AbsoluteDifferenceCost::apply(const int &x, const int &y) const
{
    return this->ADLookUpTable[x][y];
}
CensusCost::CensusCost() : Cost()
{
    //lookup table initialization
    for (int row = 0; row < 256; row++)
    {
        for (int column = row; column < 256; column++)
        {
            int result = this->process(row, column);
            this->censusLookUpTable[row][column] = result;
            this->censusLookUpTable[column][row] = result;
        }
    }
}
int CensusCost::process(const int &x, const int &y) const
{
    int bitwiseOR = x ^ y; //for hamming distance
    //I could have implemented function that counts number of ones
    //in a 8bit string of 0and1, but I used gcc built-in function
    return __builtin_popcount(bitwiseOR);
}
int CensusCost::apply(const int &x, const int &y) const
{
    return censusLookUpTable[x][y];
}
void showImage(const cv::Mat &image, std::string windowName)
{
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, image);
}
StereoMatching::StereoMatching(std::string &left, std::string &right, int &d_min, int &d_max,
                               int &radius, Cost &cost) : d_min(d_min), d_max(d_max), radius(radius), cost(cost)
{
    cv::Mat image1 = cv::imread(left);
    cv::Mat image2 = cv::imread(right);
    cv::cvtColor(image1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);
    if (needCensusTransform)
    {
        gray1 = censusTransform(gray1, 3);
        gray2 = censusTransform(gray2, 3);
    }
    window = 2 * radius + 1;
    disparityRange = this->d_max - this->d_min + 1;
    columns = gray1.cols;
    rows = gray2.rows;

    outputMap = cv::Mat::zeros(rows, columns, CV_8UC1);
}
void StereoMatching::prepareCostPerDisparity()
{
    uchar *p1, *p2, *p3;
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
                    p3[j + radius] = cost.apply((int)firstIntensity, (int)secondIntensity);
                }
            }
        }
        results.push_back(move(current));
    }
}
void destroyWindows()
{
    cv::waitKey(0);
    cv::destroyAllWindows();
}
void StereoMatching::rollingWindowCost()
{
    uchar *pOutput;
    int minIndex, currentCost, leftCost, rightCost;
    for (int row = radius; row < rows + radius; row++)
    {
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
}
cv::Mat StereoMatching::getDisparityMap()
{
    prepareCostPerDisparity();
    rollingWindowCost();
    return outputMap;
}
cv::Mat StereoMatching::censusTransform(const cv::Mat &matrix, int window)
{
    int radius = (window - 1) / 2; // w=2*r+1
    cv::Mat temporary = cv::Mat::zeros(matrix.rows, matrix.cols, CV_8UC1);
    cv::copyMakeBorder(matrix, matrix, radius, radius, radius, radius, cv::BORDER_CONSTANT, cv::Scalar(0));
    // cout << "tu sam: " << matrix << endl;
    uchar *p;
    for (int row = radius; row < matrix.rows - radius; row++)
    {
        p = temporary.ptr<uchar>(row - radius);
        for (int column = radius; column < matrix.cols - radius; column++)
        {
            int cost = 0;
            int potentialCost = 128;
            for (int i = -radius; i < radius + 1; i++)
            {
                for (int j = -radius; j < radius + 1; j++)
                {
                    if (i == 0 && j == 0)
                        continue; // I want to skip the central pixel

                    if (matrix.at<uchar>(row + i, column + j) > matrix.at<uchar>(row, column))
                        cost += potentialCost;
                    potentialCost >>= 1; //divide by 2
                }
            }
            p[column - radius] = cost;
        }
    }
    return temporary;
}
int main(int argc, char **argv)
{
    // cout << "Parameters: " << left << " " << right << " " << d_min << " " << d_max << " " << radius << " " << window << " " << cost << endl;
    //-------------------PARSING COMMAND LINE-------------------------------

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
    string cost = parser.get<string>("cost");

    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }
    if (d_min > d_max)
    {
        cout << "You provided minimum disparity larger than maximum disparity. Please try again." << endl;
        return -1;
    }
    if (d_min < 0 || d_max < 0 || radius < 0)
    {
        cout << "Some of your parameters were negative integers. Please provide only positives." << endl;
        return -1;
    }
    if (d_max - d_min >= 120)
    {
        cout << "Wow. That's pretty large range for disparity. Please wait. It shouldn't take more than 30 seconds." << endl;
    }
    auto start = high_resolution_clock::now();
    if (cost == "AD")
    {
        AbsoluteDifferenceCost absDiff = AbsoluteDifferenceCost();
        StereoMatching stereoMatching = StereoMatching(left, right, d_min, d_max, radius, absDiff);
        stereoMatching.setNeedCensusTransform(1); //1=True
        cv::Mat output = stereoMatching.getDisparityMap();
        showImage(output, "output");
    }
    else if (cost == "census")
    {
        CensusCost census = CensusCost();
        StereoMatching stereoMatching = StereoMatching(left, right, d_min, d_max, radius, census);
        cv::Mat output = stereoMatching.getDisparityMap();
        showImage(output, "output");
    }
    else
    {
        cout << "I didn't implement that cost function" << endl;
        return -1;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Total time taken to generate output map: " << duration.count() * 1e-6 << " seconds" << endl;

    // cout << output.size() << endl;
    // showImage(output, "output");
    //-----------------------------------------------------------------------------------------
    // cv::Mat image1, image2, gray1, gray2;
    // image1 = cv::imread(left);
    // image2 = cv::imread(right);
    // int columns = image1.cols;
    // int rows = image1.rows;
    // assert(image1.size == image2.size); //kasnije jos dodati malo teksta

    // cv::cvtColor(image1, gray1, cv::COLOR_BGR2GRAY);
    // cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);

    // if (!image1.data || !image2.data)
    // {
    //     printf("No image data \n");
    //     return -1;
    // }
    // auto start = high_resolution_clock::now();
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
    // auto tableInit = duration;

    // int count = 0;
    // uchar *p1, *p2, *p3;

    // Get starting timepoint
    // vector<unique_ptr<cv::Mat>> results;

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
    // int disparityRange = d_max - d_min + 1;

    // cv::Mat outputMap = cv::Mat::zeros(rows, columns, CV_8UC1);

    // stop = high_resolution_clock::now();
    // duration = duration_cast<microseconds>(stop - start);
    // cout << "Total time taken to generate window cost and estimate output map: " << duration.count() * 1e-6 << " seconds" << endl;

    // showImage(gray1, "gray1");
    // showImage(gray2, "gray2");

    // showImage(gray1, "gray1");
    // showImage(gray2, "gray2");
    // showImage(outputMap, "output");
    destroyWindows();

    // For small matrices you may use comma separated initializers:
    // cv::Mat C = (cv::Mat_<uchar>(3, 3) << 124, 74, 32, 124, 64, 18, 157, 116, 84);
    // cout << C << endl;
    // cv::Mat census = censusTransform(C, 3);
    // cout << census << endl;
    return 0;
}
