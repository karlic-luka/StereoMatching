
#include "stereo-matching.h"

using namespace std;
using namespace std::chrono;
#define MAX_SINGLE_COST 257;

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
    return this->censusLookUpTable[x][y];
}
void showImage(const cv::Mat &image, std::string windowName)
{
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, image);
}
StereoMatching::StereoMatching(std::string &left, std::string &right, int &d_min, int &d_max,
                               int &radius, Cost &cost) : d_min(d_min), d_max(d_max), radius(radius), cost(cost)
{
    gray1 = cv::imread(left);
    gray2 = cv::imread(right);
    cv::cvtColor(gray1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray2, gray2, cv::COLOR_BGR2GRAY);
    window = 2 * radius + 1;
    disparityRange = this->d_max - this->d_min + 1;
    columns = gray1.cols;
    rows = gray1.rows;
    outputMap = cv::Mat::zeros(rows, columns, CV_8UC1);
}
void StereoMatching::preprocessImages()
{
    if (needCensusTransform)
    {
        gray1 = censusTransform(gray1, 3);
        gray2 = censusTransform(gray2, 3);
        // cout << "tu sam" << endl;
    }
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
            p3 = current->ptr<uchar>(i + radius); //+radius=padding
            for (int j = 0; j < columns; ++j)
            {
                //X KOORDINATA
                if (j < disparity)
                    p3[j + radius] = p1[j]; //+radius = padding
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
    for (int row = radius; row < rows + radius; row++)
    {
        pOutput = outputMap.ptr<uchar>(row - radius);
        for (int column = radius; column < columns + radius; column++)
        {
            int minCost = window * window * MAX_SINGLE_COST;
            int currentCost, leftCost, rightCost;
            int minIndex = 0;
            //cache costs because of overlapping windows
            vector<int> previousCosts(disparityRange, 0);

            for (int i = 0; i < disparityRange; i++)
            {
                currentCost = previousCosts.at(i);
                if (column - radius < d_min + i) //x_l < disparity
                    break;
                cv::Mat leftColumn(*(results.at(i)), cv::Range(row - radius, row + radius + 1), cv::Range(column - radius, column - radius + 1)); //+1+1 bc [..>
                cv::Mat rightColumn(*(results.at(i)), cv::Range(row - radius, row + radius + 1), cv::Range(column + radius, column + radius + 1));
                leftCost = (int)cv::sum(leftColumn)[0];
                rightCost = (int)cv::sum(rightColumn)[0];
                if (column == radius) //if I am in a new row
                {
                    //already have cost from left and right column
                    cv::Mat middle(*results.at(i), cv::Range(row - radius, row + radius + 1), cv::Range(column - radius + 1, column + radius));
                    // if(row == radius && column == radius) {
                    //     cout << "middle column size: " << middle.size << endl;
                    //     cout << middle << endl;
                    // }
                    int middleCost = (int)cv::sum(middle)[0];
                    currentCost = middleCost + leftCost + rightCost;
                }
                else
                {
                    //otherwise just add right cost (left is subtracted at the end -> before assigning)
                    currentCost += rightCost;
                }
                if (currentCost < minCost)
                {
                    minCost = currentCost;
                    minIndex = i;
                }
                currentCost -= leftCost;
                previousCosts.at(i) = currentCost;
            }
            int disparity = d_min + minIndex;
            pOutput[column - radius] = disparity;
        }
    }
}
cv::Mat StereoMatching::getDisparityMap()
{
    preprocessImages();
    prepareCostPerDisparity();
    rollingWindowCost();
    return outputMap;
}
cv::Mat StereoMatching::censusTransform(const cv::Mat &matrix, int window)
{
    cv::Mat padded;
    int radius = (window - 1) / 2; // w=2*r+1
    cv::Mat temporary = cv::Mat::zeros(matrix.rows, matrix.cols, CV_8UC1);
    cv::copyMakeBorder(matrix, padded, radius, radius, radius, radius, cv::BORDER_CONSTANT, cv::Scalar(0));

    uchar *p;
    for (int row = radius; row < matrix.rows + radius; row++)
    {
        p = temporary.ptr<uchar>(row - radius);
        for (int column = radius; column < matrix.cols + radius; column++)
        {
            int cost = 0;
            int potentialCost = 128;
            for (int i = -radius; i < radius + 1; i++)
            {
                for (int j = -radius; j < radius + 1; j++)
                {
                    if (i == 0 && j == 0)
                        continue; // I want to skip the central pixel

                    if (padded.at<uchar>(row + i, column + j) > padded.at<uchar>(row, column))
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
        cv::Mat output = stereoMatching.getDisparityMap();
        showImage(output, "output");
        cout << output.size() << endl;
    }
    else if (cost == "census")
    {
        CensusCost census = CensusCost();
        StereoMatching stereoMatching = StereoMatching(left, right, d_min, d_max, radius, census);
        stereoMatching.setNeedCensusTransform(1); //1=True
        cv::Mat output = stereoMatching.getDisparityMap();
        showImage(output, "output");
        cout << output.size() << endl;
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
    // cv::Mat groundTruth = cv::imread("..//images//teddy//disp2.png", cv::IMREAD_GRAYSCALE);
    // showImage(groundTruth, "ground truth");
    // cout << groundTruth.size() << endl;

    destroyWindows();

    // For small matrices you may use comma separated initializers:
    // cv::Mat C = (cv::Mat_<uchar>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
    // cv::Mat padded;
    // cv::copyMakeBorder(C, padded, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(0));
    // cout << padded << endl;
    // int currentCost = 0, leftCost, rightCost, previousCost;
    // int minCost = 25 * 257;
    // int count = 0;
    // for (int i = 2; i < C.rows + 2; i++)
    // {
    //     for (int j = 2; j < C.cols + 2; j++)
    //     {
    //         count++;
    //         currentCost = previousCost;
    //         cv::Mat leftCol(padded, cv::Range(i - 2, i + 2 + 1), cv::Range(j - 2, j - 2 + 1));
    //         cv::Mat rightCol(padded, cv::Range(i - 2, i + 2 + 1), cv::Range(j + 2, j + 2 + 1));
    //         leftCost = (int)cv::sum(leftCol)[0];
    //         rightCost = (int)cv::sum(rightCol)[0];
    //         cout << "current left cost " << leftCost << endl;
    //         cout << "Current left " << endl;
    //         cout << leftCol << endl;
    //         cout << "right cost " << rightCost << endl;
    //         cout << "current right" << endl;
    //         cout << rightCol << endl;

    //         if (j == 2)
    //         {
    //             cv::Mat middle(padded, cv::Range(i - 2, i + 2 + 1), cv::Range(j - 2 + 1, j + 2));
    //             cout << "Middle " << middle << endl;
    //             int middleCost = (int)cv::sum(middle)[0];
    //             currentCost = leftCost + rightCost + middleCost;
    //         }
    //         else
    //         {
    //             currentCost += rightCost;
    //         }
    //         if (currentCost < minCost)
    //         {
    //             minCost = currentCost;
    //         }
    //         cout << "current cost " << currentCost << endl;
    //         currentCost -= leftCost;
    //         previousCost = currentCost;
    //     }
    // }
    // cout << minCost << endl;
    // cout << "Count: " << count << endl;
    // cv::Mat census = censusTransform(C, 3);
    // cout << census << endl;
    // AbsoluteDifferenceCost abs = AbsoluteDifferenceCost();
    // CensusCost census = CensusCost();
    // cout << abs.apply(5, 10) << " " << abs.apply(10, 5) << " " << abs.apply(255, 255) << endl;
    // cout << census.apply(5, 10) << " " << census.apply(10, 5) << " " << census.apply(255, 255) << endl;
    // cout << census.apply(125, 17) << " " << census.apply(187, 47) << " " << endl;
    return 0;
}
