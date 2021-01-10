#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <getopt.h>
#include <boost/program_options.hpp>
#include <bitset>
#include <string>
#include <array>

/**
 * Abstract class for implementing different cost functions.
 */
class Cost
{
public:
    Cost() {}
    /**
     * Returns result of a cost function applied on input integers 
     * @param x input integer
     * @param y input integer to be evaluated
     */
    virtual int apply(const int &x, const int &y) const = 0; //pure virtual function, derived class will implement it
protected:
    /**
     * Helper look-up table to apriori compute all results of applying cost function to first 256 integers.
     * It is a symmetric matrix where (i, j) stores cost(i, j) 
     */
    int lookUpTable[256][256];
};
/**
 * Cost function: absolute difference
 */
class AbsoluteDifferenceCost : public Cost
{
protected:
    /**
     * Helper method. Returns |x - y|
     * @param x
     * @param y
     */
    int process(const int &x, const int &y) const;

public:
    /* Default constructor */
    AbsoluteDifferenceCost();
    /**
     * Returns |x - y|
     * @param x input integer
     * @param y input integer 
     */
    int apply(const int &x, const int &y) const override;
};
/**
 * Cost function: census cost.
 */
class CensusCost : public Cost
{
protected:
    /**
     * Helper method that performs middle step for Census transform
     * @param x
     * @param y
     * @return Hamming distance of 8bit representation of input integers.
     */
    int process(const int &x, const int &y) const;

public:
    /* Default constructor */
    CensusCost();
    /**
     * Returns Hamming distance of 8bit representation of input integers.
     * It is meant to be applied after Census transform is done.
     * @param x input integer
     * @param y input integer
     */
    int apply(const int &x, const int &y) const override;
};
/**
 * My implementation of stereo matching using rolling windows on different cost functions.
 * Currently works for absolute difference and Census cost. 
 */
class StereoMatching
{
public:
    /**
    * Constructor.
    * @param left path to the left image
    * @param right path to the right image
    * @param d_min minimum disparity to be evaluated
    * @param d_max maximum disparity to be evaluated
    * @param radius radius of a window for cost function
    * @param cost cost function
    */
    StereoMatching(std::string &left, std::string &right, int &d_min, int &d_max,
                   int &radius, Cost &cost);
    /**
     * Returns disparity map.
     */
    cv::Mat getDisparityMap();
    /**
     * Setter for flag that implies if census transform needs to be performed.
     * Currently implemented just for the purpose of Census cost - needs to be implemented in a different way.
     */
    void setNeedCensusTransform(bool flag) { needCensusTransform = flag; }

protected:
    /**
     * Grayscale version of left input image.
     */
    cv::Mat gray1;
    /**
     * Grayscale version of right input image.
     */
    cv::Mat gray2;
    /**
     * Disparity map
     */
    cv::Mat outputMap;
    int d_min, d_max, radius, window, disparityRange, rows, columns;
    Cost &cost;
    /**
     * Vector of unique pointers to matrix objects.
     * (i, j) stores cost_function(i, j)
     * i-th position in vector stores upper result for disparity = d_min + i
     */
    std::vector<std::unique_ptr<cv::Mat>> results;
    bool needCensusTransform = 0;
    void prepareCostPerDisparity();
    /**
     * Helper method for performing rolling window cost.
     */
    void rollingWindowCost();
    /**
     * Helper method for calling Census transform on both input images.
     */
    void preprocessImages();
    /**
     * Helper method: performs Census transform
     * @param matrix input matrix
     * @param window size of a window
     */
    cv::Mat censusTransform(const cv::Mat &matrix, int window);
};

//helper methods
/**
 * Helper method for displaying image
 */
void showImage(const cv::Mat &image, std::string windowName);
/**
 * Helper method for destroying all displayed windows.
 */
void destroyWindows();