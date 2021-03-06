# StereoMatching
## About
This is my implementation of problems described in *Junior Computer Vision Engineer.pdf*. It is assignment for candidates applying for mentioned position at Gideon Brothers.
The goal was to solve the problem of dense stereo correspondence and generate disparity map using "hello world" algorithm - block matching with absolute difference or Census cost as cost functions.
The rest is explained in *Junior Computer Vision Engineer.pdf*.

## Implementation details
Crucial part in implementation was to structure the processing to avoid calculating the matching cost for each pixel and disparity for more times than it is neccessary.
Also, aggregation within the window should not incur unnecessary performance penalties. There's few things I did about that.

#### 1) ***look-up table***
Cost functions could be more complex than simple absolute difference. They may require multiplications, divisions or other pretty expensive operations for computer, especially when you have to call it tens of millions of times. Our program works with grayscale images, which means that values vary from 0 to 255. That's why I decided to use 256x256 look-up tables where I precalculated all ~ 128x256 combinations (notice, our cost functions are symmetrical - we only need upper or lower part of matrix). Look-up tables's strength is that we do not need to make the calculation, we just need to read the result and we can now do that in constant time (O(1)). 
#### 2) ***rolling window***
When it comes to calculating local cost using square windows I took advantage of a fact that overlapping windows occur. I was scanning through all pixels of an image by rows (because that's how cv::Mat is stored in memory) and each column shift to the right I would just add cost of that "new" column and subtract cost of "leftest" (the one most to the left).  In that case if window was W x W I would remember middle cost of size W x (W-2).
P.S. Maybe there's one more thing I could do -> when rows are even shift columns to the right, otherwise shift columns to the left (so I could use stored cost for upper row of window), but I think it would be unnecessary and code would be harder to read.
#### 3) ***vector of smart pointers to cv::Mat instead of 3D cv::Mat***
I needed to store costs per each pixel and per each disparity. First solution was to make 3D cv::Mat object where that third dimension would represent disparity. But, that solution was pretty bad because of how cv::Mat objects are stored in memory and also later it would be hard to get a 2D view od 3D matrix (rolling windows). I didn't even get to the part where I have to "roll windows" when I found out how slow it is to even store the information (cv::Mat.ptr.at(i, j, k) is too slow).
But, I remembered I could, actually, "immitate" 3D matrix by a std::vector<> of matrixes, which was few times faster and so much easier to use.
Reason for using smart pointers is so I could store them by pointer to avoid copying very large matrixes and don't need to think about deallocating pointers.

#### 4)***Census cost***
There's bijection between 8bit strings and integers from 0 to 255 and it was easier for me to work with numbers than strings. I implemented Census cost in 2 simple steps (after Census transform):
a) bitwise XOR so I get "1" where both bits are different
b) counting how many "1"s are after bitwise XOR
P.S. there's lots of fun algorithms which I could use to count "1"s bitwise (e.g. Brian Kernighan's algorithm - O(logn)), but I thought it's not the crucial part of my assignment, so I used builtin_popcount() from GCC.


### COMPILE AND RUN
***1) How to compile with cmake***

Place yourself somewhere where you want to copy my repository
```
1. git clone https://github.com/karlic-luka/StereoMatching.git
2. cd StereoMatching
3. mkdir build
4. cd build
5. cmake ..
6. cmake --build .
```
***2) how to run the program***
Once the program is compiled you can run it like this (if you're in the build directory)
```
./stereo-matching -left=<path to left picture> -right=<path to right image> -d_min=<minimum disparity to evaluate> -d_max=<maximum disparity to evaluate> -window=<radius of a square window> -cost=<cost function>
```

  
 All parameters except left and right are optional.
 You can also call 
 ```
 ./stereo-matching help 
 ```
 
 P.S. I was using WSL2 Ubuntu 18.04 since I don't have Linux OS.

### PREDEFINED EXAMPLES
In StereoMatching/images folder there are some pictures on which program can be evaluated.
###### a) CONES
```
a.1) ./stereo-matching -left=../images/cones/im2.png -right=../images/cones/im6.png  -cost="AD" -window=6 -d_min=20 -d_max=100
a.2) ./stereo-matching -left=../images/cones/im2.png -right=../images/cones/im6.png  -cost="AD" -window=7 -d_min=20 -d_max=100
a.3) ./stereo-matching -left=../images/cones/im2.png -right=../images/cones/im6.png  -cost="AD" -window=6 -d_min=10 -d_max=130
a.4) ./stereo-matching -left=../images/cones/im2.png -right=../images/cones/im6.png  -cost="census" -window=3 -d_min=10 -d_max=80
a.5) ./stereo-matching -left=../images/cones/im2.png -right=../images/cones/im6.png  -cost="census" -window=4 -d_min=10 -d_max=80
a.6) ./stereo-matching -left=../images/cones/im2.png -right=../images/cones/im6.png  -cost="census" -window=8 -d_min=10 -d_max=120
```
###### b) TEDDY
```
b.1) ./stereo-matching -left=../images/teddy/im2.png -right=../images/teddy/im6.png  -cost="AD" -window=4 -d_max=80
b.2) ./stereo-matching -left=../images/teddy/im2.png -right=../images/teddy/im6.png  -cost="AD" -window=9 -d_max=80
b.3) ./stereo-matching -left=../images/teddy/im2.png -right=../images/teddy/im6.png  -cost="AD" -window=11 -d_max=100
b.4) ./stereo-matching -left=../images/teddy/im2.png -right=../images/teddy/im6.png  -cost="census" -window=9 -d_max=80
b.5) ./stereo-matching -left=../images/teddy/im2.png -right=../images/teddy/im6.png  -cost="census" -window=11 -d_max=80
b.6) ./stereo-matching -left=../images/teddy/im2.png -right=../images/teddy/im6.png  -cost="census" -window=11 -d_max=90
```
###### c) TSUKUBA
```
c.1) ./stereo-matching -left=../images/tsukuba/scene1.row3.col1.ppm -right=../images/tsukuba/scene1.row3.col5.ppm  -cost="AD" -window=3 -d_min=10 -d_max=80
c.2) ./stereo-matching -left=../images/tsukuba/scene1.row3.col1.ppm -right=../images/tsukuba/scene1.row3.col5.ppm  -cost="AD" -window=5 -d_min=10 -d_max=90
c.3) ./stereo-matching -left=../images/tsukuba/scene1.row3.col1.ppm -right=../images/tsukuba/scene1.row3.col5.ppm  -cost="AD" -window=8 -d_min=10 -d_max=130
c.4) ./stereo-matching -left=../images/tsukuba/scene1.row3.col1.ppm -right=../images/tsukuba/scene1.row3.col5.ppm  -cost="census" -window=11 -d_min=10 -d_max=80
```
###### d) VENUS
```
d.1) ./stereo-matching -left=../images/venus/im0.ppm -right=../images/venus/im5.ppm  -cost="census" -window=11 -d_min=0 -d_max=80
d.2) ./stereo-matching -left=../images/venus/im0.ppm -right=../images/venus/im5.ppm  -cost="AD" -window=11 -d_min=0 -d_max=80
d.3) ./stereo-matching -left=../images/venus/im0.ppm -right=../images/venus/im5.ppm  -cost="AD" -window=13 -d_min=8 -d_max=80
```
Almost all outputs from upper program calls are stored in StereoMatching/images/my-outputs.
It is good to mention that my program is not parallelized using OpenCV and it is ran on CPU.

P.S. Maybe my outputs would "look" better if I somehow scaled final pixel intensities, we can get brighter shades. I didn't do anything regarding that, I just stored disparity that gave minimum.
