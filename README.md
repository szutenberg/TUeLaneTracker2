# TUeLaneTracker

This is a software application that detects and tracks lane boundaries. The underlying algorithm is a probabilistic classifier which was originally developed, under the strategic area of Smart Mobility, at Eindhoven University of Technology (TU/e). The algorithm exploits the concept of hierarchical classification from deep learning, however, unlike deep learning, classification at each hierarchical level is engineered instead of being trained through images. The software application is completely object oriented and follows various software design principles recommended by the safety standard ISO26262. 

This application provides a loose coupling between the software control flow and the algorithm, making it possible to generate various target specific implementations for the algorithm. Besides this generic implementation, an accelrated version for the NXP-BlueBox (s32v) is also available. The s32v specific implementation makes use of the APEX accelrators to speed-up the vision processing. The APEX architecture blends scalar and vector processing capabilities within the two fully programable cores, achieving an effective 5-10x speed-up for the algorithm. 

#### Note: A ROS package for this lane tracker is available under [tue_lane_tracker](https://github.com/RameezI/tue_lane_tracker) repository. 

### Prerequisites

What libraries do you need, in order to build and install the TUeLaneTracker

```
* OpenCv-3.1.0
* Boost-1.62.0
```
How to install them:

[OpenCV Installation Guide](http://docs.opencv.org/3.1.0/d7/d9f/tutorial_linux_install.html)

[Boost Library Sources](http://www.boost.org/users/history/version_1_62_0.html)

### Xubuntu 18.04.1

Step-by-step installation on a fresh [Xubuntu 18.04.1](http://nl.archive.ubuntu.com/ubuntu-cdimage-xubuntu/releases/18.04/release/) installation:

1. Install necessary packages
```
sudo apt-get install gcc g++
sudo apt-get install build-essential
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
```

2. Download and unpack OpenCV 3.4.5
```
wget https://github.com/opencv/opencv/archive/3.4.5.tar.gz
tar -xf 3.4.5.tar.gz && cd opencv-3.4.5
```

3. Build and install OpenCV
```
mkdir build
cd build
cmake ..
make -j4
sudo make install
```

4. Download and unpack Boost 1.62.0
```
cd ~
wget https://sourceforge.net/projects/boost/files/boost/1.62.0/boost_1_62_0.tar.gz
tar -xf boost_1_62_0.tar.gz && cd boost_1_62_0
```

5. Build and install Boost
```
./bootstrap.sh
sudo ./b2 install
```

### Building

1. clone this repository: 
```
git clone https://github.com/szutenberg/TUeLaneTracker2.git
```

2. Go to the parent directory of the cloned repository:
```
cd TUeLaneTracker2
```

3. Create a build subdirectory and invoke cmake:
```
mkdir build
cd build
cmake ..
```

4. Build
```
make
```

5. Download `GradientTangent_640x480.yaml`: **TODO: figure out what is it for, make it smaller and include in repo**
```
mkdir -p ../LaneTrackerApp/build/ConfigFiles/Templates
wget https://github.com/RameezI/TUeLaneTracker/raw/master/install/ConfigFiles/Templates/GradientTangent_640x480.yaml -O ../LaneTrackerApp/build/ConfigFiles/Templates/GradientTangent_640x480.yaml
```

6. Executable is built in `../LaneTrackerApp/build/TUeLaneTracker`

### Usage

#### Imgstore

Imgstore is a directory with images (`01.png`, `02.png`, `03.png` and so on).

```
<APP> -m imgstore -s <DIR_PATH>
```

Sample imgstore can be downloaded from [here](https://github.com/szutenberg/TUeLaneTracker/tree/master/install/DataSet).

#### Stream

It can be NetworkStream, VideoFile or V4LCapture. 

```
<APP> -m stream -s <PATH>
```


### Callibrating TUeLaneTracker

The algorithm is parameterised with a minimal set of parameters. These parameters includes, among others, intrinsic and extrinsic camera specifications. At the boot-time the algorithm takes into account these parameters, whcih are set via the [include/Config.h](https://github.com/RameezI/TUeLaneTracker/blob/master/include/Config.h) file. 

This provides a way to reconfigure the algorithm but only at the compile time. However, in case of the ROS (Robot Operating System) package for the lane tracker it is possible to dynamically reconfigure and reboot the algorithm, with new configuration, at runtime. The ros package for this lane tracker is available under the [tue_lane_tracker](https://github.com/RameezI/tue_lane_tracker) repository. 
       
       
## Youtube Videos
   These videos showcase the functional performance of the algorithm:
   * [Eindhoven](https://youtu.be/7D1vBPrcPk0)


## Built With

* [OpenCV3.1](http://docs.opencv.org/3.1.0/index.html) - Copmputer vision library
* [CMake](https://maven.apache.org/) - Dependency management and makefiles generation
* [Eigen 3.3.3](http://eigen.tuxfamily.org/index.php?title=Main_Page) - Linear Algebra  library
* [Boost 1.62.0](http://www.boost.org/users/history/version_1_62_0.html) - Provides Program Options

## Authors
* **Michal Szutenberg**
* **Rameez Ismail**

See also the list of [contributors](https://github.com/RameezI/TUeLaneTracker/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/RameezI/TUeLaneTracker/blob/master/LICENSE.md) file for details

## Acknowledgments

This research and design project was supported by NXP Semiconductors and Eindoven University of Technology. 
