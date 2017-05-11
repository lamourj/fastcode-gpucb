Processor Counter Monitor
=========================
Based on [Intel Processor Counter Monitor][pcm]
```
Commit: 8840be4128d4bf6113c1a42bd12e7b1c06a8d2f6
```
[pcm]: https://github.com/opcm/pcm

Prerequisites
=================

1. [Disable Intel® Turbo Boost Technology][TurboBoost]
2. [Disable Intel® Hyper-Threading Technology][HyperThreading]
3. [OS support for Intel® Performance Counter Monitor][IntelPCMDrivers]

[TurboBoost]: doc/TurboBoost.md
[HyperThreading]: doc/HyperThreading.md
[IntelPCMDrivers]: doc/IntelPCMDrivers.md

Linux & Mac OS X (Yosemite / El Capitan / Sierra)
---------------------

1. Install [CMake v3.0.2+][cmake]
2. Install [XCode (clang)][xcode] / [Intel® C++ Compiler Parallel Studio XE (icc)][icc] / [GCC][gcc]
3. To Build it with XCode:
```
cd IntelPCM
mkdir build
cd build
/Applications/CMake.app/Contents/bin/cmake ..
cd ..
/Applications/CMake.app/Contents/bin/cmake --build build --config Release
```

4. To build it with Intel® C++ Compiler:
```
CC=icc CXX=icpc /Applications/CMake.app/Contents/bin/cmake ..
```

5. GCC:
```
CC=gcc CXX=g++ /Applications/CMake.app/Contents/bin/cmake ..
```


Windows (7 / 10):
--------

1. Install [CMake v3.0.2+][cmake] (don't forget to add it to `%PATH%`)
2. Install [Visual Studio 2015 Community Edition][vs14] (or latter)
3. Then run:

```
cd IntelPCM
mkdir build
cd build
cmake .. -G "Visual Studio 14 2015 Win64"
cd ..
cmake --build build --config Release
```

Note that it is possible to use [MinGW](http://www.mingw.org/) in Windows as well. 
However, it has to be installed in addition of Visual Studio, since the core of
 [Intel PCM][pcm] compiles with Visual Studio only. 

Failback Mode
=================

If running on a CPU that is not Intel based, the CMake script can failback to RDTSC:

```
cmake -DRDTSC_FAILBACK=1 ..
```

[xcode]: https://developer.apple.com/xcode/
[icc]: https://software.intel.com/en-us/c-compilers
[gcc]: https://gcc.gnu.org/
[cmake]: https://cmake.org/download/
[vs14]: https://www.visualstudio.com/downloads/
