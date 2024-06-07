# Simple Matrix class with GPU support for matrix multiplication

# Installation
- Clone the repo by:
```
git clone git@github.com:kuku929/GPUMatrix.git <some-name>
cd <some-name>
```

- Make a build directory and create a build file. You can use any build system of your choice, I have used ninja
```
mkdir build | cd build
cmake .. -G Ninja
```

- Build the project
```
ninja
```

# Usage

Import GPUMatrix.h in your project and you are good to go!
NOTE : Please take a look at the CMakeLists.txt for more information on appropriate flags for best performance

