# WAD


## Abstract

This code is about the research "Walking Assistive Device"

## How to install

### Install OpenGL, Python3, etc...

```bash
sudo apt-get install freeglut3-dev libpython3-dev python3-tk python3-numpy virtualenv ipython3 cmake-curses-gui
```

### Install boost with python3

We strongly recommand that you install boost libraries from the **source code**
(not apt-get, etc...).

- Download boost sources with the version over 1.67.(https://www.boost.org/users/history/version_1_67_0.html)

- Compile and Install the sources

```bash
cd /path/to/boost_1_xx/
./bootstrap.sh --with-python=python3
sudo ./b2 --with-python --with-filesystem --with-system --with-regex install
```

- Check yourself that the libraries are installed well in your directory `/usr/local/`. (or `/usr/`)

If installed successfully, you should have something like

Include

* `/usr/local/include/boost/`
* `/usr/local/include/boost/python/`
* `/usr/local/include/boost/python/numpy`

Lib 

* `/usr/local/lib/libboost_filesystem.so`
* `/usr/local/lib/libboost_python3.so`
* `/usr/local/lib/libboost_numpy3.so`


### Install PIP things

You should first activate virtualenv.
```bash
virtualenv /path/to/venv --python=python3
source /path/to/venv/bin/activate
```

- pytorch 
```bash
pip install --upgrade pytorch
```

- numpy, matplotlib

```bash
pip3 install numpy matplotlib ipython
```

## How to compile and run

```bash
mkdir build
cd build
cmake ..
make -j8
```

- Run python code (Learning code)
```bash
cd ../pymss
source /path/to/virtualenv/
python3 main.py --option --path
```
- Run UI
```bash
./render/render --metadata
```

- Run Trained data
```bash
./render/render --metadata --network
```
