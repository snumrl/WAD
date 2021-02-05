#!/usr/bin/env bash

ENVDIR=${ENVDIR:-~/pkgenv}
SRCDIR=${SRCDIR:-~/pkgsrc}

mkdir -p $ENVDIR
mkdir -p $ENVDIR/include
mkdir -p $ENVDIR/lib
mkdir -p $ENVDIR/lib/cmake
mkdir -p $SRCDIR

install_library() {
    git clone --depth 1 --branch $3 $2 $1
    echo "==== Installing $1 at $ENVDIR ===="
    mkdir $1/build
    pushd $1/build
    if [ -f ../CMakeLists.txt ]; then
        cmake -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_PREFIX_PATH=$ENVDIR \
              -DCMAKE_INSTALL_PREFIX=$ENVDIR \
              -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
              -DCMAKE_INSTALL_RPATH=$ENVDIR \
              $4 \
              ..
    elif [ -f Makefile ]; then
        cd ..
    fi
    make -j$(nproc) install
    popd
}


install_tbb() {
    local pkgver=$1
    wget https://github.com/oneapi-src/oneTBB/archive/v$pkgver/tbb-$pkgver.tar.gz
    tar -xvzf tbb-$pkgver.tar.gz
    rm tbb-$pkgver.tar.gz

    pushd oneTBB-$pkgver
    pushd src
    make -j$(nproc) tbb tbbmalloc
    popd
    install -Dm755 build/linux_*/*.so* -t $ENVDIR/lib
    install -d $ENVDIR/include
    cp -a include/tbb $ENVDIR/include
    cmake \
        -DINSTALL_DIR=$ENVDIR/lib/cmake/TBB \
        -DSYSTEM_NAME=Linux -DTBB_VERSION_FILE=$ENVDIR/include/tbb/tbb_stddef.h \
        -P cmake/tbb_config_installer.cmake
    popd
}

install_boost() {
    git clone https://github.com/boostorg/boost.git
    pushd boost
    git checkout tags/boost-1.66.0
    ./bootstrap.sh --with-python=python3 --prefix=$ENVDIR
    ./b2 --with-python --with-filesystem --with-system --with-regex install
    popd
}

pushd $SRCDIR

install_tbb 2020.3
install_boost
install_library tinyxml2 https://github.com/leethomason/tinyxml2 8.0.0
install_library libccd https://github.com/danfis/libccd v2.0
install_library assimp https://github.com/assimp/assimp v4.0.1
install_library octomap https://github.com/OctoMap/octomap v1.8.1
install_library dart https://github.com/dartsim/dart v6.3.0 \

popd
