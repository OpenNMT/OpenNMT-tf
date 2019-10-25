#!/bin/bash

if [ $# -eq 0 ]; then
    echo "usage: $0 X.Y.Z"
    exit 1
fi

set -e

VERSION=$1
PUSH=${2:-0}
BUILD_OPTIONS="--copt=-mavx"

curl -fSsL -O https://github.com/tensorflow/serving/archive/$VERSION.tar.gz
tar xf $VERSION.tar.gz
rm $VERSION.tar.gz
cp *.patch serving-$VERSION
cd serving-$VERSION
patch -p1 < serving_dockerfile.patch

docker build -t opennmt/tensorflow-serving:$VERSION-devel \
       --build-arg TF_SERVING_VERSION_GIT_BRANCH=$VERSION \
       --build-arg TF_SERVING_BUILD_OPTIONS=$BUILD_OPTIONS \
       -f tensorflow_serving/tools/docker/Dockerfile.devel .
docker build -t opennmt/tensorflow-serving:$VERSION \
       --build-arg TF_SERVING_BUILD_IMAGE=opennmt/tensorflow-serving:$VERSION-devel \
       -f tensorflow_serving/tools/docker/Dockerfile .
if [ $PUSH -eq 1 ]; then
    docker push opennmt/tensorflow-serving:$VERSION
fi

docker build -t opennmt/tensorflow-serving:$VERSION-devel-gpu \
       --build-arg TF_SERVING_VERSION_GIT_BRANCH=$VERSION \
       --build-arg TF_SERVING_BUILD_OPTIONS=$BUILD_OPTIONS \
       -f tensorflow_serving/tools/docker/Dockerfile.devel-gpu .
docker build -t opennmt/tensorflow-serving:$VERSION-gpu \
       --build-arg TF_SERVING_BUILD_IMAGE=opennmt/tensorflow-serving:$VERSION-devel-gpu \
       -f tensorflow_serving/tools/docker/Dockerfile.gpu .
if [ $PUSH -eq 1 ]; then
    docker push opennmt/tensorflow-serving:$VERSION-gpu
fi

cd ..
rm -r serving-$VERSION
