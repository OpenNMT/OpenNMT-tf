TF_DIR=${HOME}/tensorflow

CXXFLAGS=-std=c++11 \
	-I${TF_DIR} \
	-I${TF_DIR}/bazel-genfiles \
	-I${TF_DIR}/bazel-tensorflow/external/eigen_archive \
	-I${TF_DIR}/bazel-tensorflow/external/protobuf_archive/src \
	-I${TF_DIR}/bazel-tensorflow/external/nsync/public/
LDFLAGS=-L${TF_DIR}/bazel-bin/tensorflow
LDLIBS=-ltensorflow_opennmt -ltensorflow_framework

TARGET=main

all: ${TARGET}

${TARGET}: main.cc

clean:
	${RM} ${TARGET}
