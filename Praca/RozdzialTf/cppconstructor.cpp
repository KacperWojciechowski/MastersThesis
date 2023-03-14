#include <string>

#include "third_party/tensorflow/core/framework/graph.proto.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/lib/io/path.h"
#include "third_party/tensorflow/core/platform/env.h"
#include "third_party/tensorflow/core/platform/init_main.h"
#include "third_party/tensorflow/core/platform/logging.h"
#include "third_party/tensorflow/core/platform/types.h"
#include "third_party/tensorflow/core/public/session.h"

Model(const std::string& graph_def_filename) {
    tensorflow::GraphDef graph_def;
    TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                            graph_def_filename, &graph_def));
    session_.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_CHECK_OK(session_->Create(graph_def));
  }