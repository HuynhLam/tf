node {
  name: "input"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 784
        }
      }
    }
  }
}
node {
  name: "input_layer/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\377\377\377\377\034\000\000\000\034\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "input_layer"
  op: "Reshape"
  input: "input"
  input: "input_layer/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "conv_layer_1/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000\001\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "conv_layer_1/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.0852802842855
      }
    }
  }
}
node {
  name: "conv_layer_1/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0852802842855
      }
    }
  }
}
node {
  name: "conv_layer_1/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "conv_layer_1/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "conv_layer_1/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "conv_layer_1/kernel/Initializer/random_uniform/max"
  input: "conv_layer_1/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/kernel"
      }
    }
  }
}
node {
  name: "conv_layer_1/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "conv_layer_1/kernel/Initializer/random_uniform/RandomUniform"
  input: "conv_layer_1/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/kernel"
      }
    }
  }
}
node {
  name: "conv_layer_1/kernel/Initializer/random_uniform"
  op: "Add"
  input: "conv_layer_1/kernel/Initializer/random_uniform/mul"
  input: "conv_layer_1/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/kernel"
      }
    }
  }
}
node {
  name: "conv_layer_1/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 1
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "conv_layer_1/kernel/Assign"
  op: "Assign"
  input: "conv_layer_1/kernel"
  input: "conv_layer_1/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "conv_layer_1/kernel/read"
  op: "Identity"
  input: "conv_layer_1/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/kernel"
      }
    }
  }
}
node {
  name: "conv_layer_1/bias/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "conv_layer_1/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "conv_layer_1/bias/Assign"
  op: "Assign"
  input: "conv_layer_1/bias"
  input: "conv_layer_1/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "conv_layer_1/bias/read"
  op: "Identity"
  input: "conv_layer_1/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/bias"
      }
    }
  }
}
node {
  name: "conv_layer_1/convolution/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000\001\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "conv_layer_1/convolution/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "conv_layer_1/convolution"
  op: "Conv2D"
  input: "input_layer"
  input: "conv_layer_1/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "conv_layer_1/BiasAdd"
  op: "BiasAdd"
  input: "conv_layer_1/convolution"
  input: "conv_layer_1/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "conv_layer_1/Relu"
  op: "Relu"
  input: "conv_layer_1/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "max_pooling_1/MaxPool"
  op: "MaxPool"
  input: "conv_layer_1/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "conv_layer_2/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "conv_layer_2/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.0500000007451
      }
    }
  }
}
node {
  name: "conv_layer_2/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0500000007451
      }
    }
  }
}
node {
  name: "conv_layer_2/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "conv_layer_2/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "conv_layer_2/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "conv_layer_2/kernel/Initializer/random_uniform/max"
  input: "conv_layer_2/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/kernel"
      }
    }
  }
}
node {
  name: "conv_layer_2/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "conv_layer_2/kernel/Initializer/random_uniform/RandomUniform"
  input: "conv_layer_2/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/kernel"
      }
    }
  }
}
node {
  name: "conv_layer_2/kernel/Initializer/random_uniform"
  op: "Add"
  input: "conv_layer_2/kernel/Initializer/random_uniform/mul"
  input: "conv_layer_2/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/kernel"
      }
    }
  }
}
node {
  name: "conv_layer_2/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 5
        }
        dim {
          size: 5
        }
        dim {
          size: 32
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "conv_layer_2/kernel/Assign"
  op: "Assign"
  input: "conv_layer_2/kernel"
  input: "conv_layer_2/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "conv_layer_2/kernel/read"
  op: "Identity"
  input: "conv_layer_2/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/kernel"
      }
    }
  }
}
node {
  name: "conv_layer_2/bias/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "conv_layer_2/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "conv_layer_2/bias/Assign"
  op: "Assign"
  input: "conv_layer_2/bias"
  input: "conv_layer_2/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "conv_layer_2/bias/read"
  op: "Identity"
  input: "conv_layer_2/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/bias"
      }
    }
  }
}
node {
  name: "conv_layer_2/convolution/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "conv_layer_2/convolution/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "conv_layer_2/convolution"
  op: "Conv2D"
  input: "max_pooling_1/MaxPool"
  input: "conv_layer_2/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "conv_layer_2/BiasAdd"
  op: "BiasAdd"
  input: "conv_layer_2/convolution"
  input: "conv_layer_2/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "conv_layer_2/Relu"
  op: "Relu"
  input: "conv_layer_2/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "max_pooling_2/MaxPool"
  op: "MaxPool"
  input: "conv_layer_2/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "flatten_before_dense/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377@\014\000\000"
      }
    }
  }
}
node {
  name: "flatten_before_dense"
  op: "Reshape"
  input: "max_pooling_2/MaxPool"
  input: "flatten_before_dense/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "first_dense/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "@\014\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "first_dense/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.043301269412
      }
    }
  }
}
node {
  name: "first_dense/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.043301269412
      }
    }
  }
}
node {
  name: "first_dense/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "first_dense/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "first_dense/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "first_dense/kernel/Initializer/random_uniform/max"
  input: "first_dense/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/kernel"
      }
    }
  }
}
node {
  name: "first_dense/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "first_dense/kernel/Initializer/random_uniform/RandomUniform"
  input: "first_dense/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/kernel"
      }
    }
  }
}
node {
  name: "first_dense/kernel/Initializer/random_uniform"
  op: "Add"
  input: "first_dense/kernel/Initializer/random_uniform/mul"
  input: "first_dense/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/kernel"
      }
    }
  }
}
node {
  name: "first_dense/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3136
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "first_dense/kernel/Assign"
  op: "Assign"
  input: "first_dense/kernel"
  input: "first_dense/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "first_dense/kernel/read"
  op: "Identity"
  input: "first_dense/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/kernel"
      }
    }
  }
}
node {
  name: "first_dense/bias/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "first_dense/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "first_dense/bias/Assign"
  op: "Assign"
  input: "first_dense/bias"
  input: "first_dense/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "first_dense/bias/read"
  op: "Identity"
  input: "first_dense/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/bias"
      }
    }
  }
}
node {
  name: "first_dense/MatMul"
  op: "MatMul"
  input: "flatten_before_dense"
  input: "first_dense/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "first_dense/BiasAdd"
  op: "BiasAdd"
  input: "first_dense/MatMul"
  input: "first_dense/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "first_dense/Relu"
  op: "Relu"
  input: "first_dense/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dropout1/Identity"
  op: "Identity"
  input: "first_dense/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "last_dense/kernel/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "@\000\000\000\n\000\000\000"
      }
    }
  }
}
node {
  name: "last_dense/kernel/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.284747391939
      }
    }
  }
}
node {
  name: "last_dense/kernel/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.284747391939
      }
    }
  }
}
node {
  name: "last_dense/kernel/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "last_dense/kernel/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "last_dense/kernel/Initializer/random_uniform/sub"
  op: "Sub"
  input: "last_dense/kernel/Initializer/random_uniform/max"
  input: "last_dense/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/kernel"
      }
    }
  }
}
node {
  name: "last_dense/kernel/Initializer/random_uniform/mul"
  op: "Mul"
  input: "last_dense/kernel/Initializer/random_uniform/RandomUniform"
  input: "last_dense/kernel/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/kernel"
      }
    }
  }
}
node {
  name: "last_dense/kernel/Initializer/random_uniform"
  op: "Add"
  input: "last_dense/kernel/Initializer/random_uniform/mul"
  input: "last_dense/kernel/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/kernel"
      }
    }
  }
}
node {
  name: "last_dense/kernel"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
        dim {
          size: 10
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "last_dense/kernel/Assign"
  op: "Assign"
  input: "last_dense/kernel"
  input: "last_dense/kernel/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "last_dense/kernel/read"
  op: "Identity"
  input: "last_dense/kernel"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/kernel"
      }
    }
  }
}
node {
  name: "last_dense/bias/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 10
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "last_dense/bias"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/bias"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 10
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "last_dense/bias/Assign"
  op: "Assign"
  input: "last_dense/bias"
  input: "last_dense/bias/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "last_dense/bias/read"
  op: "Identity"
  input: "last_dense/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/bias"
      }
    }
  }
}
node {
  name: "last_dense/MatMul"
  op: "MatMul"
  input: "dropout1/Identity"
  input: "last_dense/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "last_dense/BiasAdd"
  op: "BiasAdd"
  input: "last_dense/MatMul"
  input: "last_dense/bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "Y_labels"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 10
        }
      }
    }
  }
}
node {
  name: "Rank"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "Shape"
  op: "Shape"
  input: "last_dense/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Rank_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "Shape_1"
  op: "Shape"
  input: "last_dense/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Sub/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Sub"
  op: "Sub"
  input: "Rank_1"
  input: "Sub/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Slice/begin"
  op: "Pack"
  input: "Sub"
  attr {
    key: "N"
    value {
      i: 1
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "Slice/size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Slice"
  op: "Slice"
  input: "Shape_1"
  input: "Slice/begin"
  input: "Slice/size"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "concat/values_0"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "concat/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "concat"
  op: "ConcatV2"
  input: "concat/values_0"
  input: "Slice"
  input: "concat/axis"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Reshape"
  op: "Reshape"
  input: "last_dense/BiasAdd"
  input: "concat"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Rank_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "Shape_2"
  op: "Shape"
  input: "Y_labels"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Sub_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Sub_1"
  op: "Sub"
  input: "Rank_2"
  input: "Sub_1/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Slice_1/begin"
  op: "Pack"
  input: "Sub_1"
  attr {
    key: "N"
    value {
      i: 1
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "Slice_1/size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Slice_1"
  op: "Slice"
  input: "Shape_2"
  input: "Slice_1/begin"
  input: "Slice_1/size"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "concat_1/values_0"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "concat_1/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "concat_1"
  op: "ConcatV2"
  input: "concat_1/values_0"
  input: "Slice_1"
  input: "concat_1/axis"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Reshape_1"
  op: "Reshape"
  input: "Y_labels"
  input: "concat_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "SoftmaxCrossEntropyWithLogits"
  op: "SoftmaxCrossEntropyWithLogits"
  input: "Reshape"
  input: "Reshape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Sub_2/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "Sub_2"
  op: "Sub"
  input: "Rank"
  input: "Sub_2/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Slice_2/begin"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "Slice_2/size"
  op: "Pack"
  input: "Sub_2"
  attr {
    key: "N"
    value {
      i: 1
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "Slice_2"
  op: "Slice"
  input: "Shape"
  input: "Slice_2/begin"
  input: "Slice_2/size"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Reshape_2"
  op: "Reshape"
  input: "SoftmaxCrossEntropyWithLogits"
  input: "Slice_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "Mean"
  op: "Mean"
  input: "Reshape_2"
  input: "Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "gradients/Fill"
  op: "Fill"
  input: "gradients/Shape"
  input: "gradients/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Mean_grad/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "gradients/Mean_grad/Reshape"
  op: "Reshape"
  input: "gradients/Fill"
  input: "gradients/Mean_grad/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Mean_grad/Shape"
  op: "Shape"
  input: "Reshape_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Mean_grad/Tile"
  op: "Tile"
  input: "gradients/Mean_grad/Reshape"
  input: "gradients/Mean_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Mean_grad/Shape_1"
  op: "Shape"
  input: "Reshape_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Mean_grad/Shape_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients/Mean_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "gradients/Mean_grad/Prod"
  op: "Prod"
  input: "gradients/Mean_grad/Shape_1"
  input: "gradients/Mean_grad/Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/Mean_grad/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "gradients/Mean_grad/Prod_1"
  op: "Prod"
  input: "gradients/Mean_grad/Shape_2"
  input: "gradients/Mean_grad/Const_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/Mean_grad/Maximum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "gradients/Mean_grad/Maximum"
  op: "Maximum"
  input: "gradients/Mean_grad/Prod_1"
  input: "gradients/Mean_grad/Maximum/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Mean_grad/floordiv"
  op: "FloorDiv"
  input: "gradients/Mean_grad/Prod"
  input: "gradients/Mean_grad/Maximum"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Mean_grad/Cast"
  op: "Cast"
  input: "gradients/Mean_grad/floordiv"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Mean_grad/truediv"
  op: "RealDiv"
  input: "gradients/Mean_grad/Tile"
  input: "gradients/Mean_grad/Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Reshape_2_grad/Shape"
  op: "Shape"
  input: "SoftmaxCrossEntropyWithLogits"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Reshape_2_grad/Reshape"
  op: "Reshape"
  input: "gradients/Mean_grad/truediv"
  input: "gradients/Reshape_2_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/zeros_like"
  op: "ZerosLike"
  input: "SoftmaxCrossEntropyWithLogits:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims"
  op: "ExpandDims"
  input: "gradients/Reshape_2_grad/Reshape"
  input: "gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/SoftmaxCrossEntropyWithLogits_grad/mul"
  op: "Mul"
  input: "gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims"
  input: "SoftmaxCrossEntropyWithLogits:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Reshape_grad/Shape"
  op: "Shape"
  input: "last_dense/BiasAdd"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Reshape_grad/Reshape"
  op: "Reshape"
  input: "gradients/SoftmaxCrossEntropyWithLogits_grad/mul"
  input: "gradients/Reshape_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/last_dense/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "gradients/Reshape_grad/Reshape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "gradients/last_dense/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/Reshape_grad/Reshape"
  input: "^gradients/last_dense/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "gradients/last_dense/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/Reshape_grad/Reshape"
  input: "^gradients/last_dense/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/Reshape_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/last_dense/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/last_dense/BiasAdd_grad/BiasAddGrad"
  input: "^gradients/last_dense/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/last_dense/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "gradients/last_dense/MatMul_grad/MatMul"
  op: "MatMul"
  input: "gradients/last_dense/BiasAdd_grad/tuple/control_dependency"
  input: "last_dense/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/last_dense/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "dropout1/Identity"
  input: "gradients/last_dense/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/last_dense/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/last_dense/MatMul_grad/MatMul"
  input: "^gradients/last_dense/MatMul_grad/MatMul_1"
}
node {
  name: "gradients/last_dense/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/last_dense/MatMul_grad/MatMul"
  input: "^gradients/last_dense/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/last_dense/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "gradients/last_dense/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/last_dense/MatMul_grad/MatMul_1"
  input: "^gradients/last_dense/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/last_dense/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "gradients/first_dense/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "gradients/last_dense/MatMul_grad/tuple/control_dependency"
  input: "first_dense/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/first_dense/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "gradients/first_dense/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "gradients/first_dense/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/first_dense/Relu_grad/ReluGrad"
  input: "^gradients/first_dense/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "gradients/first_dense/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/first_dense/Relu_grad/ReluGrad"
  input: "^gradients/first_dense/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/first_dense/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "gradients/first_dense/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/first_dense/BiasAdd_grad/BiasAddGrad"
  input: "^gradients/first_dense/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/first_dense/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "gradients/first_dense/MatMul_grad/MatMul"
  op: "MatMul"
  input: "gradients/first_dense/BiasAdd_grad/tuple/control_dependency"
  input: "first_dense/kernel/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/first_dense/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "flatten_before_dense"
  input: "gradients/first_dense/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/first_dense/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/first_dense/MatMul_grad/MatMul"
  input: "^gradients/first_dense/MatMul_grad/MatMul_1"
}
node {
  name: "gradients/first_dense/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/first_dense/MatMul_grad/MatMul"
  input: "^gradients/first_dense/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/first_dense/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "gradients/first_dense/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/first_dense/MatMul_grad/MatMul_1"
  input: "^gradients/first_dense/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/first_dense/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "gradients/flatten_before_dense_grad/Shape"
  op: "Shape"
  input: "max_pooling_2/MaxPool"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/flatten_before_dense_grad/Reshape"
  op: "Reshape"
  input: "gradients/first_dense/MatMul_grad/tuple/control_dependency"
  input: "gradients/flatten_before_dense_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/max_pooling_2/MaxPool_grad/MaxPoolGrad"
  op: "MaxPoolGrad"
  input: "conv_layer_2/Relu"
  input: "max_pooling_2/MaxPool"
  input: "gradients/flatten_before_dense_grad/Reshape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "gradients/conv_layer_2/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "gradients/max_pooling_2/MaxPool_grad/MaxPoolGrad"
  input: "conv_layer_2/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/conv_layer_2/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "gradients/conv_layer_2/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "gradients/conv_layer_2/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/conv_layer_2/Relu_grad/ReluGrad"
  input: "^gradients/conv_layer_2/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "gradients/conv_layer_2/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/conv_layer_2/Relu_grad/ReluGrad"
  input: "^gradients/conv_layer_2/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/conv_layer_2/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "gradients/conv_layer_2/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/conv_layer_2/BiasAdd_grad/BiasAddGrad"
  input: "^gradients/conv_layer_2/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/conv_layer_2/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "gradients/conv_layer_2/convolution_grad/Shape"
  op: "Shape"
  input: "max_pooling_1/MaxPool"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/conv_layer_2/convolution_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "gradients/conv_layer_2/convolution_grad/Shape"
  input: "conv_layer_2/kernel/read"
  input: "gradients/conv_layer_2/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/conv_layer_2/convolution_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000 \000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "gradients/conv_layer_2/convolution_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "max_pooling_1/MaxPool"
  input: "gradients/conv_layer_2/convolution_grad/Shape_1"
  input: "gradients/conv_layer_2/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/conv_layer_2/convolution_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/conv_layer_2/convolution_grad/Conv2DBackpropInput"
  input: "^gradients/conv_layer_2/convolution_grad/Conv2DBackpropFilter"
}
node {
  name: "gradients/conv_layer_2/convolution_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/conv_layer_2/convolution_grad/Conv2DBackpropInput"
  input: "^gradients/conv_layer_2/convolution_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/conv_layer_2/convolution_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "gradients/conv_layer_2/convolution_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/conv_layer_2/convolution_grad/Conv2DBackpropFilter"
  input: "^gradients/conv_layer_2/convolution_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/conv_layer_2/convolution_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "gradients/max_pooling_1/MaxPool_grad/MaxPoolGrad"
  op: "MaxPoolGrad"
  input: "conv_layer_1/Relu"
  input: "max_pooling_1/MaxPool"
  input: "gradients/conv_layer_2/convolution_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "gradients/conv_layer_1/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "gradients/max_pooling_1/MaxPool_grad/MaxPoolGrad"
  input: "conv_layer_1/Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/conv_layer_1/BiasAdd_grad/BiasAddGrad"
  op: "BiasAddGrad"
  input: "gradients/conv_layer_1/Relu_grad/ReluGrad"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "gradients/conv_layer_1/BiasAdd_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/conv_layer_1/Relu_grad/ReluGrad"
  input: "^gradients/conv_layer_1/BiasAdd_grad/BiasAddGrad"
}
node {
  name: "gradients/conv_layer_1/BiasAdd_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/conv_layer_1/Relu_grad/ReluGrad"
  input: "^gradients/conv_layer_1/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/conv_layer_1/Relu_grad/ReluGrad"
      }
    }
  }
}
node {
  name: "gradients/conv_layer_1/BiasAdd_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/conv_layer_1/BiasAdd_grad/BiasAddGrad"
  input: "^gradients/conv_layer_1/BiasAdd_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/conv_layer_1/BiasAdd_grad/BiasAddGrad"
      }
    }
  }
}
node {
  name: "gradients/conv_layer_1/convolution_grad/Shape"
  op: "Shape"
  input: "input_layer"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/conv_layer_1/convolution_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "gradients/conv_layer_1/convolution_grad/Shape"
  input: "conv_layer_1/kernel/read"
  input: "gradients/conv_layer_1/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/conv_layer_1/convolution_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\005\000\000\000\005\000\000\000\001\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "gradients/conv_layer_1/convolution_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "input_layer"
  input: "gradients/conv_layer_1/convolution_grad/Shape_1"
  input: "gradients/conv_layer_1/BiasAdd_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/conv_layer_1/convolution_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/conv_layer_1/convolution_grad/Conv2DBackpropInput"
  input: "^gradients/conv_layer_1/convolution_grad/Conv2DBackpropFilter"
}
node {
  name: "gradients/conv_layer_1/convolution_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/conv_layer_1/convolution_grad/Conv2DBackpropInput"
  input: "^gradients/conv_layer_1/convolution_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/conv_layer_1/convolution_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "gradients/conv_layer_1/convolution_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/conv_layer_1/convolution_grad/Conv2DBackpropFilter"
  input: "^gradients/conv_layer_1/convolution_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/conv_layer_1/convolution_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "GradientDescent/learning_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.5
      }
    }
  }
}
node {
  name: "GradientDescent/update_conv_layer_1/kernel/ApplyGradientDescent"
  op: "ApplyGradientDescent"
  input: "conv_layer_1/kernel"
  input: "GradientDescent/learning_rate"
  input: "gradients/conv_layer_1/convolution_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "GradientDescent/update_conv_layer_1/bias/ApplyGradientDescent"
  op: "ApplyGradientDescent"
  input: "conv_layer_1/bias"
  input: "GradientDescent/learning_rate"
  input: "gradients/conv_layer_1/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "GradientDescent/update_conv_layer_2/kernel/ApplyGradientDescent"
  op: "ApplyGradientDescent"
  input: "conv_layer_2/kernel"
  input: "GradientDescent/learning_rate"
  input: "gradients/conv_layer_2/convolution_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "GradientDescent/update_conv_layer_2/bias/ApplyGradientDescent"
  op: "ApplyGradientDescent"
  input: "conv_layer_2/bias"
  input: "GradientDescent/learning_rate"
  input: "gradients/conv_layer_2/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "GradientDescent/update_first_dense/kernel/ApplyGradientDescent"
  op: "ApplyGradientDescent"
  input: "first_dense/kernel"
  input: "GradientDescent/learning_rate"
  input: "gradients/first_dense/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "GradientDescent/update_first_dense/bias/ApplyGradientDescent"
  op: "ApplyGradientDescent"
  input: "first_dense/bias"
  input: "GradientDescent/learning_rate"
  input: "gradients/first_dense/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "GradientDescent/update_last_dense/kernel/ApplyGradientDescent"
  op: "ApplyGradientDescent"
  input: "last_dense/kernel"
  input: "GradientDescent/learning_rate"
  input: "gradients/last_dense/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "GradientDescent/update_last_dense/bias/ApplyGradientDescent"
  op: "ApplyGradientDescent"
  input: "last_dense/bias"
  input: "GradientDescent/learning_rate"
  input: "gradients/last_dense/BiasAdd_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "GradientDescent"
  op: "NoOp"
  input: "^GradientDescent/update_conv_layer_1/kernel/ApplyGradientDescent"
  input: "^GradientDescent/update_conv_layer_1/bias/ApplyGradientDescent"
  input: "^GradientDescent/update_conv_layer_2/kernel/ApplyGradientDescent"
  input: "^GradientDescent/update_conv_layer_2/bias/ApplyGradientDescent"
  input: "^GradientDescent/update_first_dense/kernel/ApplyGradientDescent"
  input: "^GradientDescent/update_first_dense/bias/ApplyGradientDescent"
  input: "^GradientDescent/update_last_dense/kernel/ApplyGradientDescent"
  input: "^GradientDescent/update_last_dense/bias/ApplyGradientDescent"
}
node {
  name: "init"
  op: "NoOp"
  input: "^conv_layer_1/kernel/Assign"
  input: "^conv_layer_1/bias/Assign"
  input: "^conv_layer_2/kernel/Assign"
  input: "^conv_layer_2/bias/Assign"
  input: "^first_dense/kernel/Assign"
  input: "^first_dense/bias/Assign"
  input: "^last_dense/kernel/Assign"
  input: "^last_dense/bias/Assign"
}
node {
  name: "prediction/dimension"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "prediction"
  op: "ArgMax"
  input: "last_dense/BiasAdd"
  input: "prediction/dimension"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "output_type"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "ArgMax/dimension"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "ArgMax"
  op: "ArgMax"
  input: "last_dense/BiasAdd"
  input: "ArgMax/dimension"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "output_type"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "ArgMax_1/dimension"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "ArgMax_1"
  op: "ArgMax"
  input: "Y_labels"
  input: "ArgMax_1/dimension"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "output_type"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "correct_prediction"
  op: "Equal"
  input: "ArgMax"
  input: "ArgMax_1"
  attr {
    key: "T"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "Cast_1"
  op: "Cast"
  input: "correct_prediction"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_BOOL
    }
  }
}
node {
  name: "Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "accuracy"
  op: "Mean"
  input: "Cast_1"
  input: "Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "save/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "model"
      }
    }
  }
}
node {
  name: "save/SaveV2/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 8
          }
        }
        string_val: "conv_layer_1/bias"
        string_val: "conv_layer_1/kernel"
        string_val: "conv_layer_2/bias"
        string_val: "conv_layer_2/kernel"
        string_val: "first_dense/bias"
        string_val: "first_dense/kernel"
        string_val: "last_dense/bias"
        string_val: "last_dense/kernel"
      }
    }
  }
}
node {
  name: "save/SaveV2/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 8
          }
        }
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
      }
    }
  }
}
node {
  name: "save/SaveV2"
  op: "SaveV2"
  input: "save/Const"
  input: "save/SaveV2/tensor_names"
  input: "save/SaveV2/shape_and_slices"
  input: "conv_layer_1/bias"
  input: "conv_layer_1/kernel"
  input: "conv_layer_2/bias"
  input: "conv_layer_2/kernel"
  input: "first_dense/bias"
  input: "first_dense/kernel"
  input: "last_dense/bias"
  input: "last_dense/kernel"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/control_dependency"
  op: "Identity"
  input: "save/Const"
  input: "^save/SaveV2"
  attr {
    key: "T"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@save/Const"
      }
    }
  }
}
node {
  name: "save/RestoreV2/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "conv_layer_1/bias"
      }
    }
  }
}
node {
  name: "save/RestoreV2/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2/tensor_names"
  input: "save/RestoreV2/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign"
  op: "Assign"
  input: "conv_layer_1/bias"
  input: "save/RestoreV2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_1/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "conv_layer_1/kernel"
      }
    }
  }
}
node {
  name: "save/RestoreV2_1/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_1"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_1/tensor_names"
  input: "save/RestoreV2_1/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_1"
  op: "Assign"
  input: "conv_layer_1/kernel"
  input: "save/RestoreV2_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_1/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_2/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "conv_layer_2/bias"
      }
    }
  }
}
node {
  name: "save/RestoreV2_2/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_2"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_2/tensor_names"
  input: "save/RestoreV2_2/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_2"
  op: "Assign"
  input: "conv_layer_2/bias"
  input: "save/RestoreV2_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_3/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "conv_layer_2/kernel"
      }
    }
  }
}
node {
  name: "save/RestoreV2_3/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_3"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_3/tensor_names"
  input: "save/RestoreV2_3/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_3"
  op: "Assign"
  input: "conv_layer_2/kernel"
  input: "save/RestoreV2_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@conv_layer_2/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_4/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "first_dense/bias"
      }
    }
  }
}
node {
  name: "save/RestoreV2_4/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_4"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_4/tensor_names"
  input: "save/RestoreV2_4/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_4"
  op: "Assign"
  input: "first_dense/bias"
  input: "save/RestoreV2_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_5/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "first_dense/kernel"
      }
    }
  }
}
node {
  name: "save/RestoreV2_5/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_5"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_5/tensor_names"
  input: "save/RestoreV2_5/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_5"
  op: "Assign"
  input: "first_dense/kernel"
  input: "save/RestoreV2_5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@first_dense/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_6/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "last_dense/bias"
      }
    }
  }
}
node {
  name: "save/RestoreV2_6/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_6"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_6/tensor_names"
  input: "save/RestoreV2_6/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_6"
  op: "Assign"
  input: "last_dense/bias"
  input: "save/RestoreV2_6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/bias"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_7/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "last_dense/kernel"
      }
    }
  }
}
node {
  name: "save/RestoreV2_7/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_7"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_7/tensor_names"
  input: "save/RestoreV2_7/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_7"
  op: "Assign"
  input: "last_dense/kernel"
  input: "save/RestoreV2_7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@last_dense/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_all"
  op: "NoOp"
  input: "^save/Assign"
  input: "^save/Assign_1"
  input: "^save/Assign_2"
  input: "^save/Assign_3"
  input: "^save/Assign_4"
  input: "^save/Assign_5"
  input: "^save/Assign_6"
  input: "^save/Assign_7"
}
versions {
  producer: 24
}
