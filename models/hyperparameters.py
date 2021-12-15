class SpeakerVerifierHyperparameters:
    conv2d_in_channels = 1
    conv2d_out_channels = 64
    conv2d_kernel_size = (20, 5)
    conv2d_strides = (8, 2)

    recurrent_hidden_size = 128
    fully_connected_size = 128
    dropout_probability = 0.2

    learning_rate = 1e-3
    max_gradient_norm = 100
    gradient_clipping_max_value = 5

    batch_size = 64
    examples_evaluate = 1000