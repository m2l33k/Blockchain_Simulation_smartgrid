import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print("  - ", gpu)
else:
    print("!!! WARNING: TensorFlow could not find any GPUs. !!!")
    print("Please check your NVIDIA driver and CUDA installation.")