import tensorflow as tf
import subprocess

class OneHotModel(tf.keras.Model):
    def __init__(self, depth, on_value=1.0, off_value=0.0, axis=-1):
        super(OneHotModel, self).__init__()
        self.depth = depth
        self.on_value = on_value
        self.off_value = off_value
        self.axis = axis
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)])
    def __call__(self, indices):
        return tf.one_hot(
            indices=indices,
            depth=self.depth,
            on_value=self.on_value,
            off_value=self.off_value,
            axis=self.axis
        )
# Initialize model
depth = 3
model = OneHotModel(depth=depth, on_value=1.0, off_value=0.0, axis=-1)

# Dummy Inputs
dummy_input = tf.constant([0, 2, 3, 1], dtype=tf.int32)
print("Dummy Inputs", dummy_input)
output = model(dummy_input)
print("TensorFlow Ouputs:\n", output)

saved_model_dir = "one_hot_saved_model"
model_name = "one_hot.onnx"
tf.saved_model.save(model, saved_model_dir)
print("Model is Saved")

# ONNX Conversion
command = [
    "python", "-m", "tf2onnx.convert",
    "--saved-model", saved_model_dir,
    "--output", model_name,
    "--opset", "16"
]

result = subprocess.run(command, capture_output=True, text=True)

if result.returncode == 0:
    print("ONNX Conversion succeeded.")
else:
    print("ONNX Conversion failed.")
    print(result.stderr)