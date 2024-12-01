import tensorflow as tf
import subprocess

# One-Hotモデル定義
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
            # tf.constant([0, 2, -1, 1], dtype=tf.int32),
            indices=indices,
            depth=self.depth,
            on_value=self.on_value,
            off_value=self.off_value,
            axis=self.axis
        )
# モデルの初期化
depth = 3  # One-Hotエンコーディングの深さ
model = OneHotModel(depth=depth, on_value=1.0, off_value=0.0, axis=-1)

# ダミー入力
dummy_input = tf.constant([0, 2, 3, 1], dtype=tf.int32)  # 入力を整数型に変更
print(dummy_input)
output = model(dummy_input)  # 出力を確認
print("TensorFlowモデル出力:\n", output)

# モデルの保存
saved_model_dir = "one_hot_saved_model"
model_name = "one_hot.onnx"
tf.saved_model.save(model, saved_model_dir)
print("モデルを保存しました。")
# ONNXへの変換
# TensorFlowモデルをONNXに変換するためのコマンド
command = [
    "python", "-m", "tf2onnx.convert",
    "--saved-model", saved_model_dir,  # 保存したモデルのパス
    "--output", model_name,  # 出力するONNXファイルのパス
    "--opset", "16",  # 使用するONNXのバージョン
]

# コマンドを実行
result = subprocess.run(command, capture_output=True, text=True)

if result.returncode == 0:
    print("ONNXモデルの変換に成功しました。")
else:
    print("ONNXモデルの変換に失敗しました。")
    print(result.stderr)