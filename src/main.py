import tensorflow as tf
import tf2onnx
import subprocess

# 簡単なTensorFlowモデルを定義
class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense = tf.keras.layers.Dense(10, activation='relu')
    
    def call(self, inputs):
        return self.dense(inputs)

# モデルインスタンスを作成
model = SimpleModel()

# モデルの保存ディレクトリ
saved_model_dir = "simple_model"

# モデルを保存する
input_shape = (1, 5)  # 入力の形状 (バッチサイズ, 入力特徴数)
dummy_input = tf.random.normal(input_shape)
model(dummy_input)  # モデルを呼び出して初期化
tf.saved_model.save(model, saved_model_dir)

# ONNXへの変換

# TensorFlowモデルをONNXに変換するためのコマンド
command = [
    "python", "-m", "tf2onnx.convert",
    "--saved-model", "simple_model",  # 保存したモデルのパス
    "--output", "simple_model.onnx",  # 出力するONNXファイルのパス
    "--opset", "13"  # 使用するONNXのバージョン
]

# コマンドを実行
result = subprocess.run(command, capture_output=True, text=True)

if result.returncode == 0:
    print("ONNXモデルの変換に成功しました。")
else:
    print("ONNXモデルの変換に失敗しました。")
    print(result.stderr)