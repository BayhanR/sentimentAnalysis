from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)

# Modeli yükleme
model = tf.keras.models.load_model('model.keras')

# Etiket kodlayıcıyı da yüklemelisiniz, bu durumda bir dosyaya kaydettiğinizi varsayalım
le = LabelEncoder()
le.classes_ = np.load('classes.npy', allow_pickle=True)


@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.json
    df = pd.DataFrame(json_data)
    X_new = df.values

    # Model ile tahmin yapma
    predictions = model.predict(X_new)
    predicted_classes = le.inverse_transform(predictions.argmax(axis=1)).tolist()

    return jsonify(predicted_classes=predicted_classes)


if __name__ == '__main__':
    app.run(debug=True)