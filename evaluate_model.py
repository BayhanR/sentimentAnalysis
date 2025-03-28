import numpy as np
import tensorflow as tf

X_test = np.random.rand(200, 10)
y_test = np.random.randint(2, size=(200, 1))
model =  tf.keras.models.load_model('model.keras')
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Test Kaybı: {loss}, Test Doğruluğu: {accuracy}')