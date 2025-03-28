import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.preprocessing import LabelEncoder


def train_model(train_file_path: str, test_file_path: str) -> None:
    # Veriyi yükleme
    train_df = pd.read_csv("train_dataset.csv")
    test_df = pd.read_csv("test_dataset.csv")

    # Özellikler ve etiketler
    X_train = train_df.drop(columns=['label']).values
    y_train = train_df['label'].values
    X_test = test_df.drop(columns=['label']).values
    y_test = test_df['label'].values

    # Etiketleri sayısallaştırma
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Modeli oluşturma
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(3, activation='softmax')
    ])

    # Modeli derleme
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Modeli eğitme
    model.fit(X_train, y_train_encoded, epochs=10, validation_data=(X_test, y_test_encoded))

    # Test seti ile modelin doğruluğunu değerlendirme
    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
    print(f'Test accuracy: {test_accuracy}')
    tf.keras.models.save_model(model, 'model.keras')

if __name__ == "__main__":
    train_file_path = "train_dataset.csv"
    test_file_path = "test_dataset.csv"
    train_model(train_file_path, test_file_path)