import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

if __name__ == '__main__':

    tidal_train = pd.read_csv("./DATA/DATA.csv")

    tidal_features = tidal_train.drop(['GRADE', 'STUDENT ID'], axis=1)
    tidal_labels = tidal_train.pop('GRADE')
    print(tidal_labels)

    tidal_features = np.array(tidal_features)
    tidal_labels = np.array(tidal_labels)
    tidal_features = tidal_features.astype(np.int32)  # Convert to tf.float32
    tidal_labels = tidal_labels.astype(np.int32)  # Convert to tf.int32

    # Split the data into training and testing sets
    # Adjust test_size and random_state as needed
    features_train, features_test, labels_train, labels_test = train_test_split(
        tidal_features, tidal_labels, test_size=0.2, random_state=42, stratify=tidal_labels)
    print(features_train.shape, features_test.shape, labels_train.shape, labels_test.shape)
    categorical_features = range(tidal_features.shape[1])

    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    features_train_encoded = encoder.fit_transform(features_train)
    features_test_encoded = encoder.transform(features_test)

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(features_train_encoded.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # Assuming 10 classes, adjust accordingly
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['categorical_accuracy'])

    # Train the model
    model.fit(features_train_encoded, labels_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(features_test_encoded, features_test)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
