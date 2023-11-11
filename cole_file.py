import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
    tidal_train = pd.read_csv("./DATA/student-performance.csv")

    # Assume 'G3' is the label and rest are features
    tidal_features = tidal_train.drop(['G3'], axis=1)
    tidal_labels = tidal_train['G3']

    # Identify categorical and numerical columns
    # Replace these lists with your actual categorical and numerical columns
    categorical_cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'G1', 'G2']  # Replace with actual categorical column names
    numerical_cols = [col for col in tidal_features.columns if col not in categorical_cols]

    # Create a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Apply the transformations
    features_train, features_test, labels_train, labels_test = train_test_split(
        tidal_features, tidal_labels, test_size=0.2, random_state=42)

    features_train_encoded = preprocessor.fit_transform(features_train)
    features_test_encoded = preprocessor.transform(features_test)

    # Adjusted input shape for the model
    input_shape = features_train_encoded.shape[1]

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(20, activation='softmax')  # Adjust the number of units in the output layer as needed
    ])

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(features_train_encoded, labels_train, epochs=100, batch_size=32, validation_split=0.2)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(features_test_encoded, labels_test)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
