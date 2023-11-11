import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras_tuner.tuners import RandomSearch
from sklearn.compose import ColumnTransformer
import numpy as np


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(features_train_encoded.shape[1],)))

    # Tune the number of units in the first dense layer
    # model.add(keras.layers.Dense(units=hp.Int('units', min_value=64, max_value=512, step=16), activation='relu'))
    model.add(keras.layers.Dense(units=hp.Int('layer0Nodes', min_value=64, max_value=128, step=8), activation='relu'))

    model.add(keras.layers.Dense(units=hp.Int('layer1Nodes', min_value=128, max_value=512, step=16), activation='relu'))

    model.add(keras.layers.Dense(units=hp.Int('layer2Nodes', min_value=32, max_value=64, step=4), activation='relu'))

    model.add(keras.layers.Dropout(hp.Choice('dropout', values=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])))

    # Add more layers as needed
    model.add(keras.layers.Dense(20, activation='softmax'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    tidal_train = pd.read_csv("./DATA/student-performance.csv")

    # Assume 'G3' is the label and rest are features
    tidal_features = tidal_train.drop(['G3'], axis=1)
    tidal_labels = tidal_train['G3']

    # Identify categorical and numerical columns
    # Replace these lists with your actual categorical and numerical columns
    categorical_cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                        'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'G1',
                        'G2']  # Replace with actual categorical column names
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

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=100,  # Number of hyperparameter combinations to try
        directory='my_tuning_dir',  # Directory to save the tuning results
        project_name='my_tuning_project'
    )
    tuner.search(features_train_encoded, labels_train, epochs=10, validation_split=0.2)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    final_model = tuner.hypermodel.build(best_hps)
    final_model.summary()
    final_model.save('best_model.h5')

    # Train the model
    final_model.fit(features_train_encoded, labels_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model on the test set
    test_loss, test_accuracy = final_model.evaluate(features_test_encoded, labels_test)
    print(f'Test Loss: {test_loss:.2f}')
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    # Predict the labels for the test set
    predicted_labels = final_model.predict(features_test_encoded)

    predicted_classes = np.argmax(predicted_labels, axis=1)
    print("Predicted Classes:")
    print(predicted_classes)
    print("Actual Classes:")
    print(np.array(labels_test))