import tensorflow as tf
import pandas as pd
import numpy as np

if __name__ == '__main__':

    tidal_train = pd.read_csv("./DATA/DATA.csv")

    tidal_features = tidal_train.copy()
    tidal_labels = tidal_features.pop('GRADE')
    tidal_features.pop('STUDENT ID')

    tidal_features = np.array(tidal_features)

    tidal_features = tidal_features.astype(np.int32)  # Convert to tf.float32
    tidal_labels = tidal_labels.astype(np.int32)  # Convert to tf.int32

    normalize = tf.keras.layers.Normalization()
    normalize.adapt(tidal_features)

    tidal_model = tf.keras.Sequential([
        normalize,
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(1)
    ])

    tidal_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())

    tidal_model.fit(tidal_features, tidal_labels, epochs=10)

    # # Prepare the input data for prediction
    # student_data = data.iloc[:, 1:-2].values  # Extract features for each student
    # course_data = data["COURSE ID"].values  # Extract the course ID
    # input_data = np.column_stack((student_data, course_data))  # Combine features and course ID
    #
    # # Make predictions
    # predictions = model.predict(input_data)
    #
    # # Add the predictions to the DataFrame
    # data["PREDICTED GRADE"] = predictions
    #
    # # Save the DataFrame with predictions to a new CSV file
    # data.to_csv("predictions.csv", index=False)
    # print(tidal_train.head())

    # mnist = tf.keras.datasets.mnist
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(28, 28)),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(10)
    # ])
    # predictions = model(x_train[:1]).numpy()
    # tf.nn.softmax(predictions).numpy()
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss_fn(y_train[:1], predictions).numpy()
    # model.compile(optimizer='adam',
    #               loss=loss_fn,
    #               metrics=['accuracy'])
    # model.fit(x_train, y_train, epochs=5)
    # model.evaluate(x_test, y_test, verbose=2)
