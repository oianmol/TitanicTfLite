import numpy as np
import tflearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tflearn.data_utils import load_csv
from tflearn.datasets import titanic

# Download the Titanic dataset
titanic.download_dataset('titanic_dataset.csv')

# Load CSV file, indicate that the first column represents labels
data, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)


# Preprocessing function
def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
        # Converting 'sex' field to float (id is 1 after removing labels column)
        data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)


# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore = [1, 6]

# Preprocess data
data = preprocess(data, to_ignore)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42)


def buildModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=6))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=6))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=6, kernel_initializer='uniform', activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=2, kernel_initializer='uniform', activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20, batch_size=16, shuffle=True)
    return model


def evalModel(model):
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss)
    print(val_acc)


def saveModel(model):
    kerasfile = "keras_model.h5"
    tf.keras.models.save_model(model, kerasfile)
    return kerasfile


def convertModel(keras_file):
    # Convert to TensorFlow Lite model.
    converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(keras_file)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)


def predictValues(model):
    # Let's create some data for DiCaprio and Winslet
    dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
    winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
    # Preprocess data
    arr = preprocess([dicaprio, winslet], to_ignore)
    # Predict surviving chances (class 1 results)
    print(arr.shape)
    pred = model.predict(arr)
    print(pred)
    print("DiCaprio Surviving Rate:", pred[0][1])
    print("Winslet Surviving Rate:", pred[1][1])


print(tf.__version__)
model = buildModel()
print("eval model")
evalModel(model)
keras_file = saveModel(model)
print("predict model")
predictValues(model)
convertModel(keras_file)
