from flask import Flask, render_template, request, redirect, url_for
import os
import glob
import cv2
from werkzeug.utils import secure_filename

import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

def training():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train),(X_test, y_test) = mnist.load_data()

    # Normalizing the data (making length = 1)
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # Create a neural network model
    # Add one flattened input layer for the pixels
    # Add two dense hidden layers
    # Add one dense output layer for the 10 digits
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # Compiling and optimizing model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    model.fit(X_train, y_train, epochs=3)

    # Evaluating the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    # Save model
    model.save('num.model')
    print('New model saved!')

@app.route('/')
def home():
    if os.path.exists('./num.model'):
        print('Model already exist!')
    else:
        training()

    return render_template('home.html')

@app.route('/result', methods = ['GET','POST'])
def result():
    keras.backend.clear_session() 
    model = tf.keras.models.load_model('num.model') 

    folder = glob.glob(r'D:\Documents\ml-number-project\static\uploads\*')
    for items in folder:
        os.remove(items)
     
    if request.method == 'POST':
        if request.files['file']:

            image = request.files['file']

            img = secure_filename(image.filename)      
            path = os.path.join(r'D:\Documents\ml-number-project\static\uploads', img)
            
            image.save(path)  
        
            imgcv = cv2.imread(path)[:,:,0] 
            imgcv = cv2.resize(imgcv, (28,28))   
            imgcv = np.invert(np.array([imgcv]))
            prediction = model.predict(imgcv)
            
            result = "The number is probably a {}".format(np.argmax(prediction))        
            return render_template('result.html', result = result, img = img)

        else:

            result = 'Upload number to analyse'
            return render_template('result.html', result = result)

    else:
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(threaded=False)

