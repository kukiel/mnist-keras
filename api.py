import numpy as np

from flask import Flask, request
from flask_cors import CORS
from flask_jsonpify import jsonify
from flask_restful import Resource, Api
from json import dumps
from keras.models import load_model
from keras.utils import to_categorical
from scipy.misc import imresize
import tensorflow as tf
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

app = Flask(__name__)
CORS(app)
api = Api(app)

model = load_model('mnist-model.h5')
model_convnet = load_model('mnist-model-conv.h5')
reshape_dict = {'dense': (1, 784), 'conv': (1, 28, 28, 1)}


class Model(Resource):
    def post(self):
        scale = request.json.get('scale')
        layers_type = request.json.get('type', 'conv')
        number_to_predict = np.asarray([request.json.get('data')])

        image = number_to_predict.reshape((28 * scale, 28 * scale))
        resized_image = imresize(image, (28, 28), interp='nearest').reshape(reshape_dict.get(layers_type))
        normalized_image = np.divide(resized_image, 255)

        result = None

        if layers_type == 'dense':
            result = model.predict(normalized_image)
        elif layers_type == 'conv':
            result = model_convnet.predict(normalized_image)

        return jsonify({'prediction': result.tolist()})


api.add_resource(Model, '/predict')

if __name__ == '__main__':
    app.run(port=5000, debug=False)
