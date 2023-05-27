from flask import Flask, render_template, request
import tensorflow.compat.v1 as tf
import pickle
import base64

app = Flask(__name__)

# Load the predictions.pickle file
with open('predictions.pickle', 'rb') as f:
    predictions = pickle.load(f)

# Load label lines
label_lines = []
with tf.gfile.GFile('nn_files/retrained_labels.txt', 'r') as f:
    label_lines = [line.rstrip() for line in f]

# Create a TensorFlow session
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    # Load the TensorFlow graph
    with tf.gfile.FastGFile('nn_files/retrained_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Get the uploaded image file from the request
    image_file = request.files['image']

    # Read the image data
    image_data = image_file.read()

    # Perform image classification
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

    # Process the predictions
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    top_result = top_k[0]
    human_string = label_lines[top_result]
    score = predictions[0][top_result]
    result = (human_string, score)

    # Convert the image data to base64 for displaying in HTML
    image_data_base64 = base64.b64encode(image_data).decode('utf-8')

    # Render the classify.html template with the classification result and image data
    return render_template('classify.html', result=result, image_data=image_data_base64)

if __name__ == '__main__':
    app.run()
