from flask import Flask, render_template, request
import tensorflow.compat.v1 as tf
import pickle
import base64

app = Flask(__name__)

# Memuat hasil prediksi dari file pickle
with open('predictions.pickle', 'rb') as f:
    predictions = pickle.load(f)

# Membaca label dari file teks retrained_labels.txt
label_lines = []
with tf.gfile.GFile('nn_files/retrained_labels.txt', 'r') as f:
    label_lines = [line.rstrip() for line in f]

# Membuat grafik TensorFlow dan sesi
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with tf.gfile.FastGFile('nn_files/retrained_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():

    # Membaca file gambar yang dikirimkan melalui POST request
    image_file = request.files['image']
    image_data = image_file.read()

    # Mendapatkan referensi ke tensor output softmax dari grafik TensorFlow
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    # Menjalankan sesi TensorFlow untuk menghasilkan prediksi klasifikasi
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

    # Mengurutkan probabilitas prediksi dan mendapatkan label dan skor prediksi teratas
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    top_result = top_k[0]
    human_string = label_lines[top_result]
    score = predictions[0][top_result]
    
    # Mengubah skor menjadi persentase dengan 2 desimal
    percentage = "{:.2%}".format(score)

    result = (human_string, percentage)

    # Mengubah data gambar menjadi format base64 untuk ditampilkan di halaman HTML
    image_data_base64 = base64.b64encode(image_data).decode('utf-8')

    # Mengirimkan hasil prediksi dan data gambar ke halaman classify.html
    return render_template('classify.html', result=result, image_data=image_data_base64)


if __name__ == '__main__':
    app.run()
