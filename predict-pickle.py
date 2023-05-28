import tensorflow.compat.v1 as tf
import pickle

# Path gambar yang akan diklasifikasikan
image_path = './123.jpeg'

# Membaca data gambar dari file
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Membaca file teks retrained_labels.txt dan menyimpan setiap baris dalam daftar label_lines
label_lines = [line.rstrip() for line in tf.gfile.GFile("nn_files/retrained_labels.txt")]

# Membuka file biner retrained_graph.pb
with tf.gfile.FastGFile("nn_files/retrained_graph.pb", 'rb') as f:
    # Membuat objek GraphDef
    graph_def = tf.GraphDef()
    # Mem-parsing isi file ke dalam objek graph_def
    graph_def.ParseFromString(f.read())
    # Mengimpor definisi grafik yang disimpan dalam graph_def
    _ = tf.import_graph_def(graph_def, name='')

# Membuat sesi TensorFlow
with tf.Session() as sess:
    # Mendapatkan referensi ke tensor output softmax dari grafik yang diimpor
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    # Menjalankan sesi TensorFlow untuk menghasilkan prediksi klasifikasi
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

    # Mengurutkan probabilitas prediksi dan menyimpan indeks top k prediksi teratas
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    # Membuat daftar kosong untuk menyimpan hasil prediksi
    results = []
    for node_id in top_k:
        # Mendapatkan label dan skor prediksi terkait
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        result = (human_string, score)
        results.append(result)

# Menyimpan hasil prediksi dalam file pickle
with open('predictions.pickle', 'wb') as f:
    pickle.dump(results, f)
