import tensorflow.compat.v1 as tf
import pickle

image_path = './123.jpeg'
image_data = tf.gfile.FastGFile(image_path, 'rb').read()
label_lines = [line.rstrip() for line in tf.gfile.GFile("nn_files/retrained_labels.txt")]

with tf.gfile.FastGFile("nn_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    results = []
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        result = (human_string, score)
        results.append(result)

with open('predictions.pickle', 'wb') as f:
    pickle.dump(results, f)
