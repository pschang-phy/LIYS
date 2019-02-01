# coding: utf-8
import tensorflow as tf
from PIL import Image
import time
import os
import numpy as np
import cv2

tf.app.flags.DEFINE_string("model", "frozen_inference_graph.pb",
                           "The inference model graph.")

tf.app.flags.DEFINE_string("img", "a.jpg",
                           "The input image file.")

tf.app.flags.DEFINE_string("out", "res.jpg",
                           "The output image file.")

FLAGS = tf.app.flags.FLAGS


colormap = np.array([[0, 0, 0],
                     [128, 0, 0]], dtype=np.uint8)

def main(_):

    # Get image's height and width.
    height = 0
    width = 0
    with open(FLAGS.image_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if FLAGS.image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default(), tf.Session() as sess:
        with open(os.path.abspath(FLAGS.model_file), 'rb') as file_handle:
            graph_def = tf.GraphDef.FromString(file_handle.read())
            tf.import_graph_def(graph_def, name='')

        resize_ratio = 1.0 * 513 / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        # Generate and write image data to file.
        start_time = time.time()
        output = sess.run('SemanticPredictions:0',
                          feed_dict={'ImageTensor:0' : np.expand_dims(resized_image, axis=0)})
        end_time = time.time()
        
        seg_map = cv2.resize(output[0], (width, height), interpolation=cv2.INTER_NEAREST)
        seg_image = colormap[seg_map]

        overlapping = cv2.addWeighted(image, 1.0, seg_image, 0.7, 0)
        overlapping = cv2.cvtColor(overlapping, cv2.COLOR_RGB2BGR)
        cv2.imwrite(FLAGS.out, overlapping)

        print(height, width, output.shape, output.dtype)

        tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

        tf.logging.info('Done. Please check %s.' % FLAGS.out)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
