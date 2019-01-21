# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from PIL import Image
from preprocessing import preprocessing_factory
import reader
import model
import time
import os
import numpy as np

tf.app.flags.DEFINE_string("model_file", "models.ckpt", "")
tf.app.flags.DEFINE_string("image_file", "a.jpg", "")

FLAGS = tf.app.flags.FLAGS


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

        # Make sure 'generated' directory exists.
        generated_file = 'generated/res.jpg'
        if os.path.exists('generated') is False:
            os.makedirs('generated')

        # Generate and write image data to file.
        start_time = time.time()
        output = sess.run('StyledImage:0',
                          feed_dict={'InputImage:0' : image})

        end_time = time.time()

        print(height, width, output.shape, output.dtype)
        Image.fromarray(output.reshape((height, width, 3)).astype(np.uint8)).save(generated_file)

        tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

        tf.logging.info('Done. Please check %s.' % generated_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
