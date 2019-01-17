# coding: utf-8
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

import model
import utils


flags = tf.app.flags

FLAGS = flags.FLAGS


flags.DEFINE_string('checkpoint_path', None, 'Checkpoint path')

flags.DEFINE_string('export_path', "frozen_inference_graph.pb",
                    'Path to output Tensorflow frozen graph')

flags.DEFINE_string('saved_model_version', None,
                    'The version of exported savedModel')


# Input name of the exported model.
_INPUT_NAME = 'InputImage'

# Output name of the exported model.
_OUTPUT_NAME = 'StyledImage'


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default(), tf.Session() as sess:
        # Building graph.
        inputTensor = None
        if FLAGS.saved_model_version is not None:
            inputTensor = tf.placeholder(shape=[], dtype=tf.string, name=_INPUT_NAME)
            decoded_bytes = tf.io.decode_base64(inputTensor)
            image_data = tf.image.decode_image(decoded_bytes, channels=3)
        else:
            image_data = tf.placeholder(tf.uint8, shape=(None, None, 3), name=_INPUT_NAME)
            inputTensor = image_data

        image_shape = tf.shape(image_data)

        # Preprocessing image
        processed_image = utils.mean_image_subtraction(image_data,
                                                       [123.68, 116.779, 103.939])

        # Add batch dimension
        batched_image = tf.expand_dims(processed_image, 0)
        generated_image = model.net(batched_image, training=False)
        casted_image = tf.cast(generated_image, tf.uint8)

        # Remove batch dimension
        squeezed_image = tf.squeeze(casted_image, [0])

        cropped_image = tf.slice(squeezed_image, [0, 0, 0], image_shape)


        # Restore model variables.
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.checkpoint_path)


        if FLAGS.saved_model_version is not None:
            builder = tf.saved_model.builder.SavedModelBuilder("style/Servo/" + FLAGS.saved_model_version)

            # Creates the TensorInfo protobuf objects that encapsulates the input/output tensors
            tensor_info_input = tf.saved_model.utils.build_tensor_info(inputTensor)

            # output tensor info
            styled_image = tf.image.encode_png(cropped_image, name=_OUTPUT_NAME)
            tensor_info_output = tf.saved_model.utils.build_tensor_info(styled_image)

            sigs = {}
            sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs = {"input" : tensor_info_input},
                    outputs = {"output_bytes" : tensor_info_output},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )

            builder.add_meta_graph_and_variables(sess,
                                                 [tag_constants.SERVING],
                                                 signature_def_map=sigs,
                                                 clear_devices=True)

            builder.save()

        else:
            styled_image = tf.identity(cropped_image, name=_OUTPUT_NAME)

            output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                            sess.graph_def,
                                                                            output_node_names=[_OUTPUT_NAME])

            with tf.gfile.FastGFile(FLAGS.export_path, mode='wb') as f:
                f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    flags.mark_flag_as_required('checkpoint_path')
    tf.app.run()
