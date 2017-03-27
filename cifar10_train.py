# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import os.path
import math
import numpy as np
import tensorflow as tf

import cifar10
from cifar10_input import NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 391 * 164,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 391,
                            """How often to log results to the console.""")


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:

        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        images, labels = cifar10.distorted_inputs()
        images_eval, labels_eval = cifar10.inputs(eval_data=True)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images)
        # Calculate loss.
        loss_train = cifar10.loss(logits, labels)
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = cifar10.train(loss_train, global_step)

        tf.get_variable_scope().reuse_variables()

        logits_eval = cifar10.inference(images_eval, isTrain=False)
        # with tf.variable_scope('eval', reuse=False):
        top_k_eval = tf.nn.in_top_k(logits_eval, labels_eval, 1)
        loss_eval = cifar10.loss(logits_eval, labels_eval, isTrain=False)

        # Add summary
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=g)
        summary = tf.Summary()

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            sess.run(train_op)
            duration = time.time() - start_time

            if step % FLAGS.log_frequency == 0 or (step + 1) == FLAGS.max_steps:
                loss_value, loss_eval_value = sess.run([loss_train, loss_eval])

                assert not np.isnan(
                    loss_value), 'Model diverged with loss = NaN'

                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                # Start the queue runners.
                coord = tf.train.Coordinator()
                try:
                    threads = []
                    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                         start=True))

                    num_iter = int(
                        math.ceil(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / FLAGS.batch_size))
                    true_count = 0  # Counts the number of correct predictions.
                    total_sample_count = num_iter * FLAGS.batch_size
                    i = 0
                    while i < num_iter and not coord.should_stop():
                        predictions = sess.run([top_k_eval])
                        true_count += np.sum(predictions)
                        i += 1

                    # Compute precision @ 1.
                    precision = true_count / total_sample_count

                    summary = tf.Summary()
                    summary.ParseFromString(sess.run(summary_op))
                    summary.value.add(
                        tag='loss_eval', simple_value=loss_eval_value)
                    summary.value.add(tag='Precision @ 1',
                                      simple_value=precision)
                    summary_writer.add_summary(summary, step)
                except Exception as e:  # pylint: disable=broad-except
                    coord.request_stop(e)

                coord.request_stop()
                coord.join(threads, stop_grace_period_secs=10)

                format_str = ('%s: step %d, loss = %.3f (%.1f examples/sec; %.3f '
                              'sec/batch), prec_eval = %.3f, loss_eval = %.3f')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch, precision, loss_eval_value))

                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision @ 1',
                                  simple_value=precision)
                summary_writer.add_summary(summary, step)

                # Save the model checkpoint periodically.
                if (step + 1) % 3910 == 0 or (step + 1) == FLAGS.max_steps:
                    model_name = 'model.ckpt'
                    checkpoint_path = os.path.join(
                        FLAGS.train_dir, model_name)
                    saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
