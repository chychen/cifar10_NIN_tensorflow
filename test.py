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

import tensorflow as tf

import cifar10
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           # tf.app.flags.DEFINE_string('train_dir',
                           # '../result/with_augmentation_164_epoches/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 391 * 3,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 391,
                            """How often to log results to the console.""")

def train():
    """Train CIFAR-10 for a number of steps."""
    # with tf.Graph().as_default() as g:
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()
    images_eval, labels_eval = cifar10.inputs(eval_data=True)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)
    # Calculate loss.
    loss = cifar10.loss(logits, labels)
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    tf.get_variable_scope().reuse_variables()

    logits_eval = cifar10.inference(images_eval, isTrain=False)
    # with tf.variable_scope('eval', reuse=False):
    top_k_eval = tf.nn.in_top_k(logits_eval, labels_eval, 1)
    loss_eval = cifar10.loss(logits_eval, labels_eval, isTrain=False)

    # precision_eval = tf.placeholder(tf.float32)
    # tf.summary.scalar('prec_eval', precision_eval)
    tf.summary.scalar('loss_eval', loss_eval)

    class _LoggerHook(tf.train.SessionRunHook):
        """Logs loss and runtime."""

        def begin(self):
            self._step = -1
            self._start_time = time.time()
            self._true_count = 0

        def before_run(self, run_context):
            self._step += 1
            # Asks for loss value.
            return tf.train.SessionRunArgs([loss, loss_eval, top_k_eval])

        def after_run(self, run_context, run_values):
            if self._step % FLAGS.log_frequency == 0 or \
                    self._step == (FLAGS.max_steps - 1):
                current_time = time.time()
                duration = current_time - self._start_time
                # print ('run_values.results[0]::', run_values.results[0])
                # print ('run_values.results[1]::', run_values.results[1])
                # print ('run_values.results[2]::', run_values.results[2])
                loss_value = run_values.results[0]
                examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                sec_per_batch = float(duration / FLAGS.log_frequency)
                loss_eval = run_values.results[1]

                format_str = ('%s: step %d, loss = %.3f (%.1f examples/sec; %.3f '
                              'sec/batch), prec_eval = %.3f, loss_eval = %.3f')
                print (format_str % (datetime.now(), self._step, loss_value,
                                     examples_per_sec, sec_per_batch, precision_eval, loss_eval))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
