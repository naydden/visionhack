u'''
HelloWorld example using TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
from __future__ import absolute_import
import tensorflow as tf

# Simple hello world using TensorFlow

# Create a Constant op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
hello = tf.constant(u'Hello, TensorFlow!')

# Start tf session
sess = tf.Session()

# Run the op
print (sess.run(hello))
print u"test"
