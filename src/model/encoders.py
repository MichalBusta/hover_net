'''
Created on Aug 12, 2020

@author: Michal.Busta at gmail.com
'''

import tensorflow as tf
from tensorpack import *

from tensorpack.models import BatchNorm, BNReLU, Conv2D, MaxPooling, FixedUnPooling, AvgPooling, LinearWrap

def inception(name, x, nr1x1, nr3x3r, nr3x3, nr233r, nr233, nrpool, pooltype, freeze):
  
  stride = 2 if nr1x1 == 0 else 1
  with tf.variable_scope(name):
    outs = []
    if nr1x1 != 0:
      outs.append(Conv2D('conv1x1', x, nr1x1, 1))
    x2 = Conv2D('conv3x3r', x, nr3x3r, 1)
    outs.append(Conv2D('conv3x3', x2, nr3x3, 3, strides=stride))

    x3 = Conv2D('conv233r', x, nr233r, 1)
    x3 = Conv2D('conv233a', x3, nr233, 3)
    outs.append(Conv2D('conv233b', x3, nr233, 3, strides=stride))

    if pooltype == 'max':
      x4 = MaxPooling('mpool', x, 3, stride, padding='SAME')
    else:
      assert pooltype == 'avg'
      x4 = AvgPooling('apool', x, 3, stride, padding='SAME')
    if nrpool != 0:  # pool + passthrough if nrpool == 0
      x4 = Conv2D('poolproj', x4, nrpool, 1)
    outs.append(x4)
    l =  tf.concat(outs, 1, name='concat')
    l = tf.stop_gradient(l) if freeze else l
    return l
    
def inception_encoder(i, freeze):
  
  with argscope(Conv2D, activation=BNReLU, use_bias=False):
    d1 = (LinearWrap(i)
         .Conv2D('conv0', 64, 7, strides=1, padding='valid')
         .Conv2D('conv1', 64, 1)
         .Conv2D('conv2', 192, 3)())
    # 28
    l = inception('incep3a', d1, 64, 64, 64, 64, 96, 32, 'avg', freeze=freeze)
    l = inception('incep3b', l, 64, 64, 96, 64, 96, 64, 'avg', freeze=freeze)
    d2 = inception('incep3c', l, 0, 128, 160, 64, 96, 0, 'max', freeze=freeze)
    d2 = tf.stop_gradient(d2) if freeze else d2
    
    # 14
    l = inception('incep4a', d2, 224, 64, 96, 96, 128, 128, 'avg', freeze=freeze)
    l = inception('incep4b', l, 192, 96, 128, 96, 128, 128, 'avg', freeze=freeze)
    l = inception('incep4c', l, 160, 128, 160, 128, 160, 128, 'avg', freeze=freeze)
    l = inception('incep4d', l, 96, 128, 192, 160, 192, 128, 'avg', freeze=freeze)
    d3 = inception('incep4e', l, 0, 128, 192, 192, 256, 0, 'max', freeze=freeze)
    d3 = tf.stop_gradient(d3) if freeze else d3
    
    # 7
    l = inception('incep5a', d3, 352, 192, 320, 160, 224, 128, 'avg', freeze=freeze)
    d4 = inception('incep5b', l, 352, 192, 320, 192, 224, 128, 'max', freeze=freeze)
    d4 = AvgPooling('apool-final', d4, 3, 2, padding='SAME')
    d3= Conv2D('conv-d3-stub',  d3, 1024, 1, strides=1, activation=tf.identity)
    
    return [d1, d2, d3, d4]
    
    