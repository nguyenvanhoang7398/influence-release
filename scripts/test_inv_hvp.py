from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import pickle
import math
import copy
import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model
import sklearn.preprocessing as preprocessing
import scipy
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse

from scripts.load_animals import load_animals

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

from sklearn.metrics.pairwise import rbf_kernel

from influence.inceptionModel import BinaryInceptionModel
from influence.smooth_hinge import SmoothHinge
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
import influence.dataset as dataset
from influence.dataset import DataSet
from influence.dataset_poisoning import generate_inception_features

def test_inv_hvp():
    num_classes = 2
    num_train_ex_per_class = 900
    num_test_ex_per_class = 300

    dataset_name = 'dogfish_%s_%s' % (num_train_ex_per_class, num_test_ex_per_class)
    image_data_sets = load_animals(
        num_train_ex_per_class=num_train_ex_per_class, 
        num_test_ex_per_class=num_test_ex_per_class,
        classes=['dog', 'fish'])

    ### Generate kernelized feature vectors
    X_train = image_data_sets.train.x
    X_test = image_data_sets.test.x

    Y_train = np.copy(image_data_sets.train.labels) * 2 - 1
    Y_test = np.copy(image_data_sets.test.labels) * 2 - 1

    # X_train, X_test = X_train[:10], X_test[-2:]
    # Y_train, Y_test = Y_train[:10], Y_test[-2:]

    num_train = X_train.shape[0]
    num_test = X_test.shape[0]

    X_stacked = np.vstack((X_train, X_test))

    gamma = 0.05
    weight_decay = 0.0001

    K = rbf_kernel(X_stacked, gamma = gamma / num_train)

    L = slin.cholesky(K, lower=True)
    L_train = L[:num_train, :num_train]
    L_test = L[num_train:, :num_train]

    ### Compare top 5 influential examples from each network

    test_idxs = range(num_test)

    ## RBF

    input_channels = 1
    weight_decay = 0.001
    batch_size = num_train
    initial_learning_rate = 0.001 
    keep_probs = None
    max_lbfgs_iter = 1000
    use_bias = False
    decay_epochs = [1000, 10000]

    # X_train = image_data_sets.train.x
    # Y_train = image_data_sets.train.labels * 2 - 1
    train = DataSet(L_train, Y_train)
    test = DataSet(L_test, Y_test)

    data_sets = base.Datasets(train=train, validation=None, test=test)
    input_dim = data_sets.train.x.shape[1]

    hinge_graph = tf.Graph()
    smooth_hinge_graph = tf.Graph()

    # Train with hinge
    with hinge_graph.as_default():
        hinge_rbf_model = SmoothHinge(
            temp=0,
            use_bias=use_bias,
            input_dim=input_dim,
            weight_decay=weight_decay,
            num_classes=num_classes,
            batch_size=batch_size,
            data_sets=data_sets,
            initial_learning_rate=initial_learning_rate,
            keep_probs=keep_probs,
            decay_epochs=decay_epochs,
            mini_batch=False,
            train_dir='output',
            log_dir='log',
            model_name='dogfish_rbf_hinge_t-0')
            
        hinge_rbf_model.train()
        hinge_W = hinge_rbf_model.sess.run(hinge_rbf_model.params)[0]

    # Then load weights into smoothed version
    with smooth_hinge_graph.as_default():
        rbf_model = SmoothHinge(
            temp=0.001,
            use_bias=use_bias,
            input_dim=input_dim,
            weight_decay=weight_decay,
            num_classes=num_classes,
            batch_size=batch_size,
            data_sets=data_sets,
            initial_learning_rate=initial_learning_rate,
            keep_probs=keep_probs,
            decay_epochs=decay_epochs,
            mini_batch=False,
            train_dir='output',
            log_dir='log',
            model_name='dogfish_rbf_hinge_t-0.001')

        params_feed_dict = {}
        params_feed_dict[rbf_model.W_placeholder] = hinge_W
        rbf_model.sess.run(rbf_model.set_params_op, feed_dict=params_feed_dict)

    test_idx = 462

    hinge_inf_grads= rbf_model.get_grad_of_influence_wrt_input(
            np.arange(len(rbf_model.data_sets.train.labels)),
            [test_idx],
            force_refresh=True)

    smooth_hinge_inf_grads = rbf_model.get_grad_of_influence_wrt_input(
            np.arange(len(rbf_model.data_sets.train.labels)),
            [test_idx],
            force_refresh=True)
    np.testing.assert_array_equal(hinge_inf_grads, smooth_hinge_inf_grads)