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


def run_rbf_comparison():
    def get_Y_pred_correct_inception(model):
        Y_test = model.data_sets.test.labels
        if np.min(Y_test) < -0.5:
            Y_test = (np.copy(Y_test) + 1) / 2        
        Y_pred = model.sess.run(model.preds, feed_dict=model.all_test_feed_dict)
        Y_pred_correct = np.zeros([len(Y_test)])
        for idx, label in enumerate(Y_test):
            Y_pred_correct[idx] = Y_pred[idx, int(label)]
        return Y_pred_correct


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

    tf.reset_default_graph()

    # X_train = image_data_sets.train.x
    # Y_train = image_data_sets.train.labels * 2 - 1
    train = DataSet(L_train, Y_train)
    test = DataSet(L_test, Y_test)

    data_sets = base.Datasets(train=train, validation=None, test=test)
    input_dim = data_sets.train.x.shape[1]

    # Train with hinge
    rbf_model = SmoothHinge(
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
        
    rbf_model.train()
    hinge_W = rbf_model.sess.run(rbf_model.params)[0]

    # Then load weights into smoothed version
    tf.reset_default_graph()
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
    abs_weights = [abs(w) for w in hinge_W]

    x_test = [X_test[i] for i in test_idxs]
    y_test = [Y_test[i] for i in test_idxs]

    distances, flipped_idxs = {}, {}
    for test_idx in test_idxs:
        x_test = X_test[test_idx, :]
        y_test = Y_test[test_idx]
        distances[test_idx] = dataset.find_distances(x_test, X_train)
        flipped_idxs[test_idx] = Y_train != y_test

    rbf_margins_test = rbf_model.sess.run(rbf_model.margin, feed_dict=rbf_model.all_test_feed_dict)
    rbf_margins_train = rbf_model.sess.run(rbf_model.margin, feed_dict=rbf_model.all_train_feed_dict)

    influences = {}
    correlation_list, margin_list = [], []
    for i, test_idx in enumerate(test_idxs):
        rbf_predicted_loss_diffs = rbf_model.get_influence_on_test_loss(
            [test_idx],
            np.arange(len(rbf_model.data_sets.train.labels)),
            force_refresh=True)
        influences[test_idx] = rbf_predicted_loss_diffs
        correlation_list.append(np.corrcoef(abs_weights, rbf_predicted_loss_diffs)[0, 1])
        margin_list.append(abs(rbf_margins_test[test_idx]))

    result = {
        'test_idxs': test_idxs,
        'distances': distances,
        'flipped_idxs': flipped_idxs,
        'rbf_margins_test': rbf_margins_test,
        'rbf_margins_train': rbf_margins_train,
        'influences': influences,
        'hinge_W': hinge_W
    }

    pickle.dump((result, correlation_list, margin_list), open('output/rbf_results.p', 'wb'))