import IPython
import numpy as np

import sklearn.linear_model as linear_model
from scripts.load_animals import load_animals, load_dogfish_with_koda, load_dogfish_with_orig_and_koda
import copy

import os
from shutil import copyfile

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

from sklearn.metrics.pairwise import rbf_kernel
import scipy
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse

from influence.smooth_hinge import SmoothHinge
import influence.experiments
from influence.dataset import DataSet
from influence.dataset_poisoning import iterative_attack, select_examples_to_attack, get_projection_to_box_around_orig_point

def iterative_attack(raw_train, hinge, smooth_hinge, hinge_graph, smooth_hinge_graph, project_fn, test_indices, test_description=None, 
    indices_to_poison=None,
    num_iter=10,
    step_size=1,
    save_iter=1,
    loss_type='normal_loss',
    early_stop=None):
    # If early_stop is set and it stops early, returns True
    # Otherwise, returns False

    if test_description is None:
        test_description = test_indices

    if early_stop is not None:
        assert len(test_indices) == 1, 'Early stopping only supported for attacks on a single test index.'

    if len(indices_to_poison) == 1:
        train_idx_str = indices_to_poison
    else:
        train_idx_str = len(indices_to_poison)

    hinge_model_name = hinge.model_name
    smooth_hinge_model_name = smooth_hinge.model_name

    print('Test idx: %s' % test_indices)
    print('Indices to poison: %s' % indices_to_poison)

    smooth_hinge.update_train_x_y(
        smooth_hinge.data_sets.train.x[indices_to_poison, :],
        smooth_hinge.data_sets.train.labels[indices_to_poison])
    eff_indices_to_poison = np.arange(len(indices_to_poison))
    labels_subset = smooth_hinge.data_sets.train.labels[eff_indices_to_poison]

    for attack_iter in range(num_iter):
        print('*** Iter: %s' % attack_iter)
        
        print('Calculating grad...')

        with smooth_hinge_graph.as_default():
            grad_influence_wrt_input_val_subset = smooth_hinge.get_grad_of_influence_wrt_input(
                eff_indices_to_poison, 
                test_indices, 
                force_refresh=False,
                test_description=test_description,
                loss_type=loss_type)
            poisoned_X_train_subset = poison_with_influence_proj_gradient_step(
                raw_train.x, 
                indices_to_poison,
                grad_influence_wrt_input_val_subset,
                step_size,
                project_fn)
        
        with smooth_hinge_graph.as_default():
            snapshot_raw_X_train = copy.deepcopy(raw_train.x)
            snapshot_raw_X_train[indices_to_poison] = poisoned_raw_X_train_subset
            poisoned_kernelized_X_train = kernelize(snapshot_raw_X_train)
            poisoned_kernelized_X_train_subset = poisoned_kernelized_X_train[indices_to_poison]
            smooth_hinge_graph.update_train_x(poisoned_kernelized_X_train_subset)

        with hinge_graph.as_default():
            hinge_graph.update_train_x(poisoned_kernelized_X_train)

        # Retrain model
        print('Training...')
        with hinge_graph.as_default():
            hinge.train()
            hinge_W = hinge.sess.run(hinge_rbf_model.params)[0]
        with smooth_hinge_graph.as_default():
            params_feed_dict = {}
            params_feed_dict[smooth_hinge.W_placeholder] = hinge_W
            smooth_hinge.sess.run(smooth_hinge.set_params_op, feed_dict=params_feed_dict)

        # Print out attack effectiveness if it's not too expensive
        test_pred = None
        if len(test_indices) < 100:
            with smooth_hinge_graph.as_default():
                test_pred = smooth_hinge.sess.run(smooth_hinge.preds, feed_dict=smooth_hinge.fill_feed_dict_with_some_ex(
                    smooth_hinge.data_sets.test,
                    test_indices))
                print('Test pred (smooth hinge): %s' % test_pred)

            if ((early_stop is not None) and (len(test_indices) == 1)):
                if test_pred[0, int(smooth_hinge.data_sets.test.labels[test_indices])] < early_stop:
                    print('Successfully attacked. Saving and breaking...')
                    np.savez('output/%s_attack_%s_testidx-%s_trainidx-%s_stepsize-%s_proj_final' % (smooth_hinge.model_name, loss_type, test_description, train_idx_str, step_size), 
                        poisoned_X_train_image=poisoned_X_train_subset, 
                        poisoned_X_train_inception_features=inception_X_train_subset,
                        Y_train=labels_subset,
                        indices_to_poison=indices_to_poison,
                        attack_iter=attack_iter + 1,
                        test_pred=test_pred,
                        step_size=step_size)            
                    return True

        if (attack_iter+1) % save_iter == 0:
            np.savez('output/%s_attack_%s_testidx-%s_trainidx-%s_stepsize-%s_proj_iter-%s' % (full_model.model_name, loss_type, test_description, train_idx_str, step_size, attack_iter+1), 
                poisoned_X_train_subset=poisoned_X_train_subset, 
                poisoned_kernelized_X_train_subset=poisoned_kernelized_X_train_subset,
                Y_train=labels_subset,
                indices_to_poison=indices_to_poison,
                attack_iter=attack_iter + 1,
                test_pred=test_pred,
                step_size=step_size)

    return False

def kernelize(X_train, X_test=None):
    num_train = X_train.shape[0]

    if X_test is not None:    
        X_stacked = np.vstack((X_train, X_test))
    else:
        X_stacked = X_train
    gamma = 0.05

    K = rbf_kernel(X_stacked, gamma = gamma / num_train)

    L = slin.cholesky(K, lower=True)
    L_train = L[:num_train, :num_train]
    L_test = L[num_train:, :num_train]
    return L_train, L_test

def poison_with_influence_proj_gradient_step(raw_X_train, indices_to_poison, grad_influence_wrt_input_val_subset, step_size, project_fn):
    print("raw_X_train", np.shape(raw_X_train))
    print("indices_to_poison", np.shape(indices_to_poison))
    poisoned_X_train_subset = project_fn(
        raw_X_train[indices_to_poison, :] - step_size * np.sign(grad_influence_wrt_input_val_subset) * 2.0 / 255.0)

    print('-- max: %s, mean: %s, min: %s' % (
        np.max(grad_influence_wrt_input_val_subset),
        np.mean(grad_influence_wrt_input_val_subset),
        np.min(grad_influence_wrt_input_val_subset)))

    return poisoned_X_train_subset

def run_adversarial_atk_svm():
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

    L_train, L_test = kernelize(X_train, X_test)

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
    raw_train = DataSet(X_train, Y_train)
    raw_test = DataSet(X_test, Y_test)

    data_sets = base.Datasets(train=train, validation=None, test=test)
    input_dim = data_sets.train.x.shape[1]

    hinge_graph = tf.Graph()
    smooth_hinge_graph = tf.Graph()

    # Train with hinge
    with hinge_graph.as_default():
        hinge_rbf_name = '%s_hinge_rbf' % (dataset_name)
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
        rbf_name = '%s_rbf' % (dataset_name)
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

    max_num_to_poison = 10

    step_size = 0.005
    test_indices = np.arange(num_test)
    test_description = 'all_%s' % dataset_name
    
    with smooth_hinge_graph.as_default():
        grad_influence_wrt_input_val = rbf_model.get_grad_of_influence_wrt_input(
                np.arange(num_train), 
                test_indices, 
                force_refresh=False,
                test_description=test_description,
                loss_type='normal_loss')
        indices_to_poison = select_examples_to_attack(
                rbf_model, 
                max_num_to_poison, 
                grad_influence_wrt_input_val,
                step_size=step_size)
    print('****')
    print('Indices to poison: %s' % indices_to_poison)
    print('****')

    idx_to_poison = indices_to_poison[0]
    orig_X_train_subset = np.copy(rbf_model.data_sets.train.x[idx_to_poison, :])

    project_fn = get_projection_to_box_around_orig_point(orig_X_train_subset, box_radius_in_pixels=0.5)
    # project_fn = lambda x : x
    iterative_attack(raw_train, hinge_rbf_model, rbf_model, hinge_graph, smooth_hinge_graph, project_fn, test_indices, test_description=test_description, 
        indices_to_poison=[idx_to_poison],
        num_iter=2000,
        step_size=step_size,
        save_iter=500,
        loss_type='normal_loss')