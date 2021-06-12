"""
Some functions has been copied from https://github.com/VisionLearningGroup/DANCE and https://github.com/bhanML/Co-teaching
"""
from models.basenet import *
import torch
import numpy as np
from numpy.testing import assert_array_almost_equal
import sklearn.metrics as sk

def get_model_mme(net, num_class=13, unit_size=2048, temp=0.05):
    model_g = ResBase(net, unit_size=unit_size)
    model_c = ResClassifier_MME(num_classes=num_class, input_size=unit_size, temp=temp)
    return model_g, model_c

def get_model_mme_2head(net, num_class=13, unit_size=2048, temp=0.05):
    model_g = ResBase(net, unit_size=unit_size)
    model_c1 = ResClassifier_MME(num_classes=num_class, input_size=unit_size, temp=temp)
    model_c2 = ResClassifier_MME(num_classes=num_class, input_size=unit_size, temp=temp)
    return model_g, model_c1, model_c2

def save_model(model_g, model_c, save_path):
    save_dic = {
        'g_state_dict': model_g.state_dict(),
        'c_state_dict': model_c.state_dict(),
    }
    torch.save(save_dic, save_path)


def load_model(model_g, model_c, load_path):
    checkpoint = torch.load(load_path)
    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_c.load_state_dict(checkpoint['c_state_dict'])
    return model_g, model_c

def extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None):
 
    if not true_labels:
        true_labels = sorted(list(set(list(y_true))))
    true_label_to_id = {x : i for (i, x) in enumerate(true_labels)}
    if not pred_labels:
        pred_labels = true_labels
    pred_label_to_id = {x : i for (i, x) in enumerate(pred_labels)}
    confusion_matrix = np.zeros([len(true_labels), len(pred_labels)])
    for (true, pred) in zip(y_true, y_pred):
        confusion_matrix[true_label_to_id[true]][pred_label_to_id[pred]] += 1.0
    return confusion_matrix

def cal_metrics(measure_in, measure_out, in_out_lbl):
    measure_all = np.concatenate([measure_in, measure_out])
    auroc = sk.roc_auc_score(in_out_lbl, measure_all) # AUROC
    aupr_in = sk.average_precision_score(in_out_lbl, measure_all) # aupr in
    aupr_out = sk.average_precision_score((in_out_lbl - 1) * -1, measure_all * -1) # aupr out
    in_mea_mean = np.mean(measure_in) # Mean of out-dist measure
    out_mea_mean = np.mean(measure_out) # Mean of out-dist measure
    return auroc, aupr_in, aupr_out, in_mea_mean, out_mea_mean

def create_ood_lbl(measure_in, measure_out, id_is_larger=True):
    if id_is_larger:
        all_mea = [np.ones_like(measure_in), np.zeros_like(measure_out)]
    else:
        all_mea = [np.zeros_like(measure_in), np.ones_like(measure_out)]
    return np.concatenate(all_mea, axis=0)


# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    print (np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    print (m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print (P)

    return y_train, actual_noise

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print (P)

    return y_train, actual_noise

def noisify(nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate