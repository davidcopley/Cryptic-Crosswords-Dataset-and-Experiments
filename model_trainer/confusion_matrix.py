from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os

my_path = os.path.abspath(os.path.dirname(__file__))


cc_types = ['is_anagram', 'is_homophone', 'is_double', 'is_cryptic', 'is_contain', 'is_reverse', 'is_alternate',
            'is_init', 'is_delete', 'is_&lit', 'is_hidden', 'is_spoonerism', 'is_palindrome']


def generate_confusion_matrix(actual_Y,pred_Y):
    return confusion_matrix(actual_Y,pred_Y)


def plot_confusion_matrix( cm, classes=cc_types, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_confusion_matrix_png(cnf_matrix,title):
    # Compute confusion matrix
    np.set_printoptions(precision=2)
    class_names = cc_types
    # Plot non-normalized confusion matrix
    plt.figure(figsize=(15, 15))
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='{} - Confusion matrix - Not Normalization'.format(title))

    plt.savefig(os.path.join(my_path,"./confusion_matrices/{}_unnormalized.png".format(title)))
    # Plot normalized confusion matrix
    plt.figure(figsize=(15, 15))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='{} - Confusion matrix - Normalization'.format(title))

    plt.savefig(os.path.join(my_path, "./confusion_matrices/{}_normalized.png".format(title)))
