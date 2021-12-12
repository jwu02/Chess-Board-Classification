"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List

import numpy as np
from numpy.lib.function_base import append
from scipy.stats import multivariate_normal
from scipy.spatial import distance_matrix
import scipy.linalg
import scipy.fft

N_DIMENSIONS = 10

# uppercase white pieces
# lowercase black pieces
# . represents an empty square
PIECES = [".","K","Q","B","N","R","P","k","q","b","n","r","p"]

def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    model = process_training_data(train, train_labels)

    pcatest_data = np.dot((test - np.mean(test)), np.array(model["A"]))

    """
    # Gaussian distributions
    # clean: square = 97.1%, board = 97.1%
    # noisy: square = 93.6%, board = 93.6%
    distributions = [multivariate_normal(mean=model["means"][i], cov=model["covariances"][i]) for i in range(len(PIECES))]

    # probability of each test sample being in each class
    # for visualisation, each row is a different class, each column is a test sample
    probabilities = np.vstack([distributions[i].pdf(pcatest_data) for i in range(len(PIECES))])
    
    # for full board classification with Gaussian
    # clean: square = 97.4%, board = 97.4%
    # noisy: square = 93.6%, board = 93.6%
    # modifying the probabilities by multiply by weighting of occurrences each class in a particular square on the board
    square_count = 0
    number_of_boards = len(model["position_occurrences"][0]) # this is the number of training boards
    for p_coli in range(probabilities.shape[1]):
        square_count += 1
        for p_rowi in range(probabilities.shape[0]):
            probabilities[p_rowi, p_coli] *= 1 + model["position_occurrences"][square_count % 64].count(PIECES[p_rowi]) / number_of_boards

    labelled_indicies = np.argmax(probabilities, axis=0)
    labelled_classes = [PIECES[i] for i in labelled_indicies]
    """

    # nearest neighbour
    pcatrain_data = np.array(model["fvectors_train"])

    # compact implementation of nearest neighbour without for loop
    x = np.dot(pcatest_data, pcatrain_data.transpose())

    """
    # cosine distance
    # clean: square = 97.4%, board = 97.4%
    # noisy: square = 91.9%, board = 91.9%
    modtest = np.sqrt(np.sum(pcatest_data * pcatest_data, axis=1))
    modtrain = np.sqrt(np.sum(pcatrain_data * pcatrain_data, axis=1))
    cosine_dist = x / np.outer(modtest, modtrain.transpose())
    """

    # euclidean distance
    # clean: square = 98.1%, board = 98.1%
    # noisy: square = 91.9%, board = 91.9%
    euclidean_dist = distance_matrix(pcatest_data, pcatrain_data)

    """
    # normal nearest neighbour
    # list of indicies of train samples each test sample is closest to
    nearest = np.argmin(euclidean_dist, axis=1)
    # labels of closest train samples obtained
    labelled_classes = train_labels[nearest]
    """

    # k nearest neighbour
    # k = 8, with euclidean distance
    # clean: square = 98.1%, board = 98.1%
    # noisy: square = 94.5%, board = 94.5%
    k = 8
    labelled_classes = []
    square_count = 0

    for test_i in range(euclidean_dist.shape[0]):
        # list of k indicies of training samples each test sample is closest to
        k_nearest_i = np.argpartition(euclidean_dist[test_i], k)[:k]
        k_nearest_labels = train_labels[k_nearest_i].tolist()

        # classify using a counting system for each label (most common class in k samples)
        # classified_label = max(set(k_nearest_labels), key=k_nearest_labels.count)

        k_nearest_dist = euclidean_dist[test_i, k_nearest_i]
        class_dist_mapping = {}
        
        # classify using a weighting system for each label
        for i in range(len(k_nearest_i)):
            if k_nearest_labels[i] in class_dist_mapping.keys():
                class_dist_mapping[k_nearest_labels[i]] += 1/k_nearest_dist[i]
            else:
                class_dist_mapping[k_nearest_labels[i]] = 1/k_nearest_dist[i]

        # modify weightings according to occurence of piecs in a particular square on the board
        occurrence_probability_mapping = {}
        number_of_boards = len(model["position_occurrences"][0]) # this is the number of training boards

        for label in PIECES:
            occurrence_probability_mapping[label] = 1 + model["position_occurrences"][square_count % 64].count(label) / number_of_boards

        for label in class_dist_mapping.keys():
            class_dist_mapping[label] *= occurrence_probability_mapping[label]

        # label sample with key (class) with biggest weighting
        # (smaller the distance, more similar, the bigger the weighting, hence find max)
        classified_label = max(class_dist_mapping, key=class_dist_mapping.get)

        labelled_classes.append(classified_label)

        square_count += 1

        """
        print(classified_label)
        # add test sample to training set so remaining test samples can take into consideration of these
        np.append(train_labels, classified_label)
        np.append(pcatrain_data, pcatest_data[test_i])
        euclidean_dist = distance_matrix(pcatest_data, pcatrain_data)
        """

    return labelled_classes


# The functions below must all be provided in your solution. Think of them
# as an API that is used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    
    # dimensionality reduction using PCA
    pcatrain_data = np.dot((data - np.mean(data)), model["A"])

    return pcatrain_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labelled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels.

    model = {}
    model["labels_train"] = labels_train.tolist()
    
    # dimensionality reduction using PCA
    covx = np.cov(fvectors_train, rowvar=0)
    N = covx.shape[0]

    # v is the eigenvector of the covariance matrix covx

    # the function eigh return the eigenvectors (e.g. principal component axes)
    # as column vectors in the maxtrix v sorted by the eigenvalues w, from smallest to largest
    w, v = scipy.linalg.eigh(covx, eigvals=(N-N_DIMENSIONS, N-1)) # return largest eigenvalues
    v = np.fliplr(v)

    model["A"] = v.tolist()
    
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()

    """
    # data processing for Gaussian
    # separate data for training into different classes
    class_sets = [fvectors_train_reduced[labels_train[:]==piece, :] for piece in PIECES]

    # parameters for classification with Gaussian distributions
    means = [np.mean(class_sets[i][:, :], axis=0).tolist() for i in range(len(PIECES))]
    covariances = [np.cov(class_sets[i][:, :], rowvar=0).tolist() for i in range(len(PIECES))]

    model["means"] = means
    model["covariances"] = covariances
    """

    # for full board classification
    # every occurrences of classes in every particular position on the board
    position_occurrences = [[] for _ in range(64)]
    for i in range(len(labels_train)):
        position_occurrences[i % 64].append(labels_train[i])

    model["position_occurrences"] = position_occurrences
    
    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape # h and w of square image represented in a numpy array 50x50
    n_features = h * w
    # taking each row of pixels of a square image and concantenate it into one long strip
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        # applying to each entire row
        fvectors[i, :] = image.reshape(1, n_features)

    # fvectors is a 2d np array of x rows each representing an square image
    # and y (50x50=2500) columns representing all the pixels of a square image
    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    # ideas for further improving k-NN:
    # 1. after a test sample has been classified, add it to the training dataset
    #   I did manage to implement this, however it was not a very efficient approach and the code takes a long time to run
    # 2. classify using a weighting system for each label instead of counting the labels
    #   Successfully implemented and improved accuracy of both clean and noisy data

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # An idea that I came up with, when I was going by the Gaussain approach was:
    # construct a list of lists of size 64
    #   each list stores every label (include duplicates) thats occurred in a particular position
    #   add probability of occurence of a class to 1, and multiply the probability by the probability a class belongs to a particular Gaussian distribution
    #   (adding 1 so we don't totally eliminate an probability by multiplying by 0, if there were no occurence of a piece on a particular position)
    #   classify squares according to modified probabilities

    # The above idea was modifed for k-NN where the weighting system is futher modified to take into account of occurrence of pieces in a square

    return classify_squares(fvectors_test, model)
