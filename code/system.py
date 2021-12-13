"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List

import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import append
from scipy.stats import multivariate_normal
from scipy.spatial import distance_matrix
import scipy.linalg

N_DIMENSIONS = 10

# uppercase white pieces
# lowercase black pieces
# . represents an empty square
PIECES = {".":64, "K":1, "Q":1, "B":2, "N":2, "R":2, "P":8, "k":1, "q":1, "b":2, "n":2, "r":2, "p":8}

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

    pcatest_data = np.dot((test - np.mean(test)), np.array(model["linear_transform_matrix"]))

    """
    # Gaussian distributions
    # clean: square = 97.1%, board = 97.1%
    # noisy: square = 93.6%, board = 93.6%
    distributions = [multivariate_normal(mean=model["means"][i], cov=model["covariances"][i]) for i in range(len(PIECES.keys()))]

    # probability of each test sample being in each class
    # for visualisation, each row is a different class, each column is a test sample
    probabilities = np.vstack([distributions[i].pdf(pcatest_data) for i in range(len(PIECES.keys()))])
    
    # for full board classification with Gaussian
    # clean: square = 97.4%, board = 97.4%
    # noisy: square = 93.6%, board = 93.6%
    # modifying the probabilities by multiply by weighting of occurrences each class in a particular square on the board
    square_count = 0
    number_of_boards = len(model["position_occurrences"][0]) # this is the number of training boards
    for p_coli in range(probabilities.shape[1]):
        square_count += 1
        for p_rowi in range(probabilities.shape[0]):
            probabilities[p_rowi, p_coli] *= 1 + model["position_occurrences"][square_count % 64].count(PIECES.keys()[p_rowi]) / number_of_boards

    labelled_indicies = np.argmax(probabilities, axis=0)
    labelled_classes = [PIECES.keys()[i] for i in labelled_indicies]
    """

    # k nearest neighbour
    pcatrain_data = np.array(model["fvectors_train"])

    # euclidean distance measure
    euclidean_dist = distance_matrix(pcatest_data, pcatrain_data)

    k = 8 # this value of k seems to give the best performance
    labelled_classes = []
    square_count = 0 # used to calcuate position on board of square we are currently classifying

    for test_i in range(euclidean_dist.shape[0]):
        # list of k indicies of training samples each test sample is closest to
        k_nearest_i = np.argpartition(euclidean_dist[test_i], k)[:k]
        k_nearest_labels = train_labels[k_nearest_i].tolist()

        # classify using a weighting system for each label
        k_nearest_dist = euclidean_dist[test_i, k_nearest_i]
        class_dist_mapping = {}

        for i in range(len(k_nearest_i)):
            if k_nearest_labels[i] in class_dist_mapping.keys():
                class_dist_mapping[k_nearest_labels[i]] += 1/k_nearest_dist[i]
            else:
                class_dist_mapping[k_nearest_labels[i]] = 1/k_nearest_dist[i]

        # modify weightings for each class also considering count on number of labels
        for label in class_dist_mapping.keys():
            class_dist_mapping[label] *= k_nearest_labels.count(label)

        # taking into consideration of full board
        # modify weightings
        number_of_boards = len(model["position_occurrences"][0]) # this is the number of training boards
        board_position = square_count % 64 # board position as a number between 0-63
        for label in class_dist_mapping.keys():
            #modify weightings according to occurence of pieces on a particular position
            class_dist_mapping[label] *= 1 + model["position_occurrences"][board_position].count(label) / number_of_boards
            
            # modify weightings of white or black according to current position, nearer top or bottom of board respectively
            position_weighting = 1
            if label != ".":
                if board_position < 8 and label.islower():
                    position_weighting += 0.4
                elif board_position < 16 and label.islower():
                    position_weighting += 0.3
                elif board_position < 24 and label.islower():
                    position_weighting += 0.2
                elif board_position < 32 and label.islower():
                    position_weighting += 0.1
                elif board_position < 40 and label.isupper():
                    position_weighting += 0.1
                elif board_position < 48 and label.isupper():
                    position_weighting += 0.2
                elif board_position < 56 and label.isupper():
                    position_weighting += 0.3
                elif board_position < 64 and label.isupper():
                    position_weighting += 0.4
            
            class_dist_mapping[label] *= position_weighting
                


        # label sample with key (class) with biggest weighting
        # (smaller the distance, more similar, the bigger the weighting, hence find max)
        classified_label = max(class_dist_mapping, key=class_dist_mapping.get)

        # classify using a counting system for each label (most common class in k samples)
        # classified_label = max(set(k_nearest_labels), key=k_nearest_labels.count)

        labelled_classes.append(classified_label)

        square_count += 1

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
    pcatrain_data = np.dot((data - np.mean(data)), model["linear_transform_matrix"])

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

    # w is a list of eigenvalues ordered in ascending order
    # each w corresponds to an eigenvector of the covariance matrix in v
    # an eigenvector is basically a principal component axis
    # we want the eigenvectors which corresponds to the biggest eigenvalues, so the last 10
    w, v = scipy.linalg.eigh(covx, eigvals=(N-N_DIMENSIONS, N-1))
    v = np.fliplr(v)

    # v is an linear transform matrix used to reduce the dimension of the original data
    model["linear_transform_matrix"] = v.tolist()
    
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()

    """
    # data processing for Gaussian
    # separate data for training into different classes
    class_sets = [fvectors_train_reduced[labels_train[:]==piece, :] for piece in PIECES.keys()]

    # parameters for classification with Gaussian distributions
    means = [np.mean(class_sets[i][:, :], axis=0).tolist() for i in range(len(PIECES.keys()))]
    covariances = [np.cov(class_sets[i][:, :], rowvar=0).tolist() for i in range(len(PIECES.keys()))]

    model["means"] = means
    model["covariances"] = covariances
    """

    """
    # Hart Algorithm - reduce training set (and labels)
    prototypes_train = fvectors_train_reduced[0].reshape(1,N_DIMENSIONS)
    fvectors_train_reduced = np.delete(fvectors_train_reduced, 0, 0)

    prototypes_label = labels_train[0]
    labels_train = np.delete(labels_train, 0, 0)

    counter = 0
    i = fvectors_train_reduced.shape[0]

    while counter < i:
        classified_label = classify(prototypes_train, prototypes_label, fvectors_train_reduced[i].reshape(1,N_DIMENSIONS))
        if (classified_label != labels_train[i]):
            prototypes_train = np.append(prototypes_train, fvectors_train_reduced[i])
            fvectors_train_reduced = np.delete(fvectors_train_reduced, i, 0)

            prototypes_label = np.append(prototypes_label, labels_train[i])
            labels_train = np.delete(labels_train, i, 0)

            counter = 0
            i = fvectors_train_reduced.shape[0]
        else:
            counter += 1
    
    model["fvectors_train"] = prototypes_train.tolist()
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
    # 1. classify using a weighting system for each label instead of counting the labels
    #   Successfully implemented and improved accuracy of both clean and noisy data
    # 2. multiply weightings by count of labels to combine the voting system
    # 3. using Hart algorithm to reduce number of training samples while preserving the underlying decision boundaries
    
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

    # An idea that I came up with was
    #   construct a list of lists of size 64, each list stores every label (include duplicates) thats occurred in a particular position
    #   multiply the weightings of each unique class in k-NN samples by the probability of occurence of each class + 1
    #   (add 1 so we don't totally eliminate an probability by multiplying by 0, if there were no occurence of a piece on a particular position)
    #   classify squares according to modified probabilities
    #   this was implemented in classify()

    # on a board, there can't be a more than a certain number occurrences for each class
    #   there can't be more than 1 white kings, more than 2 black rooks, etc
    
    # top half of the board are more likely to be black pieces and bottom half, white pieces
    # as we get closer to the top/bottom board positions when we classify, multiply weightings
    # by a specific factor

    # pieces tends to be closer to and play around with their own pieces, espically kings
    # rooks, bishops, knights and queens can be an exception

    # implement chess game logic restrictions to identify potentially misclassified pieces
    # how do we know which one is misclassified? compare probability of belonging to a class
    
    labels = classify_squares(fvectors_test, model)

    # representing class labels as actual boards
    # array of shape n x 64, each row represents a board
    boards_flattened = np.array(labels).reshape(len(labels)//64, 64)

    boards = []
    for i in range(boards_flattened.shape[0]):
        boards.append(boards_flattened[i].reshape(8, 8))

    return labels
