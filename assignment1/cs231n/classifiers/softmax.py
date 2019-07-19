import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_class = W.shape[1]
    for i in range(num_train):
        scores = np.dot(X[i], W)
        l_j = 0
        for j in range(num_class):
            l_j += np.exp(scores[j])
        for j in range(num_class):
            dW[:, j] += np.exp(scores[j]) / l_j * X[i]
            if j == y[i]:
                dW[:, y[i]] -= X[i]
        loss = (loss * i + np.log(l_j) - scores[y[i]]) / (i + 1)
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    # num_class = W.shape[1]
    # dim = X.shape[1]
    scores = np.dot(X, W)
    exp_scores = np.exp(scores)
    normalizer = np.sum(exp_scores, axis=1)
    loss += np.sum(np.log(normalizer)) / num_train
    loss -= np.sum(scores[np.arange(num_train), y]) / num_train
    loss += reg * np.sum(W * W)

    normalized = (exp_scores.T / normalizer).T
    normalized[np.arange(num_train), y] -= 1
    dW = np.dot(X.T, normalized)
    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
