import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    indicator = 0
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        indicator += 1
        dW[:,j] += X[i]
    dW[:,y[i]] += -1 * indicator * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)


  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW /= num_train
  dW += 2 * reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train = X.shape[0]
  delta = 1

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #pass
  # Calculate all scores
  scores = X.dot(W)
  # For each example calc the diff between the score from each class with the
  # score for the correct class plus delta
  margins = scores - scores[np.arange(num_train),(y[:,]),None] + delta
  # For each example, set margin for accurate class to 0
  margins[np.arange(num_train),(y[:,])] = 0

  # For each class in each training example take the max between 0 and the diff
  #    maxmatrix = np.maximum(0, margins)
  # Sum each row to get the loss per training example
  #    loss_per_example = np.sum(maxmatrix, axis=1)
  loss_per_example = np.sum(
      np.maximum(0, margins),
      axis=1)
  # Sum the losses for all examples and divide by mean loss
  loss = np.sum(loss_per_example) / num_train
  # Add regularization to the loss
  loss += reg * np.sum(W * W)

 #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #pass
  # Use indicator matrix to store multiplier for gradient calculation
  # True-negatives will have indicator 0
  # False positives indicator should be 1
  # For the actual class the indicator with be a count of all false positives
  # multiplied by -1
  # Each of these indicators will be multiplied by X
  indicator = np.zeros((num_train, num_classes))
  indicator[margins > 0] = 1
  # Set correct multiplier for correct class
  sum_class_with_loss_per_example = np.sum(indicator,axis=1)
  # multiplier for correct classes
  indicator[np.arange(num_train),(y[:,])] = -1 * sum_class_with_loss_per_example
  dW = X.T.dot(indicator)
  dW /= num_train
  dW += 2 * reg * W


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
