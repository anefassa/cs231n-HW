import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg, verbose=False):
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
  #pass
  num_train = X.shape[0]
  num_classes = W.shape[1]

  # Will break down each step for ease of troubleshooting
  for i in range(num_train):
    if verbose: 
        print("X", X[i])
        print("W\n", W)
    z = X[i].dot(W)
    # Make it numerically stable
    zmax = np.max(z)
    if verbose: 
        print("z",z)
        print("zmax", zmax)
    z -= zmax
    if verbose: print("z adjusted by max",z)
    # Calculate unnormalized probabilities
    prob = np.exp(z)
    sum_of_p = np.sum(prob)
    normalized_p = prob/sum_of_p
    if verbose: 
        print("prob",prob)
        print("sum_of_p", sum_of_p)
        print("normalized_p", normalized_p)
        print("Sum of normalized_p, must add to 1:", sum(normalized_p))

    loss_i = -1 * np.log(normalized_p[y[i]])
    loss += loss_i
    if verbose: 
        print("loss_i", loss_i)
        print("sum of loss", loss)

    # Calculate gradients
    dscores = normalized_p
    if verbose: print("dscores \n",dscores)
    dscores[y[i]] -= 1
    dscores /= num_train
    if verbose: 
        print("dscores - 1 for correct class / num_train:\n",dscores)
        print("X[i] shape and value\n", X.T[:,[i]].shape,"\n", X.T[:,[i]])
        print("dscores[None,:].shape and value \n", dscores[None,:].shape, "\n", dscores[None,:] )
    dW +=  np.dot(X.T[:,[i]], dscores[None,:])
    if verbose: print("\ndW\n", dW)
    
  loss = loss / num_train
  if verbose: print("data loss", loss)
  loss += reg * np.sum(W * W)
  if verbose: print("loss w regularization", loss)
  dW += 2*reg*W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg, verbose=False):
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
  #pass
  num_train = X.shape[0]
  # Calculate scores
  scores = X.dot(W)
  # Add numeric stability by subtracting max
  scores -= np.max(scores, axis=1, keepdims=True)
  # Calculate normalized probabilities
  normalized_p = np.exp(scores) / np.sum(np.exp(scores), axis= 1, keepdims=True)
  # Sum loss for each correct class and divide by number of training examples
  loss = np.sum(-1 * np.log(normalized_p[range(num_train),y]))/num_train
  loss += reg * np.sum(W * W)

  # Calculate gradient of loss w.r.t. scores
  dscores = normalized_p
  dscores[range(num_train),y] -= 1
  dscores /= num_train
  # Backpropagate scores gradient to gradient w.r.t W
  dW = np.dot(X.T, dscores)
  dW += 2*reg*W

  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

