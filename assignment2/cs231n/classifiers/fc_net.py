from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        #pass
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        #pass
        #    The architecure should be affine - relu - affine - softmax.

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        fcl1_out, fc1_cache = affine_forward(X, W1, b1)
        h1out, relu_cache = relu_forward(fcl1_out)
        scores, fc2_cache = affine_forward(h1out, W2, b2)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #pass
        data_loss, d_softmax = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss = data_loss + reg_loss

        fc2_dx, grads['W2'], grads['b2'] = affine_backward(d_softmax, fc2_cache)
        grads['W2'] += self.reg * W2
        relu_dx = relu_backward(fc2_dx, relu_cache)
        fc1_dx, grads['W1'], grads['b1'] = affine_backward(relu_dx, fc1_cache)
        grads['W1'] += self.reg * W1

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


# Define util function to get string names for layers
def getParamNames(layer):
        """
        Get param names for each layer

        Input / output: Layer id.
        Output: Dictionary keys
        """
        layer = str(layer)
        # W = 'W' + layer
        # b = 'b' + layer
        # a_out = 'a_out' + layer
        # a_cache = 'cache' + layer
        # relu_out = 'relu_out' + layer
        # relu_cache = 'relu_cache' + layer
        # batch_out = 'batch_out' + str(layer)
        # batch_cache = 'batch_cache' + str(layer)
        # return(W, b, a_out, a_cache, relu_out, relu_cache, batch_out, batch_cache)
        #

        paramList = ('W', 'b', 'a_cache', 'relu_cache', 'batch_cache', 'gamma', 'beta', 'drop_cache')
        pIdx = {value: value + layer for (value) in paramList}
        #
        # print(pIdx)
        # for k,v in pIdx.items():
        #     print(k,v)

        return(pIdx)


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        #pass
        l_input_cols = input_dim
        layer_dims = hidden_dims + [num_classes]
        for layer in range(1, self.num_layers + 1):
            W = 'W' + str(layer)
            b = 'b' + str(layer)
            self.params[W] = weight_scale * np.random.randn(l_input_cols, layer_dims[layer-1])
            self.params[b] = np.zeros(layer_dims[layer-1])
            if self.normalization=='batchnorm' and layer < self.num_layers:
                gamma = 'gamma' + str(layer)
                beta = 'beta' + str(layer)
                self.params[gamma] = np.ones(layer_dims[layer-1])
                self.params[beta] = np.zeros(layer_dims[layer-1])

            l_input_cols = layer_dims[layer-1]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        #pass
        l_output = {}
        l_input = X

        # Do forward pass on all layers containing FC and relu
        for layer in range(1, self.num_layers + 1):
            #w_name, b_name, a_out, a_cache, relu_out, relu_cache, batch_out, batch_cache = getParamNames(layer)
            p = getParamNames(layer)
            W = self.params[p['W']]
            b = self.params[p['b']]

            output, l_output[p['a_cache']] = affine_forward(l_input, W, b)
            # Skip relu if this is the last fully connected layer
            if layer < self.num_layers:
                if self.normalization=='batchnorm':
                    output, l_output[p['batch_cache']] = \
                        batchnorm_forward(output, self.params[p['gamma']],
                                      self.params[p['beta']], self.bn_params[layer-1])
                output, l_output[p['relu_cache']] = relu_forward(output)
                if self.use_dropout:
                    output, l_output[p['drop_cache']] = dropout_forward(output, self.dropout_param)
                l_input = output
            else:
                scores = output

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #pass

        # Calculate loss and gradient for softmax
        data_loss, d_softmax = softmax_loss(scores, y)

        reg_W = 0
        input_grad = d_softmax
        for layer in range(self.num_layers, 0, -1):
            #W, b, a_cache, relu_cache, batch_cache = getParamNames(layer)
            p = getParamNames(layer)
            _, FC_w, _, _ = l_output[p['a_cache']]

            if layer < self.num_layers:
                if self.use_dropout:
                    input_grad = dropout_backward(input_grad, l_output[p['drop_cache']])
                input_grad = relu_backward(input_grad, l_output[p['relu_cache']])
                if self.normalization=='batchnorm':
                    input_grad, grads[p['gamma']], grads[p['beta']] = \
                        batchnorm_backward_alt(input_grad, l_output[p['batch_cache']])

            input_grad, grads[p['W']], grads[p['b']] = affine_backward(input_grad, l_output[p['a_cache']])
            grads[p['W']] += self.reg * FC_w
            # Aggregrate sum of square W for regularization loss
            reg_W += (np.sum(FC_w * FC_w))


        reg_loss = 0.5 * self.reg * reg_W
        loss = data_loss + reg_loss

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
