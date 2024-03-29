from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    #pass
    num_train = x.shape[0]
    x_shape = x.shape
    x = np.reshape(x, (num_train, -1))

    out = np.dot(x, w) + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, x_shape)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b, x_shape = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    #pass
    dx = np.dot(dout, w.T)
    dx = np.reshape(dx, x_shape)
    dw = np.dot(x.T, dout)
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    #pass
    out = np.maximum(0,x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    #pass
    dx = dout
    dx[x <= 0] = 0

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        #pass
        num_train = x.shape[0]
        sample_mean = np.mean(x, axis=0)
        zero_mean_X = x - sample_mean
        sample_var = np.sum(np.square(zero_mean_X), axis=0) / num_train
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        norm_X = zero_mean_X / np.sqrt(sample_var + eps)
        out = gamma*norm_X + beta
        sample_var_p_eps = sample_var + eps

        cache = (zero_mean_X, sample_var_p_eps, norm_X, gamma)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        #pass
        out = (x - running_mean) / np.sqrt(running_var + eps)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    #pass
    zero_mean_X, sample_var_p_eps, norm_X, gamma = cache
    m = norm_X.shape[0]

    dbeta = np.sum(dout, axis=0).T #(Dx1)
    dgamma = np.sum(dout * norm_X, axis=0).T #(Dx1)

    # Upstream derivative of normalized X aka x hat
    d_norm_X = (dout*gamma)  # NxD

    # denominator is std dev
    denominator = 1/np.sqrt(sample_var_p_eps)
    d_norm_X_wrt_nominator = denominator * d_norm_X #NxD
    d_norm_X_wrt_denom = np.sum(zero_mean_X * d_norm_X, axis=0) #Dx1

    d_local_sqrt_sample_var_p_eps = -0.5 / (sample_var_p_eps ** (3/2)) #Dx1

    d_variance_upstream = d_local_sqrt_sample_var_p_eps * d_norm_X_wrt_denom #Dx1
    d_var_wrt_zero_mu_X = (2 * zero_mean_X) * (d_variance_upstream/m) #NxD broadcast

    d_z_mean_X_upstream = d_norm_X_wrt_nominator + d_var_wrt_zero_mu_X #(NxD)

    d_zero_mu_X_wrt_mean = -1 * np.sum(d_z_mean_X_upstream, axis=0) #1xD
    d_mean_wrt_x = (d_zero_mu_X_wrt_mean / m)   #1xD
    d_mean_wrt_x = np.tile(d_mean_wrt_x, (m,1))   # NxD
    dx = d_z_mean_X_upstream + d_mean_wrt_x

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    #pass
    _, sample_var_p_eps, norm_X, gamma = cache
    m = norm_X.shape[0]

    dbeta = np.sum(dout, axis=0).T #(Dx1)
    dgamma = np.sum(dout * norm_X, axis=0).T #(Dx1)

    # Upstream derivative of normalized X aka x hat
    d_norm_X = (dout*gamma)  # NxD

    dx = np.array( (m*d_norm_X - np.sum(d_norm_X, axis=0) - \
         norm_X * np.sum(d_norm_X * norm_X, axis=0)) * \
         (1/ (m * np.sqrt(sample_var_p_eps))) )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    #pass
    num_feat = x.shape[1]
    sample_mean = np.mean(x, axis=1)
    zero_mean_X = x - sample_mean[:,None]
    sample_var = np.sum(np.square(zero_mean_X), axis=1) / num_feat
    norm_X = zero_mean_X / np.sqrt(sample_var + eps)[:,None]
    out = gamma * norm_X + beta
    sample_var_p_eps = sample_var + eps

    cache = (zero_mean_X, sample_var_p_eps, norm_X, gamma)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    #pass
    _, sample_var_p_eps, norm_X, gamma = cache
    m = norm_X.shape[1]

    dbeta = np.sum(dout, axis=0, keepdims=True) #(Dx1)
    print('dbeta', dbeta.shape)
    dgamma = np.sum(dout * norm_X, axis=0, keepdims=True) #(Dx1)
    print('dgamma', dgamma.shape)

    # Upstream derivative of normalized X aka x hat
    d_norm_X = (dout*gamma)  # NxD

    dx = np.array( (m*d_norm_X - np.sum(d_norm_X, axis=1)[:,None] - \
         norm_X * np.sum(d_norm_X * norm_X, axis=1)[:,None]) * \
         (1/ (m * np.sqrt(sample_var_p_eps)))[:,None] )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        #pass
        # Create mask in shape of input
        # scale values by p so they don't have to be adjusted during test time
        mask = (np.random.rand(*x.shape) < p) / p

        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        #pass
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        #pass
        dx = mask * dout
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param, verbose=False):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    #pass
    num_train = x.shape[0]
    x_h = x.shape[2]
    x_w = x.shape[3]
    num_filter = w.shape[0]
    f_h = w.shape[2]
    f_w = w.shape[3]
    pad = conv_param['pad']
    stride = conv_param['stride']
    out_h = int((x_h + 2*pad - f_h) / stride + 1)
    out_w = int((x_w + 2*pad - f_w) / stride + 1)
    out = np.zeros((num_train, num_filter, out_h, out_w))

    if verbose: print("\n*** x ***\n",x)
    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=(0, 0))
    if verbose: print("\n*** x_pad ***\n",x_pad)
    if verbose: print("\n*** x ***\n",x)
    for f in range(num_filter):
        #print('f ', f)
        filter = w[f,:,:,:]
        #print('Filter size', filter.shape)
        for i in range(num_train):
            for vert_pos in range(out_h):
                frame_beg_h = vert_pos * stride
                frame_end_h = frame_beg_h + f_h
                h_slice = slice(frame_beg_h,frame_end_h)
                for hor_pos in range(out_w):
                    frame_beg_w = hor_pos * stride
                    frame_end_w = frame_beg_w + f_w
                    w_slice = slice(frame_beg_w, frame_end_w)
                    #print('h_slice', h_slice, ' w_slice ', w_slice)
                    #print('h %d w %d' % (vert_pos, hor_pos))
                    #print('xpad section\n', x_pad[i,:, h_slice,w_slice])
                    #test = np.sum(x_pad[i,:,h_slice,w_slice] * filter) + b[f]
                    #print('test shape', test.shape)
                    #print("test value", test)
                    out[i, f , vert_pos, hor_pos] = np.sum(x_pad[i,:,h_slice,w_slice] * filter) + b[f]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    #pass

    (x, w, b, conv_param) = cache
    (num_train, _, x_h, x_w) = x.shape
    (num_filter, _, f_h, f_w) = w.shape
    (_, _, out_h, out_w) = dout.shape
    pad = conv_param['pad']
    stride = conv_param['stride']

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=(0, 0))
    dx_pad = np.zeros_like(x_pad)
    for f in range(num_filter):
        filter = w[f,:,:,:]
        for i in range(num_train):
            for vert_pos in range(out_h):
                frame_beg_h = vert_pos * stride
                frame_end_h = frame_beg_h + f_h
                h_slice = slice(frame_beg_h,frame_end_h)
                for hor_pos in range(out_w):
                    frame_beg_w = hor_pos * stride
                    frame_end_w = frame_beg_w + f_w
                    w_slice = slice(frame_beg_w, frame_end_w)

                    dx_pad[i, : , h_slice, w_slice] += filter * dout[i, f, vert_pos, hor_pos]
                    dw[f, :, :, :] += x_pad[i, :, h_slice, w_slice] * dout[i,f, vert_pos, hor_pos]
                    db[f] += dout[i, f, vert_pos, hor_pos]

    dx = dx_pad[:, :, 1:-1, 1:-1]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param, verbose=False):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    #pass
    (num_train, num_c, x_h, x_w) = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']
    out_h = int((x_h - pool_h) / stride + 1)
    out_w = int((x_w - pool_w) / stride + 1)

    out = np.zeros((num_train, num_c, out_h, out_w))
    for i in range(num_train):
        for vert_pos in range(out_h):
            frame_beg_h = vert_pos * stride
            frame_end_h = frame_beg_h + pool_h
            h_slice = slice(frame_beg_h,frame_end_h)
            for hor_pos in range(out_w):
                frame_beg_w = hor_pos * stride
                frame_end_w = frame_beg_w + pool_w
                w_slice = slice(frame_beg_w, frame_end_w)
                if verbose:
                    print('x slice \n', x[i,:, h_slice,w_slice])
                    print('max\n',np.amax(x[i,:, h_slice,w_slice], axis=(1,2), keepdims=False))
                    print('max shape', np.amax(x[i,:, h_slice,w_slice], axis=(1,2), keepdims=False).shape)
                    ind = np.unravel_index(np.argmax(x[i, :, h_slice, w_slice]), x[i, :, h_slice, w_slice].shape)
                    print("ind", ind)
                out[i,:,vert_pos,hor_pos] = \
                    np.amax(x[i,:, h_slice,w_slice], axis=(1,2), keepdims=False)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache, verbose=False):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    #pass
    (x, pool_param) = cache
    (num_train, num_c, x_h, x_w) = x.shape
    (_, _, up_h, up_w) = dout.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']

    dx = np.zeros_like(x)
    for i in range(num_train):
        for vert_pos in range(up_h):
            frame_beg_h = vert_pos * stride
            frame_end_h = frame_beg_h + pool_h
            h_slice = slice(frame_beg_h,frame_end_h)
            for hor_pos in range(up_w):
                frame_beg_w = hor_pos * stride
                frame_end_w = frame_beg_w + pool_w
                w_slice = slice(frame_beg_w, frame_end_w)
                for c in range(num_c):
                    ind = np.unravel_index(np.argmax(x[i, c, h_slice, w_slice]), x[i, c, h_slice, w_slice].shape)
                    ind_h = frame_beg_h + ind[0]
                    ind_w = frame_beg_w + ind[1]
                    if verbose:
                        print('** c ', c)
                        print('x slice \n', x[i,c, h_slice,w_slice])
                        print('max\n',np.amax(x[i,c, h_slice,w_slice], keepdims=False))
                        print('**max ind: ', ind)
                        print("***access arg by idx ", x[i, c, ind_h, ind_w])

                    dx[i, c, ind_h, ind_w] = dout[i,c, vert_pos, hor_pos]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    #pass
    # transpose data to isolate C and then reshape since
    # batchnorm_forward expects 2d input
    # will generate means and std dev for each channel C
    (N, C, H, W) = x.shape
    x_new = x.transpose(0, 2, 3, 1)
    x_new = np.reshape(x_new, (-1, C), 'C')
    (out, cache) = batchnorm_forward(x_new, gamma, beta, bn_param)
    out = np.reshape(out, (N, H, W, C))
    out = out.transpose(0, 3, 1, 2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    #pass
    (N, C, H, W) = dout.shape
    dout_new = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx, dgamma, dbeta = batchnorm_backward(dout_new, cache)
    dx = np.reshape(dx, (N, H, W, C)).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape  (1,C,1,1)
    - beta: Shift parameter, of shape  (1,C,1,1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    #pass


    (N, C, H, W) = x.shape
    group_depth = C // G
    #print('x shape', x.shape)

    x_new = x.reshape(N, G, group_depth, H, W)
    #print('x new shape', x_new.shape)

    sample_mean = np.mean(x_new, axis=2, keepdims=True)
    #print('shape mean', sample_mean.shape)
    zero_mean_X = x_new - sample_mean
    sample_var = np.sum(np.square(zero_mean_X), axis=2, keepdims=True) / group_depth
    norm_X = zero_mean_X / np.sqrt(sample_var + eps)
    norm_X = norm_X.reshape(N, C, H, W)
    out = gamma * norm_X + beta
    sample_var_p_eps = sample_var + eps

    #out = out.reshape(N, C, H, W)

    cache = (sample_var_p_eps, norm_X, gamma, G)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape  (1,C,1,1)
    - dbeta: Gradient with respect to shift parameter, of shape  (1,C,1,1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    #pass
    sample_var_p_eps, norm_X, gamma, G = cache
    (N, C, H, W) = norm_X.shape
    #print('orig x shape', norm_X.shape)
    group_depth = C // G
    gnorm_X = norm_X.reshape(N, G, group_depth, H, W)
    #print('resized norm x shape ', gnorm_X.shape)

    dbeta = np.sum(dout, axis=(0,2,3), keepdims=True) #(1,C,1,1)
    #print('dbeta shape', dbeta.shape)
    dgamma = np.sum(dout * norm_X, axis=(0,2,3), keepdims=True) #(1,C,1,1)
    #print('dgamma shape', dgamma.shape)

    # Upstream derivative of normalized_X aka x_hat
    d_norm_X = (dout*gamma)
    d_norm_X = d_norm_X.reshape(N, G, group_depth, H, W)
    #print('dnorm x shape', d_norm_X.shape)

    dx = np.array( (group_depth*d_norm_X - np.sum(d_norm_X, axis=2, keepdims=True) - \
         gnorm_X * np.sum(d_norm_X * gnorm_X, axis=2, keepdims=True)) * \
         (1/ (group_depth * np.sqrt(sample_var_p_eps))) )

    #print('dx shape', dx.shape)

    dx = dx.reshape(N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
