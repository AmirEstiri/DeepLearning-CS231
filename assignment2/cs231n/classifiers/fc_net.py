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

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = np.random.normal(0.0, weight_scale, (input_dim, hidden_dim))
        self.params['W2'] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']
        layer1, cache1 = affine_forward(X, W1, b1)
        relu1, cache_relu1 = relu_forward(layer1)
        scores, cache2 = affine_forward(relu1, W2, b2)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dscores = softmax_loss(scores, y)
        drelu1, dW2, db2 = affine_backward(dscores, cache2)
        dlayer1 = relu_backward(drelu1, cache_relu1)
        dX, dW1, db1 = affine_backward(dlayer1, cache1)
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        loss += 0.5 * self.reg * np.sum(W1 * W1)
        loss += 0.5 * self.reg * np.sum(W2 * W2)
        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['b1'] = db1
        grads['b2'] = db2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


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

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        L = len(hidden_dims)
        self.params['W1'] = np.random.normal(0.0, weight_scale, (input_dim, hidden_dims[0]))
        self.params['b1'] = np.zeros(hidden_dims[0])
        for depth in range(2, L + 1):
            self.params['W' + str(depth)] = np.random.normal(0.0, weight_scale,
                                                             (hidden_dims[depth - 2], hidden_dims[depth - 1]))
            self.params['b' + str(depth)] = np.zeros(hidden_dims[depth - 1])
            if self.normalization == 'batchnorm':
                self.params['gamma' + str(depth - 1)] = np.ones(hidden_dims[depth - 2])
                self.params['beta' + str(depth - 1)] = np.zeros(hidden_dims[depth - 2])
            elif self.normalization == 'layernorm':
                self.params['gamma' + str(depth - 1)] = np.ones(hidden_dims[depth - 2])
                self.params['beta' + str(depth - 1)] = np.zeros(hidden_dims[depth - 2])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization == 'layernorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

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
        if self.normalization == 'batchnorm':
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        L = self.num_layers - 1
        cache = []
        cache_rel = []
        cache_n = []
        cache_drop = []
        if L > 1:
            aff, cachei = affine_forward(X, self.params['W1'], self.params['b1'])
            cache.append(cachei)
            if self.normalization == 'batchnorm':
                n, cache_ni = batchnorm_forward(aff, self.params['gamma1'], self.params['beta1']
                                            , self.bn_params[0])
                cache_n.append(cache_ni)
            elif self.normalization == 'layernorm':
                n, cache_ni = layernorm_forward(aff, self.params['gamma1'], self.params['beta1']
                                                , self.bn_params[0])
                cache_n.append(cache_ni)
            else:
                n = aff
            rel, cache_reli = relu_forward(n)
            cache_rel.append(cache_reli)
            if self.use_dropout:
                print('drop')
                drop, cache_dropi = dropout_forward(rel, self.dropout_param)
                cache_drop.append(cache_dropi)
            else:
                drop = rel
        for depth in range(2, L):
            aff, cachei = affine_forward(drop,
                                         self.params['W' + str(depth)], self.params['b' + str(depth)])
            cache.append(cachei)
            if self.normalization == 'batchnorm':
                n, cache_ni = batchnorm_forward(aff, self.params['gamma' + str(depth)]
                                                , self.params['beta' + str(depth)]
                                                , self.bn_params[depth - 1])
                cache_n.append(cache_ni)
            elif self.normalization == 'layernorm':
                n, cache_ni = layernorm_forward(aff, self.params['gamma' + str(depth)]
                                                , self.params['beta' + str(depth)]
                                                , self.bn_params[depth - 1])
                cache_n.append(cache_ni)
            else:
                n = aff
            rel, cache_reli = relu_forward(n)
            cache_rel.append(cache_reli)
            if self.use_dropout:
                drop, cache_dropi = dropout_forward(rel, self.dropout_param)
                cache_drop.append(cache_dropi)
            else:
                drop = rel
        if L > 1:
            scores, cache_score = affine_forward(drop, self.params['W' + str(L)],
                                                 self.params['b' + str(L)])
            cache.append(cache_score)
        else:
            scores, cache_score = affine_forward(X, self.params['W' + str(L)],
                                                 self.params['b' + str(L)])
            cache.append(cache_score)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dout = softmax_loss(scores, y)
        drel, dW, db = affine_backward(dout, cache[L - 1])
        grads['W' + str(L)] = dW + self.reg * self.params['W' + str(L)]
        grads['b' + str(L)] = db
        loss += 0.5 * self.reg * np.sum(self.params['W' + str(L)] * self.params['W' + str(L)])
        for depth in range(L - 1, 0, -1):
            loss += 0.5 * self.reg * np.sum(self.params['W' + str(depth)] * self.params['W' + str(depth)])
            if self.use_dropout:
                drel = dropout_backward(drel, cache_drop[depth - 1])
            dn = relu_backward(drel, cache_rel[depth - 1])
            if self.normalization == 'batchnorm':
                daff, dgamma, dbeta = batchnorm_backward(dn, cache_n[depth - 1])
                grads['gamma' + str(depth)] = dgamma
                grads['beta' + str(depth)] = dbeta
            elif self.normalization == 'layernorm':
                daff, dgamma, dbeta = layernorm_backward(dn, cache_n[depth - 1])
                grads['gamma' + str(depth)] = dgamma
                grads['beta' + str(depth)] = dbeta
            else:
                daff = dn
            drel, dW, db = affine_backward(daff, cache[depth - 1])
            grads['W' + str(depth)] = dW + self.reg * self.params['W' + str(depth)]
            grads['b' + str(depth)] = db

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
