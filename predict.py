#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Applies a trained CNN to given input data.

Author: Jan Schlüter
"""

import sys
import h5py
import numpy as np
import scipy.signal
import scipy.ndimage
if map(int, scipy.__version__.split('.')) < (0, 12, 0):
    import warnings
    warnings.simplefilter("ignore", np.ComplexWarning)

try:
    from scipy.special import expit as sigmoid
except ImportError:
    def sigmoid(x, out):
        if out is not x:
            out[:] = x
        np.negative(out, out)
        np.exp(out, out)
        out += 1
        np.reciprocal(out, out)
        return out

def lrelu(x, out):
    if out is not x:
        out[:] = x
    out[x < 0] *= .01
    return out

transfuns = {
    'tanh': np.tanh,
    'relu': lambda x, out: np.maximum(0, x, out),
    'lrelu': lrelu,
    'sigmoid': sigmoid,
}

def load_cnn(modelfile, modelvars=None):
    arch = modelvars and modelvars.get('cnn.arch', None)
    if not arch:
        raise ValueError("missing architecture definition")
    if modelfile.endswith('.h5'):
        import h5py
        with h5py.File(modelfile, 'r') as f:
            params = {k: f[k].value for k in f.keys()}
    else:
        f = np.load(modelfile)
        params = {k: f[k] for k in f.files}
        f.close()
    params = [params['param%d' % i] for i in range(len(params))]
    model = []
    arch = [layerdef.split(':') for layerdef in arch.split(',')]
    while arch:
        kind, shape = arch.pop(0)
        if kind == 'conv':
            layerparams = {'W': params.pop(0), 'bias': params.pop(0)}
            if arch[0][0] == 'pool':
                _, poolshape = arch.pop(0)
                layerparams['maxpool'] = tuple(map(int, poolshape.split('x')))
            model.append((kind, lrelu, layerparams))
        elif kind == 'dense':
            layerparams = {'W': params.pop(0), 'bias': params.pop(0)}
            model.append((kind, lrelu, layerparams))
    if params:
        model.append(('dense', sigmoid, {'W': params.pop(0), 'bias': params.pop(0)}))
    return model

def forward_pass(datapoints, blocklen, model):
    # this is a non-optimal reimplementation of what could be done with theano,
    # to avoid the dependency on theano.
    # it simulates extracting blocks of the given blocklen with a hopsize of 1
    # and passing them through the given model. to avoid redundant computation,
    # it applies the convolution and max-pooling to the full input at once,
    # keeping track of the blocklen, then forms blocks for the fully-connected
    # layers only.

    def pool(data, shape, blocklen):
        # data is a list of offsets,
        # each offset is a list of channels,
        # each channel is a 2D numpy array.
        # If we wanted to expand this into blocks, we would take blocks of
        # `blocklen` frames of all channels in a round-robin manner over the
        # offsets, i.e.:
        # data[0][:][0:blocklen], data[1][:][0:blocklen],
        # data[2][:][0:blocklen], data[3][:][0:blocklen], ...,
        # data[0][:][1:blocklen+1], data[1][:][1:blocklen+1],
        # data[2][:][1:blocklen+1], data[3][:][1:blocklen+1], ...,
        # Initially, `forward_pass` starts with just a single offset. Each
        # time we max-pool over time by a factor of N, the number of
        # different offsets to consider increases by a factor of N. By
        # managing different offsets, we can perform convolution and pooling
        # over the 2D numpy arrays without dividing them into blocks (and
        # running into redundant computations), even when pooling over time.

        # first, we just apply a sliding maximum filter for each channel,
        # subsampling in frequency direction, but not in time direction
        mf = scipy.ndimage.filters.maximum_filter
        data = [[mf(channel, tuple(shape), mode='constant')[:, shape[1]//2::shape[1]]
                for channel in offset] for offset in data]

        # shortcut: if we do not pool over time, we can return early
        if shape[0] <= 1:
            return data

        # when pooling over time, from the sliding maximum we need to ignore
        # the last few frames that involved zero-padding,
        trim_right = (shape[0] - 1) // 2
        # plus the last few frames that might fall away if we'd actually
        # extract the blocks and pooled them
        trim_right += blocklen % shape[0]

        # we iterate over the combination of newly created offsets (due to
        # pooling over time) and the existing different input offsets in
        # correct order to yield an output following the input convention
        output = []
        for off in xrange(shape[0]):
            for offset in data:
                piece = [channel[shape[0]//2 + off:-trim_right or None:shape[0]]
                        for channel in offset]
                if len(piece[0]) >= blocklen // shape[0]:
                    output.append(piece)

        # that's it
        return output

    def conv(datapoints, blocklen, transfun, W, bias, maxpool=None):
        # convolve each offset and channel separately with each filter, summing up results over input channels
        datapoints = [[sum(scipy.signal.convolve2d(channel, w[c], mode='valid') for c, channel in enumerate(offset)) for w in W] for offset in datapoints]
        blocklen = blocklen - W.shape[2] + 1
        # max-pool
        if maxpool is not None:
            datapoints = pool(datapoints, maxpool, blocklen)
            blocklen = blocklen // maxpool[0]

        # join channels to single matrix
        datapoints = [np.asarray(offset) for offset in datapoints]
        for offset in datapoints:
            # add bias per output channel
            offset += bias[:,np.newaxis,np.newaxis]
            # apply nonlinearity (in-place)
            transfun(offset, offset)
        return datapoints, blocklen

    def full(datapoints, blocklen, transfun, W, bias):
        if blocklen <= 0:
            raise ValueError("blocklen must be greater than zero")
        # if datapoints is not a 2D array, we turn it into one by copying out
        # blocks (possibly in round-robin across different offsets) as needed
        if isinstance(datapoints, list) or (datapoints.ndim == 4):
            num_offsets = len(datapoints)
            if (blocklen == 1) and (num_offsets == 1):
                # shortcut: block length 1 and single offset: just reshape it
                datapoints = datapoints[0,0]
            else:
                datapoints = np.vstack(datapoints[off][:,idx:idx+blocklen].ravel()
                        for idx in xrange(datapoints[0].shape[1] - blocklen + 1)
                        for off in xrange(num_offsets)
                        if idx+blocklen <= datapoints[off].shape[1])
        # compute neuron activations
        datapoints = np.dot(datapoints, W)
        datapoints += bias
        # apply nonlinearity (in-place)
        transfun(datapoints, datapoints)
        return datapoints, blocklen

    # add an extra dimension to handle multiple different max-pooling offsets
    datapoints = datapoints[np.newaxis]
    for layertype, transfun, layerparams in model:
        #print layertype, layerparams['W'].shape
        if layertype == 'conv':
            datapoints, blocklen = conv(datapoints, blocklen, transfun, **layerparams)
        elif layertype == 'dense':
            datapoints, blocklen = full(datapoints, blocklen, transfun, **layerparams)
        else:
            raise ValueError('unknown layer type %s' % layertype)
    return datapoints

def lasagne_forward_pass(indata, blocklen, model):
    # for some reason, we cannot have a global import for this, otherwise the
    # WSGI web app never comes up.
    import lasagne
    import theano
    import theano.tensor as T
    class DilatedMaxPool2DLayer(lasagne.layers.MaxPool2DLayer):
        def __init__(self, *args, **kwargs):
            dilation = kwargs.pop('dilation', (1, 1))
            super(DilatedMaxPool2DLayer, self).__init__(*args, **kwargs)
            self.dilation = lasagne.utils.as_tuple(dilation, 2, int)
            assert self.dilation[1] == 1, "only implemented dilation over time"
            assert self.stride[0] == 1, "require unstrided pooling over time"
        def get_output_shape_for(self, input_shape):
            shape = super(DilatedMaxPool2DLayer, self).get_output_shape_for(input_shape)
            return (shape[0], shape[1],
                    lasagne.layers.pool.pool_output_length(input_shape[2],
                        pool_size=(self.pool_size[0] - 1) * self.dilation[0] + 1,
                        stride=self.stride[0], pad=self.pad[0], ignore_border=self.ignore_border),
                    shape[3])
        def get_output_for(self, input, **kwargs):
            input_shape = input.shape
            if self.dilation[0] > 1:
                pad_w = (self.dilation[0] - input_shape[2] % self.dilation[0]) % self.dilation[0]
                input = T.concatenate((input, T.zeros((input_shape[0], input_shape[1], pad_w, input_shape[3]), input.dtype)), axis=2)
                input = input.reshape((input_shape[0], input_shape[1], -1, self.dilation[0], input_shape[3]))
                input = input.transpose(0, 3, 1, 2, 4)
                input = input.reshape((-1,) + tuple(input.shape[2:]))
            output = super(DilatedMaxPool2DLayer, self).get_output_for(input, **kwargs)
            if self.dilation[0] > 1:
                output = output.reshape((input_shape[0], self.dilation[0]) + tuple(output.shape[1:]))
                output = output.transpose(0, 2, 3, 1, 4)
                output = output.reshape((input_shape[0], output.shape[1], -1, output.shape[4]))
                output = output[:, :, :output.shape[2] - pad_w]
            return output

    invar = theano.tensor.tensor4('input')
    layer = lasagne.layers.InputLayer((1, indata.shape[0], None, indata.shape[2]), invar)
    dilate = 1
    for layertype, transfun, layerparams in model:
        #print layertype, layer.output_shape, layerparams['W'].shape
        if layertype == 'conv':
            W = layerparams['W']
            kwargs = dict(num_filters=W.shape[0], filter_size=W.shape[-2:], b=layerparams['bias'], nonlinearity=None)
            if dilate != 1:
                layer = lasagne.layers.DilatedConv2DLayer(layer, W=W.transpose(1,0,2,3)[:,:,::-1,::-1], dilation=(dilate, 1), **kwargs)
            else:
                layer = lasagne.layers.Conv2DLayer(layer, W=W, **kwargs)
            blocklen = blocklen - W.shape[2] + 1
            if layerparams.get('maxpool', None):
                poolshape = layerparams['maxpool']
                if dilate != 1:
                    layer = DilatedMaxPool2DLayer(layer, poolshape, stride=(1, poolshape[1]), dilation=(dilate, 1))
                else:
                    layer = lasagne.layers.MaxPool2DLayer(layer, poolshape, stride=(1, poolshape[1]))
                blocklen = blocklen // poolshape[0]
                dilate *= poolshape[0]  # need to increase dilation for future layers
        elif layertype == 'dense':
            if (blocklen > 1) or (dilate != 1):
                W = layerparams['W']
                W = W.T.reshape((W.shape[1], layer.output_shape[1], blocklen, layer.output_shape[-1]))
                layer = lasagne.layers.DilatedConv2DLayer(layer, num_filters=W.shape[0], filter_size=W.shape[-2:], W=W.transpose(1,0,2,3), b=layerparams['bias'], dilation=(dilate, 1), nonlinearity=None)
                layer = lasagne.layers.DimshuffleLayer(layer, (2, 0, 1, 3))
                blocklen = 1
                dilate = 1
            else:
                W = layerparams['W']
                layer = lasagne.layers.DenseLayer(layer, num_units=W.shape[1], W=W, b=layerparams['bias'], nonlinearity=None)
        if transfun is transfuns['lrelu']:
            layer = lasagne.layers.NonlinearityLayer(layer, lasagne.nonlinearities.leaky_rectify)
        elif transfun is transfuns['sigmoid']:
            layer = lasagne.layers.NonlinearityLayer(layer, lasagne.nonlinearities.sigmoid)
    #print "compiling..."
    fn = theano.function([invar], lasagne.layers.get_output(layer, deterministic=True))
    #print "computing..."
    return fn(indata[np.newaxis]), fn


def apply_cnn(indata, blocklen, modelfile, autopad=True, modelvars=None):
    try:
        import lasagne
    except ImportError:
        lasagne = None
    # stack channels if needed
    if isinstance(indata, (list, tuple)):
        indata = np.vstack(channel[np.newaxis, ...] for channel in indata)  # num_channels x num_datapoints x num_feats
    # zero-pad if needed (actually, we pad by repeating the first or last frame)
    if autopad and blocklen > 1:
        # XXX: this adds one frame too little for an even blocklen
        zeropad = np.zeros((indata.shape[0], blocklen/2, indata.shape[2]), dtype=indata.dtype)
        indata = np.concatenate((zeropad + indata[:,:1], indata, zeropad + indata[:,-1:]), axis=1)
    # shortcut: if we have lasagne and we cached the compiled function, call it
    if lasagne is not None and (modelfile, blocklen) in apply_cnn.cache:
        fn = apply_cnn.cache[(modelfile, blocklen)]
        return fn(indata[np.newaxis])
    # load cnn model
    model = load_cnn(modelfile, modelvars)
    # pass the full data array to the CNN
    if lasagne is not None:
        outputs, fn = lasagne_forward_pass(indata, blocklen, model)
        apply_cnn.cache[(modelfile, blocklen)] = fn
    else:
        outputs = forward_pass(indata, blocklen, model)
    return outputs
apply_cnn.cache = {}


def pad_data(block, pad_left, pad_right):
    # pad block
    out = np.empty((block.shape[0] + pad_left + pad_right,) + block.shape[1:], block.dtype)
    out[:pad_left] = 0
    out[pad_left:-pad_right or None] = block
    out[-pad_right or len(out):] = 0
    return out

def predict(spect, modelfile, stdfile):
    with open(modelfile + '.vars', 'rb') as f:
        modelvars = dict(l.rstrip('\r\n').split('=') for l in f if l.rstrip('\r\n'))
    blocklen = int(modelvars.get('spect.blocksize', 115))

    # padding
    pad_left  = blocklen // 2
    pad_right = blocklen - pad_left
    spect = pad_data(spect, pad_left, pad_right)

    # z-scoring
    if stdfile:
        with h5py.File(stdfile, 'r') as f:
            stdmean = {k: v.value for k, v in f.iteritems()}
        spect -= stdmean['mean'].ravel()
        spect /= stdmean['std'].ravel()

    # CNN
    activations = apply_cnn(spect[np.newaxis], blocklen=blocklen, modelfile=modelfile, autopad=False, modelvars=modelvars)

    # smoothing
    activations = scipy.ndimage.filters.median_filter(activations, (56,) + (1,) * (activations.ndim - 1), mode='nearest')

    return activations
