"""
Stubs for pyClarion Keras integration.

This file includes the following:

-   A polished version of the input/output vectorizer that we had worked on last
    semester. It should work, though it can still be improved (handle multiple
    inputs/outputs).
-   A stub for integration of keras Sequential models into pyClarion flows. I am
    not sure if this will do reinforcement learning because of the update
    algorithm signature. This is something you could experiment with. It should
    just work with binary classifiers and other supervised models.

Installation notes:

Keras does not support Python 3.7, but it supports Python 3.6. pyClarion expects
python >= 3.7, so you will have to hack it until I push out a new version.
Assuming you already have an environment where keras is working (python>=3.6),
here is how:

1.  Get the latest pyClarion version.
2.  Navigate to your pyClarion/ directory.
3.  Open setup.py.
4.  On line 25 you should see `python_requires=">=3.7"` change this to
    `python_requires=">=3.6"` and save.
5.  Now you can install pyClarion as usual by running the following command
    in terminal from within the pyClarion/ folder (where setup.py is):
        pip install -e .
    Do not ommit the '.' in the above, else will throw error.
"""

from pyClarion import *
from typing import List, Iterable, Dict, Hashable, Mapping
import numpy as np


class ActivationMapVectorizer(object):
    """
    Converts between activation maps and vectors for neural network I/O.

    Consistently encodes activation maps for input nodes as vectors and
    activation vectors for network outputs to activation maps for output nodes.

    Note: Currently only processes one set of activations at a time. May be
    extended to process multiple activations simultaneously in the future.
    """

    def __init__(self, inputs, outputs, default_strength):
        """
        Initializer a microfeature to vector converter.

        :param inputs: Expected input nodes in ordered iterable.
        :param outputs: Expected output nodes in ordered iterable.
        :param default_strength: The default activation strength.
        """

        self.inputs = inputs
        self.outputs = outputs
        self.default_strength = default_strength

    def input2vector(self, d):
        """
        Encode input node activations into an input vector.

        :param node: Mapping from nodes to strengths.
        """

        output = np.zeros((len(self.inputs),))
        for index, node in enumerate(self.inputs):
            output[index] = d.get(node, self.default_strength)
        return output

    def vector2output(self, vector):
        """
        Encode activation vector into output node activations.

        :param vector: An activation vector.
        """
        vector = np.squeeze(vector)
        assert len(vector.shape) == 1  # must be one dimensional
        assert len(self.outputs) == vector.shape[0]  # must match self.outputs

        output = {}
        for index, activation in enumerate(vector):
            output[self.outputs[index]] = activation
        return output


class KerasSequentialNet(object):
    """Embeds a Keras Sequential model within a pyClarion FlowRealizer."""

    def __init__(self, model, inputs, outputs, default_strength):
        """
        Initialize a new pyClarion integrated keras network.

        model: Keras Sequential model.
        inputs: List of input nodes (Chunk or Feature) passing activations to
            keras sequential model.
        outputs: List of output nodes (Chunk or Feature) receiving activations
            from keras sequential model.
        default_strength: Callable mapping nodes (Chunk or Feature) to their
            default activation values.
        """

        self.model = model
        self.vectorizer = ActivationMapVectorizer(
            inputs, outputs, default_strength
        )

    def __call__(self, d):
        """
        Process activations with keras network.

        :param d: An mapping from nodes (Chunk and/or Feature) to activations
            representing input to keras model.
        """
    
        ipt_vec = self.vectorizer.input2vector(d)
        prediction = self.model.predict(ipt_vec)
        output = self.vectorizer.vector2output(prediction)
        return output

    def update(self, training_input, training_output):
        """
        Update keras model parameters given behavioral outcomes.

        Note: this may have a different signature for q-learning. TBD.

        :param training_input: Input activation vector/array.
        :param training_output: Expected output activation vector/array.
        """

        batch_size_ = training_input.shape[0]
        self.model.fit(training_input, training_output, epochs=1, batch_size=batch_size_)
