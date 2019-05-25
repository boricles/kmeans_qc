# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import logging

from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class
#from qiskit.aqua.algorithms.classical.svm import (_SVM_Classical_Binary,
#                                                  _SVM_Classical_Multiclass,
#                                                  _RBF_SVC_Estimator)
#from qiskit.aqua.utils import get_num_classes

logger = logging.getLogger(__name__)


class KMeans(QuantumAlgorithm):
    """
    The classical kmeans interface.
    Internally, it will calculate the distance ...
    """

    CONFIGURATION = {
        'name': 'KMeans',
        'description': 'KMeans Algorithm',
        'classical': True,
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'KMeans_schema',
            'type': 'object',
            'additionalProperties': False
        },
        'problems': ['clustering'],
        'depends': [
            {'pluggable_type': 'multiclass_extension'},
        ],
    }

    def __init__(self, training_dataset, test_dataset=None, datapoints=None):
        self.validate(locals())
        super().__init__()
        if training_dataset is None:
            raise AquaError('Training dataset must be provided.')

        self.instance = kmeans_instance

    @classmethod
    def init_params(cls, params, algo_input):
        kmeans_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)

        return cls(algo_input.training_dataset, algo_input.test_dataset,
                   algo_input.datapoints)

    def train(self, data, labels):
        """
        train the kmeans
        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
            labels (numpy.ndarray): Nx1 array, where N is the number of data
        """
        self.instance.train(data, labels)

    def test(self, data, labels):
        """
        test the kmeans
        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
            labels (numpy.ndarray): Nx1 array, where N is the number of data
        Returns:
            float: accuracy
        """
        return self.instance.test(data, labels)

    def predict(self, data):
        """
        predict using the kmeans
        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
        Returns:
            numpy.ndarray: predicted labels, Nx1 array
        """
        return self.instance.predict(data)

    def _run(self):
        return self.instance.run()

    @property
    def label_to_class(self):
        return self.instance.label_to_class

    @property
    def class_to_label(self):
        return self.instance.class_to_label

    @property
    def ret(self):
        return self.instance.ret

    @ret.setter
    def ret(self, new_ret):
        self.instance.ret = new_ret

    def load_model(self, file_path):
        """Load a model from a file path.
        Args:
            file_path (str): tthe path of the saved model.
        """
        self.instance.load_model(file_path)

    def save_model(self, file_path):
        """Save the model to a file path.
        Args:
            file_path (str): a path to save the model.
        """
        self.instance.save_model(file_path)