# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import tensorflow as tf

from patent_classification.model.single_layer import SingleTaskLayer


# class TMM_Generator:

#     def __init__(self, config, num_of_labels):
#         """Constructor

#         Args:
#             config (dict): Model Config
#             num_of_labels (int): Number of labels
#         """ 
#         self.config = config
#         self.num_of_labels = num_of_labels
#         self.outputs = list()
#         for index in range(self.num_of_labels):
#             self.outputs.append(SingleTaskLayer(self.config, 2, "softmax"))

#     def get_outputs(self, input):
#         """Get outputs for the model.

#         Args:
#             input (_type_): _description_

#         Returns:
#             _type_: _description_
#         """
#         probs = []
#         for layer in self.outputs:
#             softmax_x, _ = layer(input)
#             probs.append(softmax_x)
#         return probs


class TMM(tf.keras.layers.Layer):

    def __init__(self, config, num_of_labels):
        """TMM model.

        Args:
            config (dict): Model Config
            num_of_labels (int): Number of labels
        """
        super(TMM, self).__init__()
        self.config = config
        self.num_of_labels = num_of_labels
        self.outputs = list()
        for index in range(self.num_of_labels):
            self.outputs.append(SingleTaskLayer(self.config, 2, "softmax"))

    def call(self, input):
        """Forward Pass.

        Args:
            input (tensor): Model Input

        Returns:
            tensor: Probability values.
        """
        probs = []
        for layer in self.outputs:
            softmax_x, _ = layer(input)
            probs.append(softmax_x)
        return probs
