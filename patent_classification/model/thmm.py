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

ROOT = "<ROOT>"


def extend_hierarchy(hierarchy, y_labs):
    """
    Create Hierarchical Tree

    Args:
        hierarchy (dict): Hierarchical Taxonomy
        y_labs (list): labels

    Returns:
        dict: Hierarchical Taxonomy
    """
    for samples_t in y_labs:
        if not isinstance(samples_t, list):
            samples = [samples_t]
        else:
            samples = samples_t
        for lab in samples:
            par_1 = lab[0]
            par_2 = lab[:3]
            child = lab[:]

            if par_1 not in hierarchy[ROOT]:
                hierarchy[ROOT].append(par_1)
            if par_1 not in hierarchy:
                hierarchy[par_1] = [par_2]
            else:
                if par_2 not in hierarchy[par_1]:
                    hierarchy[par_1].append(par_2)
            if par_2 not in hierarchy:
                hierarchy[par_2] = [child]
            else:
                if child not in hierarchy[par_2]:
                    hierarchy[par_2].append(child)
    return hierarchy


def build_hierarchy(issues):
    hierarchy = {ROOT: []}
    for i in issues:
        par_1 = i[0]
        par_2 = i[:3]
        child = i[:]

        if par_1 not in hierarchy[ROOT]:
            hierarchy[ROOT].append(par_1)
        if par_1 not in hierarchy:
            hierarchy[par_1] = [par_2]
        else:
            if par_2 not in hierarchy[par_1]:
                hierarchy[par_1].append(par_2)
        if par_2 not in hierarchy:
            hierarchy[par_2] = [child]
        else:
            hierarchy[par_2].append(child)
    return hierarchy


class THMM_Generator:

    def __init__(self, config, label_list, hierarchical_tree):
        """Constructor

        Args:
            config (dict): Model Config
            label_list (list): labels
            hierarchical_tree (dict): Hierarchical Tree
        """
        self.config = config
        self.label_list = label_list
        self.g = hierarchical_tree
        self.tasks = dict()
        self.outputs = dict()
        for label in label_list:
            self.tasks[label] = SingleTaskLayer(self.config)

    def gen_output(self, node, input, parent_hidden_state):
        """Generate Output.

        Args:
            node (str): identifier for the node.
            input (tensor): input a classification head.
            parent_hidden_state (tensor): parent hidden state.

        Returns:
            tensor: Output for a classification head.
        """

        successors = [item for item in self.g.successors(node)]

        if node != ROOT:
            if parent_hidden_state != None:
                task_input = tf.keras.layers.concatenate([input, parent_hidden_state])
                print(node, 'cls_emb + parent_hidden', task_input.shape)
            else:
                task_input = input
                print(node, 'cls_emb', task_input.shape)

            softmax_output, logits = self.tasks[node](task_input)
            self.outputs[node] = softmax_output

            if len(successors):
                for successor in successors:
                    self.gen_output(successor, input, logits)
            else:
                return None

    def get_outputs(self, input):
        """Get Outputs.

        Args:
            input (tensor): Model input

        Returns:
            list: outputs
        """
        for node in self.g.successors(ROOT):
            self.gen_output(node, input, None)

        outputs_sorted = list()
        for label in self.label_list:
            outputs_sorted.append(self.outputs[label])
        return outputs_sorted
