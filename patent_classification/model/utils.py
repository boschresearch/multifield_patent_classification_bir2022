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


from networkx import DiGraph
from sklearn_hierarchical_classification.constants import ROOT

 
ROOT = "<ROOT>"


def extend_hierarchy(hierarchy, y_labs):
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
                print(lab, par_1, ROOT)
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


def get_hierarchical_tree(y):
    hierarchy_f = build_hierarchy([tj for tk in y for tj in tk])
    class_hierarchy = extend_hierarchy(hierarchy_f, y)
    g = DiGraph(class_hierarchy)
    return g


def multlabel_to_multitask(labels, mlb):
    """
    Generate a multi-task model, from a multi-label model.
    :param labels:
    :param mlb:
    :return:
    """
    pass
