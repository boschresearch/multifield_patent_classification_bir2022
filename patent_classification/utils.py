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


from sklearn.metrics import precision_score, recall_score, f1_score


def calc_metrics(true, pred, run_label=None, split_index=0):
    """
    Calculation Metrics.

    Args:
        true (numpy): true values
        pred (numpy): predicted values
        run_label (str, optional): run label. Defaults to None.
        split_index (int, optional): split index. Defaults to 0.

    Returns:
        dict: Performance Metric.
    """
    return {
        'run': run_label,
        'precision_macro': round(precision_score(true, pred, average='macro'), 3),
        'recall_macro': round(recall_score(true, pred, average='macro'), 3),
        'f1_macro': round(f1_score(true, pred, average='macro'), 3),
        'precision_micro': round(precision_score(true, pred, average='micro'), 3),
        'recall_micro': round(recall_score(true, pred, average='micro'), 3),
        'f1_micro': round(f1_score(true, pred, average='micro'), 3),
        'index': split_index
    }
