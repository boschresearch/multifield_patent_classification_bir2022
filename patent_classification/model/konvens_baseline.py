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


import os
import pandas as pd
import numpy as np
from patent_classification.utils import calc_metrics

from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


def extend_hierarchy(hierarchy, y_labs):
    """
    Extend Hierarchy.

    Args:
        hierarchy (dict): hierarchy for a taxonomy.
        y_labs (list): list of labels

    Returns:
        dict: Hierarchical taxonomy tree.
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


def build_hierarchy(labels):
    """
    Generate a hierarchical tree for a label.

    Args:
        labels (list): list of labels

    Returns:
        dict: hierarchical taxonomy
    """
    hierarchy = {ROOT: []}
    for label in labels:
        par_1 = label[0]
        par_2 = label[:3]
        child = label[:]

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


def build_feature_extractor():
    """Initialize a TF-IDF feature vector object.

    Returns:
        TF-IDF feature generator: _description_
    """
    context_features = FeatureUnion(
        transformer_list=[
            ('word', TfidfVectorizer(
                strip_accents=None,
                lowercase=True,
                analyzer='word',
                ngram_range=(1, 1),
                max_df=1.0,
                min_df=0.0,
                binary=False,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,
                max_features=70000,
                stop_words='english'
            )),
        ]
    )
    features = FeatureUnion(
        transformer_list=[
            ('context', Pipeline(
                steps=[('vect', context_features)]
            )),
        ]
    )
    return features


def train_hier_svm(X_train, y_train,
        X_dev, y_dev, 
        X_test, y_test, 
        X_heldout, y_heldout, 
        class_hierarchy, 
        mlb,
        exp_dir, 
        decision_function_threshold=None):
    """Train Hierarchical SVM.

    Args:
        X_train (list): list containing texts for train.
        y_train (numpy): binarized labels y for train
        X_dev (_type_): list containing texts for dev.
        y_dev (_type_): binarized labels y for dev
        X_test (_type_): list containing texts for test.
        y_test (_type_): binarized labels y for test
        X_heldout (_type_): list containing texts for heldout.
        y_heldout (_type_): binarized labels y for heldout
        class_hierarchy (dict): hierarchical taxonomy tree
        mlb (MultiLabelBinarizer): Binarizer for converting the labels to binary vector.
        exp_dir (str): Path to experiment directory
        decision_function_threshold (float, optional): Decision function threshold. Defaults to None.

    Returns:
        dict: contains the performance metrics.
    """


    context_features = FeatureUnion(
        transformer_list=[
            ('word', TfidfVectorizer(
                strip_accents=None,
                lowercase=True,
                analyzer='word',
                ngram_range=(1, 1),
                max_df=1.0,
                min_df=0.0,
                binary=False,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,
                max_features=70000,
                stop_words='english'
            )),
        ]
    )
    features = FeatureUnion(
        transformer_list=[
            ('context', Pipeline(
                steps=[('vect', context_features)]
            )),
        ]
    )

    bclf = OneVsRestClassifier(LinearSVC())
    base_estimator = make_pipeline(features, bclf)
    clf = HierarchicalClassifier(
        base_estimator=base_estimator,
        class_hierarchy=class_hierarchy,
        algorithm="lcn",
        training_strategy="siblings",
        preprocessing=True,
        mlb=mlb,
        use_decision_function=True)

    clf.fit(X_train, y_train)

    if decision_function_threshold is None:
        metrics_dev = list()
        thresholds = []
        decision_function_threshold = -1.0
        for index in range(1, 20):
            thresholds.append(decision_function_threshold)
            decision_function_threshold += 0.1
        for decision_function_threshold in thresholds:
            y_dev_pred = clf.predict_proba(X_dev)
            y_dev_pred = y_dev_pred > decision_function_threshold
            perf = calc_metrics(y_dev, y_dev_pred, 'svc+tf-idf')
            perf["threshold"] = decision_function_threshold
            metrics_dev.append(perf)
        df_metrics_dev = pd.DataFrame(metrics_dev)
        decision_function_threshold = df_metrics_dev.threshold.iloc[df_metrics_dev.f1_macro.idxmax()]

    pred_test = clf.predict_proba(X_test)
    df_test = pd.DataFrame(pred_test)
    df_test.columns = mlb.classes_
    df_test.to_csv(os.path.join(exp_dir, "pred_proba_test.csv"), index=False)

    pred_test[np.where(pred_test == 0)] = -10
    pred_test = pred_test > decision_function_threshold

    df = pd.DataFrame(y_test)
    df.columns = mlb.classes_
    df.to_csv(os.path.join(exp_dir, "true_values_test.csv"), index=False)


    perf_heldout = None
    if X_heldout is not None:
        df = pd.DataFrame(y_heldout)
        df.columns = mlb.classes_
        df.to_csv(os.path.join(exp_dir, "true_values_heldout.csv"), index=False)

        pred_heldout = clf.predict_proba(X_heldout)
        df_heldout = pd.DataFrame(pred_heldout)
        df_heldout.columns = mlb.classes_
        df_heldout.to_csv(os.path.join(exp_dir, "pred_proba_heldout.csv"), index=False)

        pred_heldout[np.where(pred_heldout == 0)] = -10
        pred_heldout = pred_heldout > decision_function_threshold
        perf_heldout = calc_metrics(y_heldout, pred_heldout)

    return calc_metrics(y_test, pred_test), perf_heldout

    
