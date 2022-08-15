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
import re
import pickle
import logging
from sklearn.preprocessing import MultiLabelBinarizer
# from tensorflow.python.framework.constant_op import constant
# from tensorflow.python.ops.gen_math_ops import cos
# from patent_classification.features.vectorizer import CNNTokenizer
# from patent_classification.features.vectorizer import HuggingFaceTokenizer
from patent_classification.features.vectorizer import PrecomputedEmbedding
from patent_classification.features.vectorizer import TFIDF_Vectorizer
from patent_classification.model.utils import get_hierarchical_tree
from patent_classification import constants


GENERIC_FEATURES = [constants.CONTENT_EMB_SCIBERT, 
                    constants.CONTENT_EMB_LONGFORMER, 
                    constants.CONTENT_EMB_TFIDF, 
                    constants.CONTENT_EMB_CNN,
                    constants.CONTENT_EMB_SCIBERT_PRECOMPUTED
                ]


def get_xml_rem_text(text):
    """
    Remove the XML text.

    Args:
        text (str): text that needs to be filtered.

    Returns:
        _type_: _description_
    """
    return re.sub('<[^<]+>', "", text)


def get_str_list(numbers):
    """Convert the number to str list.

    Args:
        numbers (list): 

    Returns:
        list : list of strings.
    """
    return [str(number) for number in numbers]


class PreprocessTargetLabel:
    """
    Preprocess the Target Labels.
    """

    def __init__(self, logger):
        """
        Constructor.

        Args:
            logger (logger object): the logger object to save the dataset logs.
        """
        self.logger = logger

    def pre_process_label():
        """
        Preprocess Label.

        Raises:
            NotImplementedError: Raise the NotImplementedError in case method is not implemented.
        """
        raise NotImplementedError("this method needs to implemented in the child class.")


class PrePorcessTargetLabelCPC(PreprocessTargetLabel):

    def __init__(self, exp_dir, is_flat=True, logger=None):
        """
        Class constructor.

        Args:
            exp_dir (str): experiment directory
            is_flat (bool, optional): is the target label taxonomy flat. Defaults to True.
            logger (logger, optional): the logger object to save the output. Defaults to None.
        """
        super().__init__(logger)
        self.is_flat = is_flat
        self.exp_dir = exp_dir
        self.logger = logger

    def pre_process_label(self, y_train, y_test, y_dev, y_heldout=None):
        """
        Preprocess target labels.

        Args:
            y_train (list): list of target labels train.
            y_test (list): list of target labels test.
            y_dev (list): list of target labels dev.
            y_heldout (list, optional): list of target labels heldout. 
            This is set to None in case the heldout set is not present. Defaults to None.

        Returns:
            tuple: binarize output, multi-label binarizer object, and a hierarchical data structure.
        """

        y_train = [eval(item) for item in y_train]
        y_test = [eval(item) for item in y_test]
        y_dev = [eval(item) for item in y_dev]

        if y_heldout is not None:
            y_heldout = [eval(item) for item in y_heldout]

        # Generate hierarchical tree
        hierarchical_tree =  None
        
        if not self.is_flat:
            hierarchical_tree = get_hierarchical_tree(y_train)
            y_train = [list(set(sum([[j[0], j[:3], j] for j in tk], []))) for tk in y_train]
            y_test = [list(set(sum([[j[0], j[:3], j] for j in tk], []))) for tk in y_test]
            y_dev = [list(set(sum([[j[0], j[:3], j] for j in tk], []))) for tk in y_dev]
            pickle.dump(hierarchical_tree, open(os.path.join(self.exp_dir, "label_hier.pkl"), "wb"))


        # Train the multilabel binarizer on test dataset.
        mlb = MultiLabelBinarizer()
        mlb.fit(y_train)
        pickle.dump(mlb, open(os.path.join(self.exp_dir, "mlb.pkl"), "wb"))


        logging.info("classes in mlb : %s" % "|".join(mlb.classes_))

        y_train = mlb.transform(y_train)
        y_test = mlb.transform(y_test)
        y_dev = mlb.transform(y_dev)
        logging.info("y_train shape: %s" % str(y_train.shape))
        logging.info("y_test shape: %s " % str(y_test.shape))
        logging.info("y_dev shape: %s " % str(y_dev.shape))

        if y_heldout:
            y_heldout = mlb.transform(y_heldout)
            logging.info("y_test_heldout shape: %s" % str(y_heldout.shape))

        return y_train, y_test, y_dev, y_heldout, mlb, hierarchical_tree


class DataLoader:

    def __init__(self, preprocess_label, unique_id_label, target_label_column, logger):
        """Constructor.

        Args:
            preprocess_label (_type_): 
            unique_id_label (_type_): the unique ide label which identifies an instance uniquely.
            target_label_column (str): target label column.
            logger (_type_): logger object.
        """
        self.preprocess_label = preprocess_label
        self.target_label_column = target_label_column
        self.unique_id_label = unique_id_label
        self.hierarchical_tree = None
        self.logger = logger

    @staticmethod
    def get_reduced_dataset(train, test, dev, heldout):
        """When the experiment mode is ``debug'', reduce the dataset size.

        Args:
            train (list): list containing traget values for train dataset
            test (list): list containing traget values for test dataset
            dev (list): list containing traget values for dev dataset
            heldout (list): list containing traget values for heldout dataset

        Returns:
            tuple : binarized vector values
        """
        train = train[:10]
        test = test[:10]
        dev = dev[:10]
        if heldout is not None:
            heldout = heldout[:10]
        return train, test, dev, heldout

    def load_train_test(self, data, exp_dir, train_ids, test_ids, dev_ids, heldout_ids=None, mode="debug", split_index=0):
        """Load dataset.

        Args:
            data (DataFrame): dataset
            exp_dir (str): experiment directory path
            train_ids (set): train ids
            test_ids (set): test ids
            dev_ids (set): dev ids
            heldout_ids (set, optional): heldout ids. Defaults to None.
            mode (str, optional): the mode of the experiment. Defaults to "debug".
            split_index (int, optional): split index in case of cross-validation. Defaults to 0.

        Returns:
            tuple: DataFrames and target columns.
        """

        train = data[data[self.unique_id_label].isin(set(train_ids))]
        test = data[data[self.unique_id_label].isin(set(test_ids))]
        dev = data[data[self.unique_id_label].isin(set(dev_ids))]

        train = train.sort_values(by=self.unique_id_label, ascending=True)
        test = test.sort_values(by=self.unique_id_label, ascending=True)
        dev = dev.sort_values(by=self.unique_id_label, ascending=True)
        
        if heldout_ids:
            heldout = data[data[self.unique_id_label].isin(set(heldout_ids))]
            heldout = heldout.sort_values(by=self.unique_id_label, ascending=True)
            self.logger.info("heldout.shape - %s " % str(heldout.shape))
        else:
            heldout = None

        if mode == "debug":
            train, test, dev, heldout = DataLoader.get_reduced_dataset(train, test, dev, heldout)

        print(train.shape)
        open(os.path.join(exp_dir, "ids_train_%s.txt" % split_index), "w").write("\n".join(get_str_list(train[self.unique_id_label].tolist())))
        open(os.path.join(exp_dir, "ids_test_%s.txt" % split_index), "w").write("\n".join(get_str_list(test[self.unique_id_label].tolist())))
        open(os.path.join(exp_dir, "ids_dev_%s.txt" % split_index), "w").write("\n".join(get_str_list(dev[self.unique_id_label].tolist())))

        if heldout_ids:
            open(os.path.join(exp_dir, "ids_heldout_%s.txt" % split_index), "w").write("\n".join(get_str_list(heldout[self.unique_id_label].tolist())))
            self.logger.info("data_utils heldout ids : %s " % str(heldout[self.unique_id_label].tolist()[:10]) )

        self.logger.info("data_utils train ids : %s " % str(train[self.unique_id_label].tolist()[:10]) )
        self.logger.info("data_utils test ids : %s " % str(test[self.unique_id_label].tolist()[:10]) )
        self.logger.info("data_utils dev ids : %s " % str(dev[self.unique_id_label].tolist()[:10]) )
    
        self.logger.info("train len: %s " % len(train))
        self.logger.info("test len: %s " % len(test))
        self.logger.info("dev len: %s " % len(dev))

        y_train = train[self.target_label_column].tolist()
        y_test = test[self.target_label_column].tolist()
        y_dev = dev[self.target_label_column].tolist()

        # preprocess_label = PrePorcessTargetLabelPatentLandscape(exp_dir, True)
        if heldout_ids:
            y_test_heldout = heldout[self.target_label_column].tolist()
            y_train, y_test, y_dev, y_heldout, mlb, hier_tree = self.preprocess_label.pre_process_label(y_train, y_test, y_dev, y_test_heldout)
        else:
            y_train, y_test, y_dev, y_heldout, mlb, hier_tree = self.preprocess_label.pre_process_label(y_train, y_test, y_dev, None)

        return train, test, dev, heldout, y_train, y_test, y_dev, y_heldout, mlb, hier_tree


class FeatureGenerator:

    def __init__(self, train, test, dev, heldout=None, exp_dir=None, mode="debug", logger=None, id_column=None):
        """Feature Generator.

        Args:
            train (DataFrame): train dataset
            test (DataFrame): test dataset
            dev (DataFrame): dev dataset
            heldout (DataFrame, optional): heldout dataset. Defaults to None.
            exp_dir (str, optional): experiment directory path. Defaults to None.
            mode (str, optional): experiment mode. Defaults to "debug".
            logger (logger, optional): logger object. Defaults to None.
            id_column (str, optional): unique column. Defaults to None.
        """

        self.train = train
        self.test = test
        self.dev = dev
        self.heldout = heldout

        self.X_train = list()
        self.X_dev = list()
        self.X_test = list()
        self.X_heldout = list()

        self.exp_dir = exp_dir
        self.mode = mode
        
        self.embedding_weights = None
        self.logger = logger
        self.id_column = id_column


    def generate_generic_features(self, emb_info, X_train_text, X_test_text, X_dev_text, X_heldout_text):
        """Generate Generic Features which apply across datasets.

        Args:
            emb_info (dict): information on a certain embedding
            X_train_text (list): list of train text
            X_test_text (list): list of test text
            X_dev_text (list): list of dev text
            X_heldout_text (list): list of heldout text

        Raises:
            Exception: raise Exception in case of a unknown embedding type.
        """

        if emb_info["type"] == constants.CONTENT_EMB_TFIDF:
            tfidf = TFIDF_Vectorizer()
            tfidf.fit(X_train_text)
            tfidf.save(os.path.join(self.exp_dir, "tf_idf.pkl"))

            self.X_train.append(tfidf.transform(X_train_text))
            self.X_test.append(tfidf.transform(X_test_text))
            self.X_dev.append(tfidf.transform(X_dev_text))

            if self.heldout is not None:
                self.X_heldout.append(tfidf.transform(X_heldout_text))

        elif emb_info["type"] == constants.CONTENT_EMB_SCIBERT_PRECOMPUTED:
            precomputed_embedding = PrecomputedEmbedding(emb_info["path"])
            self.X_train.append(precomputed_embedding.transform(self.train[self.id_column]))
            self.X_test.append(precomputed_embedding.transform(self.test[self.id_column]))
            self.X_dev.append(precomputed_embedding.transform(self.dev[self.id_column]))
            if self.heldout is not None:
                self.X_heldout.append(precomputed_embedding.transform(self.heldout[self.id_column]))

        else:
            raise Exception("unknown emb type : %s " % emb_info["type"])

    def get_split_text(self, content_fields, sel_fields, label=False):
        """
        Get text for the document ids.

        Args:
            content_fields (list): content field list
            sel_fields (list): selected field list
            label (bool, optional): Should label text be added to the train text. Defaults to False.

        Returns:
            tuple: multiple lists, one corresponding to each of the train, test, dev and heldout splits.
        """

        train_documents = FeatureGenerator.get_field_documents(content_fields, self.train)
        test_documents = FeatureGenerator.get_field_documents(content_fields, self.test)
        dev_documents = FeatureGenerator.get_field_documents(content_fields, self.dev)
        if self.heldout is not None:
            heldout_documents = FeatureGenerator.get_field_documents(content_fields, self.heldout)

        X_train_text = FeatureGenerator.get_text(train_documents, sel_fields)
        X_test_text = FeatureGenerator.get_text(test_documents, sel_fields)
        X_dev_text = FeatureGenerator.get_text(dev_documents, sel_fields)
        X_heldout_text = None
        if self.heldout is not None:
            X_heldout_text = FeatureGenerator.get_text(heldout_documents, sel_fields)

        if label:
            X_train_text, X_test_text, X_dev_text, X_heldout_text = self.append_additional_text(X_train_text, X_test_text, X_dev_text, X_heldout_text)
        return X_train_text, X_test_text, X_dev_text, X_heldout_text

    def generate_feature_vector(self, emb_info_list, fields):
        """
        Generate feature vector.

        Args:
            emb_info_list (list): Embedding info list
            fields (list): fields for which text should be included

        Returns:
            list: Containing multiple feature matrix, one corresponding to each embedding type.
        """
        
        train_documents = FeatureGenerator.get_field_documents(fields, self.train)
        test_documents = FeatureGenerator.get_field_documents(fields, self.test)
        dev_documents = FeatureGenerator.get_field_documents(fields, self.dev)
        if self.heldout is not None:
            heldout_documents = FeatureGenerator.get_field_documents(fields, self.heldout)

        emb_info_list_return = list()
        for emb_info in emb_info_list:
            self.logger.info("generating embedding for : %s " % emb_info["type"])

            X_train_text = X_test_text = X_dev_text = X_heldout_text = None

            if emb_info.get("fields"):
                X_train_text = FeatureGenerator.get_text(train_documents, emb_info["fields"])
                X_test_text = FeatureGenerator.get_text(test_documents, emb_info["fields"])
                X_dev_text = FeatureGenerator.get_text(dev_documents, emb_info["fields"])
                if self.heldout is not None:
                    X_heldout_text = FeatureGenerator.get_text(heldout_documents, emb_info["fields"])

            if emb_info["type"] in GENERIC_FEATURES:
                self.generate_generic_features(emb_info, X_train_text, X_test_text, X_dev_text, X_heldout_text)
            else:
                self.generate_specific_features(emb_info, X_train_text, X_test_text, X_dev_text, X_heldout_text)

            self.logger.info("embedding type: %s and embedding size: %s " % (emb_info["type"], self.X_train[-1].shape[1]))
            emb_info_list_return.append({**emb_info, **{"emb_size": self.X_train[-1].shape[1]}})
        return emb_info_list_return

    @staticmethod
    def get_text(documents, fields):
        """
        Get Text.

        Args:
            documents (list): list of document, where each document is a dictionary
            fields (list): list of fields

        Returns:
            list: list of texts
        """
        texts = ["" for index in range(len(documents))]
        for field in fields:
            for document_index, document in enumerate(documents):
                texts[document_index] += " " + document[field]
        return texts

    @staticmethod
    def get_field_documents(fields, df):
        """Get field documents.

        Args:
            fields (list): list containing fields
            df (DataFrame): DataFrame field

        Returns:
            list: list of document dict.
        """
        documents = list()
        for index, field in enumerate(fields):
            for row_index, text in enumerate(df[field].tolist()):
                
                if type(text) is not str:
                    text = ""
                else:
                    text = get_xml_rem_text(text.lower())

                if index == 0:
                    documents.append({field: text})
                else:
                    documents[row_index][field] = text
        return documents

    def append_additional_text(self, X_train_text, X_test_text, X_train_dev, X_heldout_text):
        raise NotImplementedError


class FeatureGeneratorCPC(FeatureGenerator):

    def __init__(self, train, test, dev, heldout, exp_dir, mode, logger=None, id_column=None):
        super().__init__(train, test, dev, heldout=heldout, exp_dir=exp_dir, mode=mode, logger=logger, id_column=id_column)


