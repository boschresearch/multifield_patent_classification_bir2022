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

import pickle
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from transformers import BertTokenizer, LongformerTokenizer

# from document_rep.doc_rep_utils import get_doc_rep_from_label_emb
from patent_classification import constants


class FeatureVectorizer:

    def __init__(self, filename):
        """Constructor.

        Args:
            filename (str): the filename to load a feature model.
        """
        pass

    def fit(self):
        raise NotImplementedError("This class needs to be implemented in the Child class.")

    def transform(self):
        raise NotImplementedError("This class needs to be implemented in the Child class.")


class TFIDF_Vectorizer(FeatureVectorizer):

    def __init__(self, filename=None):
        """Constructor.

        Args:
            filename (str, optional): the filename to load the TF-IDF feature vector. Defaults to None.
        """
        if filename:
            self.tf_idf = pickle.load(open(filename, "rb"))
        else:
            context_features = FeatureUnion(
                transformer_list=[
                    ('word', TfidfVectorizer(
                        strip_accents=None,
                        lowercase=True,
                        analyzer='word',
                        ngram_range=(1, 2),
                        max_df=1.0,
                        min_df=0.0,
                        binary=False,
                        use_idf=True,
                        smooth_idf=True,
                        sublinear_tf=True,
                        max_features=70000
                    )),
                    ('char', TfidfVectorizer(
                        strip_accents=None,
                        lowercase=False,
                        analyzer='char',
                        ngram_range=(3, 6),
                        max_df=1.0,
                        min_df=0.0,
                        binary=False,
                        use_idf=True,
                        smooth_idf=True,
                        sublinear_tf=True,
                        max_features=70000
                    )),
                ]
            )

            vectorizer = FeatureUnion(
                transformer_list=[('context', Pipeline(steps=[('vect', context_features)]))])
            self.tf_idf = make_pipeline(vectorizer)

    def fit(self, texts):
        """Fit the feature vector with the text

        Args:
            texts (list): the list of text.
        """
        self.tf_idf.fit(texts)

    def transform(self, texts):
        """Transform the input list of text with the trained feature vector.

        Args:
            texts (list): list of texts

        Returns:
            numpy array: the feature vector.
        """
        return tf.cast(self.tf_idf.transform(texts).toarray(), "float32")

    def save(self, filename):
        """Save the TF-IDF model.

        Args:
            filename (str): the filename to store the TF-IDF model.
        """
        pickle.dump(self.tf_idf, open(filename, "wb"))


class InputTokenizer:

    def __init__(self):
        """
        Constructor.
        """
        pass

    def fit(self, docs):
        raise NotImplementedError("The method needs to be implemented in the child class.")

    def transform(self, docs):
        raise NotImplementedError("The method needs to be implemented in the child class.")


class HuggingFaceTokenizer(InputTokenizer):

    def __init__(self, filename, max_len, emb_type):
        """
        Constructor.

        Args:
            filename (str): the directory or filename to load the Hugging Face language model.
            max_len (int): the maximum length of the text.
            emb_type (str): name of the embedding to use.
        """
        if emb_type == constants.CONTENT_EMB_LONGFORMER:
            self.tokenizer = LongformerTokenizer.from_pretrained(filename, do_lower_case=True)
        elif emb_type == constants.CONTENT_EMB_SCIBERT:
            self.tokenizer = BertTokenizer.from_pretrained(filename, do_lower_case=True)
        elif emb_type == constants.LABEL_EMB_TEXT_BERT_TRAINABLE:
            self.tokenizer = BertTokenizer.from_pretrained(filename, do_lower_case=True)
        self.max_len = max_len

    def fit(self, docs):
        """
        Fit method do not apply for HuggingFace library.

        Args:
            docs (list): texts

        Raises:
            NotImplementedError: This method is not implemented in case of Hugging Face language model. 
        """
        raise NotImplementedError("Hugging Face tokenizer do not needs a fit model")

    def transform(self, docs):
        tokenize = self.tokenizer(docs, return_tensors="tf", max_length=self.max_len, truncation=True,
                                  padding='max_length')
        return [tokenize["input_ids"], tokenize["attention_mask"]]


class PrecomputedEmbedding(InputTokenizer):

    def __init__(self, path):
        """
        Constructor.

        Args:
            path (str): a path to embedding file.
        """
        self.path = path
        self.doc_vector = dict()
        with open(self.path, "r") as fileR:
            for line in fileR:
                tokens = line.strip().split(" ")
                doc_id = int(tokens[0])
                vector = [float(item) for item in tokens[1:]]
                self.doc_vector[doc_id] = vector

    def fit(self, docs):
        raise NotImplementedError("this function is not applicable for this class.")

    def transform(self, ids):
        """Transform.

        Args:
            ids (list): list of ids in the feature vector.

        Returns:
            tensor: feature vector.
        """
        vectors = list()
        for _id in ids:
            vectors.append(self.doc_vector[_id])
        return tf.cast(vectors, "float32")