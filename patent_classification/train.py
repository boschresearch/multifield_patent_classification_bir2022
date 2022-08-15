# Copyright (c) 2021 Robert Bosch GmbH
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

import sys
import os
# project_base = "/home/ujp5kor/patent_classification/code/bir2022-neural-patent-classification"
project_base = "<project base dir>"
sys.path.append(os.path.join(project_base))
sys.path.append("<path to konvens baseline>")

import pandas as pd
import json
import logging
import argparse

from patent_classification.label_binarizer import get_multitask_output
from patent_classification.classification_utils import training
from patent_classification.data.utils import DataLoader, FeatureGeneratorCPC, PrePorcessTargetLabelCPC


def get_exp_dir_name(config, run_prefix):
    """
    Get the name of the experiment directory.

    Args:
        config (dict): Experiment configuration.
        run_prefix (dict): the prefix to abe added to the experiment directory name.

    Returns:
        tuple: exp_dir (str), exp_run_string (str)
    """

    exp_run_string = list()

    if "model" in config:
        exp_run_string.append(config["model"])

    if "embs" in config["doc_rep_params"]:
        exp_run_string.extend([emb_info["type"]
                              for emb_info in config["doc_rep_params"]["embs"]])

    if "emb_agg" in config["model_params"]:
        exp_run_string.append(config["model_params"]["emb_agg"])

    exp_dir = os.path.join(
        config["exp_dir"], "_".join(exp_run_string) + run_prefix)

    if "exp_dir_prefix" in config:
        exp_dir += "_" + config["exp_dir_prefix"]

    return exp_dir, exp_run_string


def load_and_train(config, 
                data, 
                train_ids, test_ids, dev_ids, heldout_ids,
                id_column,
                exp_dir, 
                metrics, metrics_heldout, 
                split_index, 
                mode, 
                content_fields, 
                cpc_code_filter_list=None, 
                logger=None):
    """
    This method performs two task, it loads the dataset, creates the train, dev, test and heldout data frames and train the model.

    Args:
        config (dict): the input configuration for the experiment.
        data (DataFrame): the input dataset.
        train_ids (set): the set of train ids.
        test_ids (set): the set of test ids.
        dev_ids (set): the set of dev ids.
        heldout_ids (set): the set of heldout ids.
        id_column (str): the column that uniquely identifies an instance.
        exp_dir (str): the experiment directory to save the output files.
        metrics (list): output metrics for the test set.
        metrics_heldout (list): output metrics for the heldout set.
        split_index (int): the index id in case of k-fold cross validation.
        mode (str): mode to run the experiment - debug, dev, prod
        content_fields (list): the content fields within the data frame. 
        cpc_code_filter_list (_type_, optional): _description_. Defaults to None.
        logger (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: Missing information.
    """

    
    logger.info("loading dataset ... ")

    # is their any special processing required for the target label.
    process_label = PrePorcessTargetLabelCPC(exp_dir, is_flat=False, logger=logger)

    # load the dataset
    data_loader = DataLoader(process_label, "patent_id", "labels", logger=logger)
    train, test, dev, heldout, y_train, y_test, y_dev, y_heldout, mlb, hier_tree = data_loader.load_train_test(
        data, exp_dir, train_ids, test_ids, dev_ids, heldout_ids, mode=mode, split_index=split_index)

    logger.info("train.shape: %s " % str(train.shape))
    logger.info("test.shape: %s" % str(test.shape))
    logger.info("dev.shape:  %s" % str(dev.shape))
    
    if heldout is not None:
        logger.info("train.shape:  %s" % str(heldout.shape))

    # get the feature vector
    feature_generator = FeatureGeneratorCPC(train, test, dev, heldout, exp_dir, "debug", logger=logger, id_column=id_column)
    logger.info("Model - %s" % config["model"])
    if config["model"] == "SVM" or config["model"] == "hierSVM":
        if config["doc_rep_params"]["embs"]:
            fields = config["doc_rep_params"]["embs"][0]["fields"]
            label =  config["doc_rep_params"]["embs"][0]["label_text"]
            X_train, X_test, X_dev, X_heldout = feature_generator.get_split_text(content_fields, fields, label)
            y_train_mt = y_train
            emb_info_list = None
            embedding_weights = None
        else:
            raise ValueError("Fields information missing for SVM model.")

    elif config["model"] == "TMM" or config["model"] == "THMM":

        emb_info_list = feature_generator.generate_feature_vector(config["doc_rep_params"]["embs"], content_fields)
        X_train, X_test, X_dev, X_heldout = feature_generator.X_train, feature_generator.X_test, feature_generator.X_dev, feature_generator.X_heldout
        y_train_mt = get_multitask_output(y_train)
        embedding_weights = feature_generator.embedding_weights

        for emb_info in emb_info_list:
            logger.info(json.dumps(emb_info))

        for emb in X_train:
            logger.info("X_train -- emb size: %s" % str(emb.shape))

        for emb in X_test:
            logger.info("X_test -- emb size: %s" % str(emb.shape))

        for emb in X_dev:
            logger.info("X_dev -- emb size: %s" % str(emb.shape))

        if X_heldout:
            for emb in X_heldout:
                logger.info("X_heldout -- emb size: %s" % str(emb.shape))

    # train the model
    perf_test, perf_test_heldout = training(X_train, y_train_mt,
                                            X_test, y_test,
                                            X_dev, y_dev,
                                            X_heldout, y_heldout,
                                            train_ids, test_ids, dev_ids, heldout_ids,
                                            model_config=config["model_params"],
                                            exp_dir=exp_dir,
                                            mlb=mlb,
                                            emb_info_list=emb_info_list,
                                            embedding_weights=embedding_weights,
                                            hierarchical_label_tree=hier_tree,
                                            label="_".join(sorted([emb_info["type"] for emb_info in config["doc_rep_params"]["embs"]])),
                                            split_index=split_index, 
                                            logger=logger)

    logger.info("perf_test : %s" % json.dumps(perf_test))
    logger.info("perf_test_heldout : %s" % json.dumps(perf_test_heldout))

    metrics = list()
    metrics_heldout = list()
    iter_index = 0

    perf = dict()
    perf["split"] = split_index

    if iter_index:
        perf["iter_index"] = iter_index

    perf["model"] = config.get("model")
    perf["agg-type"] = config["model_params"].get("emb_agg")
    if config["model"] in ["TMM", "THMM"]:
        if "embs" in config.get("doc_rep_params"):
            for emb_info in config["doc_rep_params"].get("embs"):
                perf[emb_info["type"]] = True

    if perf_test_heldout:
        perf_test_heldout = {**perf, **perf_test_heldout}
        metrics_heldout.append(perf_test_heldout)
        df = pd.DataFrame(metrics_heldout)
        df.to_csv(os.path.join(exp_dir, "metrics_heldout_cv.csv"), index=False)

    if perf_test:
        perf_test = {**perf, **perf_test}
        metrics.append(perf_test)
        df = pd.DataFrame(metrics)
        df.to_csv(os.path.join(exp_dir, "metrics_cv.csv"), index=False)

        metrics_avg = list()
        for run in df.run.unique().tolist():
            df_run = df[df.run == run]
            perf_avg = {
                "run": run,
                "precision_macro": round(df_run.precision_macro.mean(), 3),
                "recall_macro": round(df_run.recall_macro.mean(), 3),
                "f1_macro": round(df_run.f1_macro.mean(), 3),
                "precision_micro": round(df_run.precision_micro.mean(), 3),
                "recall_micro": round(df_run.recall_micro.mean(), 3),
                "f1_micro": round(df_run.f1_micro.mean(), 3)
            }
            metrics_avg.append({**perf, **perf_avg})
        pd.DataFrame(metrics_avg).to_csv(os.path.join(exp_dir, "metrics_avg.tsv"), index=False)


def get_logger(filename):
    """
    Get the logger object.

    Args:
        filename (str): the logger filename

    Returns:
        logger : returns a logger object.
    """
    logger = logging.getLogger('server_logger')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def run_config(config, mode):
    """
    Run the input configuration.

    Args:
        config (dict): the config file for an experiment.
        mode (str): In which mode you wish to run the current experiment.
    """
    exp_dir, exp_dir_prefix = get_exp_dir_name(config, "")

    # create experiment directory
    os.makedirs(exp_dir, exist_ok=True)

    # cerate a new logging file
    open(os.path.join(exp_dir, "logging.log"), "w").close()

    # save config
    open(os.path.join(exp_dir, "config.json"), "w").write(json.dumps(config))

    logging_filename = os.path.join(exp_dir, "logging.log")
    print(logging_filename)

    logger = get_logger(logging_filename)
    logger.info("logging_filename: %s" % logging_filename)
    logger.info("exp_dir : %s" % exp_dir)
    logger.info(json.dumps(config))

    logger.info("mode: %s " % mode)

    data = pd.read_csv(os.path.join(config["input_dir_path"], config["dataset_filename"]))
    train_ids = [int(_id.strip()) for _id in open(os.path.join(config["input_dir_path"], "train_ids.csv")).readlines()]
    test_ids = [int(_id.strip()) for _id in open(os.path.join(config["input_dir_path"], "test_ids.csv")).readlines()]
    dev_ids = [int(_id.strip()) for _id in open(os.path.join(config["input_dir_path"], "dev_ids.csv")).readlines()]

    metrics = list()
    metrics_heldout = list()

    content_fields = [
        "title", 
        "abstract", 
        "claims", 
        "description", 
        "brief-desc", 
        "fig-desc"
    ]

    load_and_train(config, 
                data, 
                train_ids, test_ids, dev_ids, None,
                "patent_id",
                exp_dir, 
                metrics, 
                metrics_heldout, 
                split_index=0,
                mode=mode,
                content_fields=content_fields, 
                cpc_code_filter_list=None, 
                logger=logger)


config = {
    "input_dir_path": "/fs/scratch/rng_cr_bcai_dl_students/r26/patent_classification/dataset/cpc_ipc_cls/uspto-50k/all-sections",
    "dataset_filename": "all_data.csv",
    "doc_rep_params": {
        "embs": [
            {
                "type": "bert-precomputed",
                "fields": [
                    "title"
                ],
                "path": "/home/ujp5kor/patent_classification/embedding/experiments_cpc/uspto-10ktest/scibert/title.csv",
                "max_len": 512,
                "label_text": False,
                "trainable": True
            },
            {
                "type": "bert-precomputed",
                "fields": [
                    "abstract"
                ],
                "path": "/home/ujp5kor/patent_classification/embedding/experiments_cpc/uspto-10ktest/scibert/abstract.csv",
                "max_len": 512,
                "label_text": False,
                "trainable": True
            }
        ]
    },
    "model": "THMM",
    "model_params": {
        "model": "THMM",
        "dense_layer_size": 256,
        "dropout_rate": 0.25,
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 64,
        "emb_agg": "sum",
        "encoder_size": 768,
        "kernel": "rbf"
    },
    "exp_dir_prefix": "thmm_t-a",
    "exp_dir": "/fs/scratch/rng_cr_bcai_dl/ujp5kor/output_dir/experiments-cpc-debug/experiments-cpc-thmm-using-fryderyks-emb"
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--mode', type=str, default="debug", required=False)
    args = parser.parse_args()
    mode = args.mode
    cfg = args.cfg
    config = json.loads(open(cfg).read())
    run_config(config, mode)


if __name__ == "__main__":
    # main method
    # main()
    
    # debug
    mode = "debug"
    run_config(config, mode)