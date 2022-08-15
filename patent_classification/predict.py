import argparse
import os
import sys
import logging
import json
import pandas as pd

project_base = "/home/ujp5kor/patent_classification/code/bir2022-neural-patent-classification"
sys.path.append(os.path.join(project_base))
sys.path.append("/home/ujp5kor/patent_classification/code/baseline_konvens")


from patent_classification.utils import calc_metrics
from patent_classification.classification_utils import define_classifier
from patent_classification.data.utils import DataLoader, FeatureGeneratorCPC, PrePorcessTargetLabelCPC
from patent_classification.label_binarizer import get_mlb_output, get_multitask_output


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


def test_thmm(config, exp_dir):
    """Test

    Args:
        config (dict): experiment configuration
        exp_dir (str): path to the experiment directory

    Raises:
        FileNotFoundError: the trained weights not found.
    """

    logger = get_logger(os.path.join(exp_dir, "test_log.log"))

    # load model config
    model_config = config["model_params"]

    # load dataset
    data = pd.read_csv(os.path.join(config["input_dir_path"], config["dataset_filename"]))
    train_ids = [int(_id.strip()) for _id in open(os.path.join(config["input_dir_path"], "train_ids.csv")).readlines()]
    test_ids = [int(_id.strip()) for _id in open(os.path.join(config["input_dir_path"], "test_ids.csv")).readlines()]
    dev_ids = [int(_id.strip()) for _id in open(os.path.join(config["input_dir_path"], "dev_ids.csv")).readlines()]
    heldout_ids = None
    mode = "debug"
    split_index = 0

    # Is their any special processing required for the target label?
    process_label = PrePorcessTargetLabelCPC(exp_dir, is_flat=False, logger=logger)

    data_loader = DataLoader(process_label, "patent_id", "labels", logger=logger)
    train, test, dev, heldout, y_train, y_test, y_dev, y_heldout, mlb, hier_tree = data_loader.load_train_test(data, exp_dir, train_ids, test_ids, dev_ids, heldout_ids, mode=mode, split_index=split_index)

    content_fields = [
        "title", 
        "abstract", 
        "claims", 
        "description", 
        "brief-desc", 
        "fig-desc"
    ]
    
    # feature generator object
    feature_generator = FeatureGeneratorCPC(train, test, dev, heldout, exp_dir, "debug", logger=logger, id_column="patent_id")

    # embedding info list
    emb_info_list = feature_generator.generate_feature_vector(config["doc_rep_params"]["embs"], content_fields)
    
    # loading the features
    X_train, X_test, X_dev, X_heldout = feature_generator.X_train, feature_generator.X_test, feature_generator.X_dev, feature_generator.X_heldout
    
    embedding_weights = feature_generator.embedding_weights

    logger.info("model initialization starts ...")
    model = define_classifier(model_config, mlb.classes_, emb_info_list, embedding_weights, hier_tree, logger)
    logger.info("... model initialized")

    if os.path.exists(os.path.join(exp_dir, "best-model.h5")):
        logger.info("model weights found in path: %s" % os.path.join(exp_dir, "best-model.h5"))
        model.load_weights(os.path.join(exp_dir, "best-model.h5"))
    else:
        logger.info("model weights not found in path: %s" % os.path.join(exp_dir, "best-model.h5"))
        raise FileNotFoundError("Error in loading weights.")

    y_test_pred = model.predict(X_test)

    y_test_pred = get_mlb_output(y_test_pred)

    y_test_pred = y_test_pred >= 0.5

    print("y_test_pred.shape: ", y_test_pred.shape) 

    print("y_test.shape: ", y_test.shape) 

    print(calc_metrics(y_test, y_test_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--exp_dir', type=str, required=True)
    args = parser.parse_args()
    exp_dir = args.exp_dir
    cfg = args.cfg

    config = json.load(open(cfg))

    test_thmm(config, exp_dir)

