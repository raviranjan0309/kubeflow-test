from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import pandas as pd
from six.moves import urllib
import tensorflow as tf
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf-data-dir',
                      type=str,
                      default='/tmp/data/train.csv',
                      help='GCS path or local path of training data.')
    parser.add_argument('--tf-model-dir',
                      type=str,
                      default='/tmp/model/',
                      help='GCS path or local directory.')
    parser.add_argument('--tf-export-dir',
                      type=str,
                      default='/tmp/export/',
                      help='GCS path or local directory to export model')
    parser.add_argument('--hd1',
                        type=int,
                        default=100,
                        help='The hyperparamer value for layer one')

    parser.add_argument('--tf-train-steps',
                        type=int,
                        default=100,
                        help='The number of training steps to perform.')
    parser.add_argument('--tf-embedding-size',
                        type=int,
                        default=10,
                        help='The embedding size.')

    parser.add_argument('--tf-batch-size',
                        type=int,
                        default=1024,
                        help='The number of batch size during training')
    
    args = parser.parse_args()
    return args

COLUMNS = [
    "count_conscecutive_visits", "recency_day", "recency_week", "visit_number", "output_binary"
]

FEATURE_COLUMNS = [
    "count_conscecutive_visits", "recency_day", "recency_week", "visit_number"
]

INPUT_COLUMNS = [
  # Continuous base columns.
    tf.feature_column.numeric_column('count_conscecutive_visits'),
    tf.feature_column.numeric_column('recency_day'),
    tf.feature_column.numeric_column('recency_week'),
    tf.feature_column.numeric_column('visit_number'),
]

BATCH_SIZE = 40
num_epochs = 1
shuffle = True

# [START serving-function]
def serving_input_receiver_fn():
    """Build the serving inputs."""
    global FEATURE_COLUMNS
    inputs = {}
    for feat in FEATURE_COLUMNS:
        inputs[feat] = tf.placeholder(shape=[None], dtype=tf.float32)
    
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
# [END serving-function]


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_arguments()

    tf_config = os.environ.get('TF_CONFIG', '{}')
    tf.logging.info("TF_CONFIG %s", tf_config)
    tf_config_json = json.loads(tf_config)
    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')
    tf.logging.info("cluster=%s job_name=%s task_index=%s", cluster, job_name,
                    task_index)

    is_chief = False
    if not job_name or job_name.lower() in ["chief", "master"]:
        is_chief = True
        tf.logging.info("Will export model")
    else:
        tf.logging.info("Will not export model")
    
    #update the path
    train_file = '/tmp/data/train.csv'
    df = pd.read_csv(train_file)
    y = df["output_binary"]
    del df["output_binary"]
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=12345)

    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=X_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        num_epochs=num_epochs,
        shuffle=shuffle)
    
    test_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=X_test,
        y=y_test,
        batch_size=BATCH_SIZE,
        num_epochs=num_epochs,
        shuffle=shuffle)

    classifier = tf.estimator.DNNLinearCombinedClassifier(
              linear_feature_columns=INPUT_COLUMNS,
              dnn_feature_columns=INPUT_COLUMNS,
              dnn_hidden_units=[args.hd1, 70, 50, 25])
    

    serving_fn = serving_input_receiver_fn

    #update the path
    export_final = tf.estimator.FinalExporter(
      args.tf_export_dir, serving_input_receiver_fn=serving_input_receiver_fn)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=5)

    eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn,
                                      steps=1,
                                      throttle_secs=1,
                                      start_delay_secs=1)

    print("Train and evaluate")
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    print("Training done")

    print("Export saved model")
    #update the path
    classifier.export_savedmodel(args.tf_export_dir, serving_input_receiver_fn=serving_fn)
    print("Done exporting the model")

if __name__ == '__main__':
    tf.app.run()