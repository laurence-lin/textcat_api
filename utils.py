#-----------------------------------------------------------------------------
# Import libraries 
#-----------------------------------------------------------------------------
# Standard library imports
from pathlib import Path
import json

# Third-party imports
import pandas as pd
import spacy
from tqdm.auto import tqdm
from spacy.tokens import DocBin, Doc

#-----------------------------------------------------------------------------
# Initialize a pretrained spaCy model
#-----------------------------------------------------------------------------
nlp = spacy.load("zh_core_web_lg", disable=["tagger", "parser", "ner"])

#-----------------------------------------------------------------------------
# Utility functions
#-----------------------------------------------------------------------------
# Create a dataframe of category, counts, train/test
def train_test_counts(train_df, test_df, label_col):
    train_count = train_df.groupby(label_col).size().reset_index(name='類別數')
    train_count["資料類別"] = "訓練集"
    test_count = test_df.groupby(label_col).size().reset_index(name='類別數')
    test_count["資料類別"] = "測試集"
    df = pd.concat([train_count, test_count], ignore_index=True)
    return df

# Convert a dataframe to a list
def df2list(df, text_col, label_col):
  df.loc[ : , 'tuples'] = df.apply(lambda row: (row[text_col], row[label_col]), axis=1)
  data_list = df['tuples'].tolist()
  return data_list

# Convert a list to a spaCy DOC object and save it to disk
def make_docs(data_list, unique_labels, dest_path):
    docs = []
    for doc, label in tqdm(nlp.pipe(data_list, as_tuples=True), total = len(data_list)):
        label_dict = {label: False for label in unique_labels}
        doc.cats = label_dict
        doc.cats[label] = True
        docs.append(doc)
    doc_bin = DocBin(docs=docs)
    doc_bin.to_disk(dest_path)

# Determine the base_config file for training
def make_base_config(efficency):
    if efficency:
        base_config = Path.cwd() / "base_config_cpu_efficiency.cfg"
    else:
        base_config = Path.cwd() / "base_config_cpu_accuracy.cfg"    
    return base_config
    #if cpu and efficency:
        #base_config = Path.cwd() / "base_config_cpu_efficiency.cfg"
    #elif cpu and not efficency:
        #base_config = Path.cwd() / "base_config_cpu_accuracy.cfg"
    #elif not cpu and efficency:
        #base_config = Path.cwd() / "base_config_gpu_efficiency.cfg"
    #elif not cpu and not efficency:
        #base_config = Path.cwd() / "base_config_gpu_accuracy.cfg"


# Determine the config key for ngram_size
def make_ngram_key(efficency):
    if efficency:
        key = "components.textcat.model.ngram_size"
    else:
        key = "components.textcat.model.linear_model.ngram_size"
    return key

# Get category distribution given a spaCy DOC object 
'''
def get_cats(doc: Doc) -> Dict[str, Any]:
    return {
        "input": doc.text, 
        "output_proba": doc.cats,
        "output_max": distribution_to_max(doc.cats),
        }
'''

# convert category distribution to a tuple of category and highest confidence score
def distribution_to_max(cat_distribution):
    max_cat = ["", 0]
    for k, v in cat_distribution.items():
        if v > max_cat[1]:
            max_cat[1] = v
            max_cat[0] = k
    max_cat[1] = round(max_cat[1], 3)
    return max_cat

# get the cat with the highest confidence score
def get_cat(doc: Doc) -> str:
    max_cat = distribution_to_max(doc.cats)
    return max_cat[0]

# get the highest confidence score
def get_conf(doc: Doc) -> float:
    max_cat = distribution_to_max(doc.cats)
    return max_cat[1]

# Get the metrics of a trained model
def make_metrics(model_path):
    meta_path = model_path / "meta.json"
    if meta_path.exists():
        with open(meta_path) as json_file:
            metrics = json.load(json_file)
        performance = metrics['performance']
        f1 = performance['cats_macro_f']
        precision = performance['cats_macro_p']
        recall = performance['cats_macro_r']
        auc = performance['cats_macro_auc']
        per_cat_dict = performance['cats_f_per_type']
        metrics = {
            'f1': round(f1, 3),
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'auc': round(auc, 3),
            'per_category': per_cat_dict,
        }
    else:
        metrics = {}
    return metrics

'''
#-----------------------------------------------------------------------------
# Settings for the Redis server 
#-----------------------------------------------------------------------------
REDIS_HOST = '192.168.201.176'
REDIS_PORT = 6379
REDIS_DB = 1

# Connect to the Redis server
def redis_connect() -> redis.StrictRedis:
    try:
        pool = redis.ConnectionPool(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
        )
        client = redis.StrictRedis(connection_pool=pool)
        ping = client.ping()
        if ping is True:
            return client
    except redis.AuthenticationError:
        print("AuthenticationError")
        sys.exit(1)

client = redis_connect()
'''