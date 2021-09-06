#-----------------------------------------------------------------------------
# Import libraries 
#-----------------------------------------------------------------------------
# Standard library imports
from pathlib import Path
import subprocess
import sys
import time
import logging
#import uuid
import sys
#import pickle

# Third-party imports
from fastapi import FastAPI, BackgroundTasks
import uvicorn
#from fastapi.encoders import jsonable_encoder
#from fastapi.responses import JSONResponse
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import mlflow
import mlflow.spacy
#from mlflow.models.signature import infer_signature
import plotly.express as px
from sqlitedict import SqliteDict
import requests
#import redis

# Local application imports
import schemas
import utils

#-----------------------------------------------------------------------------
# Initialize the FastAPI app
#-----------------------------------------------------------------------------
app = FastAPI(
  title = "文本分類API",
  version= "v2.1 2021.07 HamastarAI",
  description = '''用於文本分類建模與預測。端點說明：
  1. /create: 建立模型訓練任務，回傳訓練代號
  2. /predict/{task_id}: 使用訓練代號預測資料
  '''
)

#-----------------------------------------------------------------------------
# Variables for the dataset
#-----------------------------------------------------------------------------
#dataset column matching 
TEXT_COL, LABEL_COL = "text", "label"

#-----------------------------------------------------------------------------
# The MLflow server where tracking is saved
#-----------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://192.168.201.176:443"

#-----------------------------------------------------------------------------
# SQLiteDict database for training results
#-----------------------------------------------------------------------------
TRAINING_RESULTS_DB = './training_results_db.sqlite'
PLOTS_DB = './plots_db.sqlite'

#-----------------------------------------------------------------------------
# API endpoint for training results
#-----------------------------------------------------------------------------
BASE_URL = 'https://aihub.hamastar.com.tw/api/model/feedback/'

#-----------------------------------------------------------------------------
# Background functions
#-----------------------------------------------------------------------------
# The training function
def train_model(task: schemas.TrainingRequestModel):
    #paths
    task_str = str(task.task_id)
    task_folder = Path.cwd() / 'projects' / task_str 
    data_folder = task_folder / 'data'
    data_folder.mkdir(parents=True, exist_ok=True)
    train_path = data_folder / 'train.spacy'
    dev_path = data_folder / 'dev.spacy'
    config_path = task_folder / 'config.cfg' 
    output_folder = task_folder / 'output'
    model_folder = output_folder / 'model-best'
    log_path = Path.cwd() / 'logging.txt'
    #training_response_path = task_folder / 'training_response.pickle'
    logging.basicConfig(filename=log_path, filemode='a', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logging.info(f"Received a training request: {task_str}")

    #response 
    response = schemas.TrainingResponseModel()
    training_response_data = [{}] * 5 # for five tabs
    
    #parameters
    base_config_path = utils.make_base_config(task.efficiency)
    ngram_config_key = utils.make_ngram_key(task.efficiency)
    ngram_size = task.ngram_size

    if base_config_path:
        #loading dataset    
        data = ({TEXT_COL: item.text, LABEL_COL: item.label} for item in task.dataset)
        df = pd.DataFrame(data)

        #filtering dataset    
        df.drop_duplicates(subset=TEXT_COL, inplace=True)
        df = df.astype(str)
        len_mask = df[TEXT_COL].str.len() > task.min_len # set minimum text length 
        filtered_df = df.loc[len_mask]    

        #finalized dataset        
        unique_labels = filtered_df[LABEL_COL].unique().tolist()
        labels_str = " | ".join(unique_labels)
        dataset_size = sys.getsizeof(filtered_df) 
        train_df, dev_df = train_test_split(filtered_df, test_size=task.test_size, random_state=1, stratify=filtered_df[LABEL_COL])
        training_samples, testing_samples = train_df.shape[0], dev_df.shape[0]

        #connecting to MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"專案{task_str}")
        experiment = mlflow.get_experiment_by_name(f"專案{task_str}")
        experiment_id = experiment.experiment_id

        with mlflow.start_run():    
            run = mlflow.active_run()
            run_id = run.info.run_id
            plot_folder = f"{experiment_id}/{run_id}/artifacts/plots/"

            #data transformation    
            train_list = utils.df2list(train_df, TEXT_COL, LABEL_COL)
            dev_list = utils.df2list(dev_df, TEXT_COL, LABEL_COL)
            logging.info("Starting to convert the dataset to spaCy format...")
            training_start = time.time()
            utils.make_docs(train_list, unique_labels, train_path)
            logging.info("The training data have been successfully converted and saved.")
            utils.make_docs(dev_list, unique_labels, dev_path)
            logging.info("The testing data have been successfully converted and saved.")
           
            #model training
            config_command = f"python -m spacy init fill-config {base_config_path} {config_path}"  # initialize spacy project
            config_command_list = config_command.split()
            config_command_output = subprocess.run(config_command_list, capture_output=True)
            logging.info(config_command_output.stdout.decode("utf-8"))
            temp_message = "Starting to train models..."
            print(temp_message)
            logging.info(temp_message)
            # train with spacy settings
            train_command = f"python -m spacy train {config_path} --output {output_folder} --paths.train {train_path} --paths.dev {dev_path} --{ngram_config_key} {ngram_size} --verbose"
            train_command_list = train_command.split()
            train_command_output = subprocess.run(train_command_list, capture_output=True)
            logging.info(train_command_output.stdout.decode("utf-8"))

            #tracking params with mlflow
            mlflow.log_param("task_id", task_str)
            mlflow.log_param("test_size", str(task.test_size))
            mlflow.log_param("efficiency", str(task.efficiency))
            mlflow.log_param("dataset_size", str(dataset_size))
            mlflow.log_param("training_samples", str(training_samples))
            mlflow.log_param("testing_samples", str(testing_samples))
            mlflow.log_param("labels", labels_str)
            mlflow.log_param("ngram_size", str(ngram_size))
            mlflow.log_param("min_len", str(task.min_len))

            #adding params to response
            params_value = schemas.ParamsValue(
                task_id=task_str,
                test_size=str(task.test_size),
                efficiency=str(task.efficiency),
                dataset_size=str(dataset_size),
                training_samples=str(training_samples),
                testing_samples=str(testing_samples),
                labels=labels_str,
                ngram_size=str(ngram_size), 
                min_len=str(task.min_len),
                #cpu=str(task.cpu),
            )
            re_params = schemas.Params(value=params_value)
            tab1 = re_params
            training_response_data[0] = tab1
            logging.info("The training parameters have been logged.")

            #saving a plot of class counts to the artifact folder 
            artifact_folder = task_folder / 'artifact'
            artifact_folder.mkdir(parents=True, exist_ok=True)
            counts = utils.train_test_counts(train_df, dev_df, LABEL_COL)
            fig = px.bar(counts, 
                        x=LABEL_COL, 
                        y="類別數",
                        color="資料類別",
                        title="類別分布圖",
                        barmode="group",
                        height=800,
                        )
            class_counts_filename = "class_counts.html"
            class_counts_path = artifact_folder / class_counts_filename
            fig.write_html(str(class_counts_path)) #save to fastAPI server
            mlflow.log_artifact(str(class_counts_path), artifact_path='plots') #save to MLflow server
            temp_name = plot_folder + class_counts_filename
            re_plot_count = schemas.PlotCount(value=temp_name)
            tab3 = re_plot_count
            training_response_data[2] = tab3
            logging.info("A plot of class counts has been saved.")
            temp_html = fig.to_html()
            with SqliteDict(PLOTS_DB, autocommit=True) as mydict:
                mydict[temp_name] = temp_html

            #checking if the training is successful
            if train_command_output.returncode == 0:
                training_end = time.time()
                training_time = training_end - training_start
                #re_params_value.training_time = str(training_time)
                result_message = "The model has been successfully trained."
                print(result_message)
                logging.info(result_message)
                training_response_status = True

                #saving a plot of heatmap per cat to the artifact folder                
                metrics = utils.make_metrics(model_folder)
                per_cat_dict = metrics['per_category']
                per_cat_df = pd.DataFrame(per_cat_dict)
                fig = px.imshow(per_cat_df,
                                x=per_cat_df.columns.tolist(),
                                y="精準率 召回率 F1值".split(),
                                labels=dict(x="類別名稱", y="評估指標", color="數值大小"),
                                )
                heatmap_per_cat_filename = "metrics_per_category.html"
                heatmap_per_cat_path = artifact_folder / heatmap_per_cat_filename
                fig.write_html(str(heatmap_per_cat_path)) #save to fastAPI server
                mlflow.log_artifact(str(heatmap_per_cat_path), artifact_path='plots') #save to MLflow server
                temp_name = plot_folder + heatmap_per_cat_filename
                re_plot_per_cat = schemas.PlotPerCat(value=temp_name)
                tab4 = re_plot_per_cat
                training_response_data[3] = tab4
                logging.info("A plot of model performance per category has been saved.")
                temp_html = fig.to_html()
                with SqliteDict(PLOTS_DB, autocommit=True) as mydict:
                    mydict[temp_name] = temp_html

                #saving a plot of confusion matrix to the artifact folder 
                trained_nlp = spacy.load(model_folder)
                y_true = [tup[1] for tup in dev_list]
                dev_texts = (tup[0] for tup in dev_list)
                y_pred = []
                for doc in trained_nlp.pipe(dev_texts):
                    y_pred.append(utils.get_cat(doc))
                cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
                accuracy = accuracy_score(y_true, y_pred)
                fig = px.imshow(cm,
                                x=unique_labels,
                                y=unique_labels,
                                labels=dict(x="預測類別", y="實際類別", color="數值大小"),
                                )
                cm_filename = "confusion_matrix.html"
                cm_path = artifact_folder / cm_filename
                fig.write_html(str(cm_path)) #save to fastAPI server
                mlflow.log_artifact(str(cm_path), artifact_path='plots') #save to MLflow server
                temp_name = plot_folder + cm_filename
                re_plot_confusion_matrix = schemas.PlotConfusionMatrix(value=temp_name)
                tab5 = re_plot_confusion_matrix
                training_response_data[4] = tab5
                logging.info("A plot of confusion matrix has been saved.")
                temp_html = fig.to_html()
                with SqliteDict(PLOTS_DB, autocommit=True) as mydict:
                    mydict[temp_name] = temp_html

                #tracking metrics with mlflow 
                mlflow.log_metric("training_time", training_time)
                mlflow.log_metric("F1", metrics["f1"])
                mlflow.log_metric("precision", metrics["precision"])
                mlflow.log_metric("recall", metrics["recall"])
                mlflow.log_metric("AUC", metrics["auc"])
                mlflow.log_metric("accuracy", round(accuracy, 3))
                metrics_value = schemas.MetricsValue(
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    F1=metrics["f1"],
                    AUC=metrics["auc"],
                    accuracy=round(accuracy, 3),
                    training_time=training_time
                )

                #addinng metrics to response                
                re_metrics = schemas.Metrics(value=metrics_value)
                tab2 = re_metrics
                training_response_data[1] = tab2
                logging.info("The testing metrics have been logged.")

                #logging the trained model with signature
                ''' 
                input_example = {"text": "文章內容"}
                train_schema = dev_df[TEXT_COL]
                train_texts = train_schema.tolist()
                predictions = [{"cat": utils.get_cat(doc), "conf": utils.get_conf(doc)} for doc in trained_nlp.pipe(train_texts)]
                prediction_schema = pd.DataFrame(predictions)
                signature = infer_signature(train_schema, prediction_schema)
                mlflow.spacy.log_model(
                    spacy_model=trained_nlp, 
                    artifact_path='model',
                    input_example=input_example,
                    signature=signature,
                )
                '''
                         
            else:
                logging.error("The model training was unsuccessful!")
                training_response_status = False
            
    else:
        logging.error("The base_config file is not found!")
        training_response_status = False
      
    response.status = training_response_status
    response.data = training_response_data

    #saving the response to sqlite
    with SqliteDict(TRAINING_RESULTS_DB, autocommit=True) as mydict:
        mydict[task_str] = response
    logging.info("The training response has been saved to SQLite.")

    #posting the response to IMO endpoint
    full_url = BASE_URL + task_str
    try:
        temp = requests.post(full_url, json=response.dict())
        logging.info(f"Response from the external API endpoint: {temp}")
    except:
        logging.warning("The training response was not successfully sent.")

    temp = "The training process is complete."
    logging.info(temp)
    print(temp)
    '''
    #saving the response to FastAPI server
    with open(training_response_path, 'wb') as f:
        pickle.dump(response, f)

    #json_data = jsonable_encoder(response)
    #saving the response to Redis

    pickled_object = pickle.dumps(response)
    save = client.set(task_str, pickled_object)
    if save:
        print("The training response has been saved to Redis.")
    else:
        print("The training response is NOT saved to Redis!")

    #publishing the response to Redis channels 
    #client.publish(f"training_results_{task_str}", pickled_object)
    
    '''
    return None

#-----------------------------------------------------------------------------
# FastAPI endpoints
#-----------------------------------------------------------------------------
# Create a training task
@app.post("/create", status_code=201, response_model=schemas.TaskResponseModel)
async def create_tasks(task: schemas.TrainingRequestModel, background_tasks: BackgroundTasks):
    background_tasks.add_task(train_model, task)
    response = schemas.TaskResponseModel(task_id=task.task_id)
    return response

# Predict classes with a classification model loaded from the FastAPI server
@app.post("/predict/{task_id}", response_model=schemas.InferenceResponseModel)
def get_predictions(task_id: str, query: schemas.InferenceRequestModel):
    model_folder = Path.cwd() / 'projects' / task_id / 'output' / 'model-best'
    if model_folder.exists():
        model_nlp = spacy.load(model_folder)
        texts = (article.text for article in query.articles)
        SN = 1
        predictions = []

        for doc in model_nlp.pipe(texts):
            prediction = schemas.Prediction(
                sn=SN,
                text=doc.text,
                cat=utils.get_cat(doc),
                conf=utils.get_conf(doc),
            )
            predictions.append(prediction)
            SN += 1

        response_data = predictions
        response_status = True
    else:
        response_data = []
        response_status = False

    response = schemas.InferenceResponseModel(
        status=response_status,
        feed_back=,
        data=response_data,
    )
    return response


@app.post()

if __name__ == "__main__":
    #uvicorn.run(app)
    uvicorn.run(app, host="0.0.0.0", port=5000)