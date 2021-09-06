# -----------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------
# Standard library imports
from pathlib import Path
import subprocess
import sys
import time
import logging
import sys

# Third-party imports
from fastapi import FastAPI, BackgroundTasks
import uvicorn  # API伺服器
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import mlflow
import mlflow.spacy

# from mlflow.models.signature import infer_signature #mlflow.log_model的時時候用
import plotly.express as px
from sqlitedict import SqliteDict
import requests

# Local application imports
import schemas  # 各種輸入輸出的Class
import utils  # 輔助函數

# -----------------------------------------------------------------------------
# Initialize the FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="文本分類API",
    version="v2.1 2021.07 HamastarAI",
    description="""用於文本分類建模與預測。端點說明：
  1. /create: 建立模型訓練任務，回傳訓練代號
  2. /predict/{task_id}: 使用訓練代號預測資料
  """,
)

# -----------------------------------------------------------------------------
# Variables for the dataset
# -----------------------------------------------------------------------------
# dataset column matching
TEXT_COL, LABEL_COL = "text", "label"

# -----------------------------------------------------------------------------
# The MLflow server where tracking is saved
# -----------------------------------------------------------------------------
# MLflow伺服器位址
MLFLOW_TRACKING_URI = "http://192.168.201.176:443"

# -----------------------------------------------------------------------------
# SQLiteDict database for training results
# -----------------------------------------------------------------------------
# 儲存訓練結果的SQLite DB
TRAINING_RESULTS_DB = "./training_results_db.sqlite"
PLOTS_DB = "./plots_db.sqlite"

# -----------------------------------------------------------------------------
# API endpoint for training results
# -----------------------------------------------------------------------------
# ASIP平台接收訓練結果基本網址
BASE_URL = "https://aihub.hamastar.com.tw/api/model/feedback/"

# -----------------------------------------------------------------------------
# Background functions
# -----------------------------------------------------------------------------
# The training function
# 建模任務的背景工作
def train_model(task: schemas.TrainingRequestModel):
    # 定義各種路徑
    task_str = str(task.task_id)
    task_folder = Path.cwd() / "projects" / task_str
    data_folder = task_folder / "data"
    data_folder.mkdir(parents=True, exist_ok=True)
    train_path = data_folder / "train.spacy"
    dev_path = data_folder / "dev.spacy"
    config_path = task_folder / "config.cfg"
    output_folder = task_folder / "output"
    model_folder = output_folder / "model-best"
    artifact_folder = task_folder / "artifact"
    artifact_folder.mkdir(parents=True, exist_ok=True)
    log_path = Path.cwd() / "logging.txt"
    logging.basicConfig(
        filename=log_path,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logging.info(f"Received a training request: {task_str}")

    # 回傳結果格式設定
    response = schemas.TrainingResponseModel()
    training_response_data = [{}] * 5  # 回傳五個結果，平台以五個頁籤呈現

    # 建模參數
    base_config_path = utils.make_base_config(task.efficiency)
    ngram_config_key = utils.make_ngram_key(task.efficiency)
    NGRAM_SIZE = task.ngram_size
    MAX_LEN = task.max_len

    # 確認基本設定檔存在
    if base_config_path:
        # 建模基本設定
        config_command = (
            f"python -m spacy init fill-config {base_config_path} {config_path}"
        )
        config_command_list = config_command.split()
        config_command_output = subprocess.run(config_command_list, capture_output=True)
        logging.info(config_command_output.stdout.decode("utf-8"))

        # 載入資料集
        data = ({TEXT_COL: item.text, LABEL_COL: item.label} for item in task.dataset)
        df = pd.DataFrame(data)

        # 資料集前處理
        # drop duplicates
        df.drop_duplicates(subset=TEXT_COL, inplace=True)
        df = df.astype(str)
        # drop texts shorter than min_len
        len_mask = df[TEXT_COL].str.len() > task.min_len
        df = df.loc[len_mask]
        # preprocess raw text
        df.loc[:, TEXT_COL] = df[TEXT_COL].apply(utils.preprocess_text)

        # 最終使用的資料集
        df.loc[:, LABEL_COL] = df[LABEL_COL].str.strip()
        unique_labels = df[LABEL_COL].unique().tolist()
        labels_str = " | ".join(unique_labels)
        dataset_size = sys.getsizeof(df)
        train_df, dev_df = train_test_split(
            df,
            test_size=task.test_size,
            random_state=1,
            stratify=df[LABEL_COL],
        )
        training_samples, testing_samples = train_df.shape[0], dev_df.shape[0]

        # 資料集格式轉換
        train_list = utils.df2list(train_df, TEXT_COL, LABEL_COL)
        dev_list = utils.df2list(dev_df, TEXT_COL, LABEL_COL)
        logging.info("Starting to convert the dataset to spaCy format...")
        utils.make_docs(train_list, unique_labels, train_path)
        logging.info("The training data have been successfully converted and saved.")
        utils.make_docs(dev_list, unique_labels, dev_path)
        logging.info("The testing data have been successfully converted and saved.")

        # MLflow伺服器設定
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"專案{task_str}")  # 命名experiment
        experiment = mlflow.get_experiment_by_name(f"專案{task_str}")
        experiment_id = experiment.experiment_id
        # 舉例 http://192.168.201.176:443/#/experiments/90 experiment_id = 90

        # 開始experiment的第一run
        with mlflow.start_run():
            # 取得run_id
            # 舉例 http://192.168.201.176:443/#/experiments/91/runs/f4c4d765812348fca2e718056371417d
            # run_id = f4c4d765812348fca2e718056371417d
            run = mlflow.active_run()
            run_id = run.info.run_id
            plot_folder = f"{experiment_id}/{run_id}/artifacts/plots/"

            # 開始建模
            temp_message = "Starting to train models..."
            print(temp_message)
            logging.info(temp_message)
            train_command = f"""python -m spacy train {config_path} 
                            --output {output_folder} 
                            --paths.train {train_path} 
                            --paths.dev {dev_path} 
                            --{ngram_config_key} {NGRAM_SIZE} 
                            --corpora.train.max_length {MAX_LEN} 
                            --verbose"""
            train_command_list = train_command.split()
            training_start = time.time()
            train_command_output = subprocess.run(
                train_command_list, capture_output=True
            )

            # 紀錄訓練歷史
            training_history = train_command_output.stdout.decode("utf-8")
            training_history_path = artifact_folder / "history.txt"
            # 記錄在API伺服器
            logging.info(training_history)
            # 記錄在此一run的artifact_folder
            training_history_path.write_text(training_history)
            # 記錄在MLflow伺服器
            mlflow.log_artifact(str(training_history_path))

            # 將建模參數紀錄在MLflow伺服器
            mlflow.log_param("task_id", task_str)
            mlflow.log_param("test_size", str(task.test_size))
            mlflow.log_param("efficiency", str(task.efficiency))
            mlflow.log_param("dataset_size", str(dataset_size))
            mlflow.log_param("training_samples", str(training_samples))
            mlflow.log_param("testing_samples", str(testing_samples))
            mlflow.log_param("labels", labels_str)
            mlflow.log_param("ngram_size", str(NGRAM_SIZE))
            mlflow.log_param("max_len", str(MAX_LEN))
            mlflow.log_param("min_len", str(task.min_len))
            # mlflow.log_param("cpu", str(task.cpu))

            # 將建模參數存到回傳結果頁籤1
            params_value = schemas.ParamsValue(
                task_id=task_str,
                test_size=str(task.test_size),
                efficiency=str(task.efficiency),
                dataset_size=str(dataset_size),
                training_samples=str(training_samples),
                testing_samples=str(testing_samples),
                labels=labels_str,
                ngram_size=str(NGRAM_SIZE),
                max_len=str(MAX_LEN),
                min_len=str(task.min_len),
                # cpu=str(task.cpu),
            )
            re_params = schemas.Params(value=params_value)
            tab1 = re_params
            training_response_data[0] = tab1
            logging.info("The training parameters have been logged.")

            # 繪製類別數量圖
            counts = utils.train_test_counts(train_df, dev_df, LABEL_COL)
            fig = px.bar(
                counts,
                x=LABEL_COL,
                y="類別數",
                color="資料類別",
                title="類別分布圖",
                barmode="group",
                height=400,
            )
            class_counts_filename = "class_counts.html"
            class_counts_path = artifact_folder / class_counts_filename
            # 將圖寫到API伺服器
            fig.write_html(str(class_counts_path))
            # 將圖存到MLflow伺服器
            mlflow.log_artifact(str(class_counts_path), artifact_path="plots")
            # 將圖存到回傳結果頁籤3
            temp_name = plot_folder + class_counts_filename
            re_plot_count = schemas.PlotCount(value=temp_name)
            tab3 = re_plot_count
            training_response_data[2] = tab3
            logging.info("A plot of class counts has been saved.")
            temp_html = fig.to_html()
            # 將圖存到SQLite DB
            with SqliteDict(PLOTS_DB, autocommit=True) as mydict:
                mydict[temp_name] = temp_html

            # 若訓練結果正常
            if train_command_output.returncode == 0:
                training_end = time.time()
                training_time = training_end - training_start
                result_message = "The model has been successfully trained."
                print(result_message)
                logging.info(result_message)
                training_response_status = True

                # 繪製各類別評估指標熱力圖
                metrics = utils.make_metrics(model_folder)
                per_cat_dict = metrics["per_category"]
                per_cat_df = pd.DataFrame(per_cat_dict)
                fig = px.imshow(
                    per_cat_df,
                    x=per_cat_df.columns.tolist(),
                    y="精準率 召回率 F1值".split(),
                    labels=dict(x="類別名稱", y="評估指標", color="數值大小"),
                )
                heatmap_per_cat_filename = "metrics_per_category.html"
                heatmap_per_cat_path = artifact_folder / heatmap_per_cat_filename
                # 將圖寫到API伺服器
                fig.write_html(str(heatmap_per_cat_path))
                # 將圖存到MLflow伺服器
                mlflow.log_artifact(str(heatmap_per_cat_path), artifact_path="plots")
                # 將圖存到回傳結果頁籤4
                temp_name = plot_folder + heatmap_per_cat_filename
                re_plot_per_cat = schemas.PlotPerCat(value=temp_name)
                tab4 = re_plot_per_cat
                training_response_data[3] = tab4
                logging.info("A plot of model performance per category has been saved.")
                temp_html = fig.to_html()
                # 將圖存到SQLite DB
                with SqliteDict(PLOTS_DB, autocommit=True) as mydict:
                    mydict[temp_name] = temp_html

                # 繪製混淆矩陣熱力圖
                trained_nlp = spacy.load(model_folder)
                y_true = [tup[1] for tup in dev_list]
                dev_texts = (tup[0] for tup in dev_list)
                y_pred = []
                for doc in trained_nlp.pipe(dev_texts):
                    y_pred.append(utils.get_cat(doc))
                cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
                accuracy = accuracy_score(y_true, y_pred)
                fig = px.imshow(
                    cm,
                    x=unique_labels,
                    y=unique_labels,
                    labels=dict(x="預測類別", y="實際類別", color="數值大小"),
                )
                cm_filename = "confusion_matrix.html"
                cm_path = artifact_folder / cm_filename
                # 將圖寫到API伺服器
                fig.write_html(str(cm_path))
                # 將圖存到MLflow伺服器
                mlflow.log_artifact(str(cm_path), artifact_path="plots")
                temp_name = plot_folder + cm_filename
                re_plot_confusion_matrix = schemas.PlotConfusionMatrix(value=temp_name)
                tab5 = re_plot_confusion_matrix
                training_response_data[4] = tab5
                logging.info("A plot of confusion matrix has been saved.")
                temp_html = fig.to_html()
                # 將圖存到SQLite DB
                with SqliteDict(PLOTS_DB, autocommit=True) as mydict:
                    mydict[temp_name] = temp_html

                # 將模型表現紀錄在MLflow伺服器
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
                    training_time=training_time,
                )

                # 將模型表現存到回傳結果頁籤2
                re_metrics = schemas.Metrics(value=metrics_value)
                tab2 = re_metrics
                training_response_data[1] = tab2
                logging.info("The testing metrics have been logged.")

                # 將模型存到MLflow伺服器
                """ 
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
                """

            else:
                logging.error("The model training was unsuccessful!")
                training_response_status = False

    else:
        logging.error("The base_config file is not found!")
        training_response_status = False

    # 更新訓練回傳結果
    response.status = training_response_status
    response.data = training_response_data

    # 將訓練回傳結果存到SQLite DB
    with SqliteDict(TRAINING_RESULTS_DB, autocommit=True) as mydict:
        mydict[task_str] = response
    logging.info("The training response has been saved to SQLite.")

    # 將訓練回傳結果傳送至AISP平台
    full_url = BASE_URL + task_str
    try:
        temp = requests.post(full_url, json=response.dict())
        logging.info(f"Response from the external API endpoint: {temp}")
    except:
        logging.warning("The training response was not successfully sent.")

    temp = "The training process is complete."
    logging.info(temp)
    print(temp)

    return None


# -----------------------------------------------------------------------------
# FastAPI endpoints
# -----------------------------------------------------------------------------
# Create a training task
# 創建建模任務請求
@app.post("/create", status_code=201, response_model=schemas.TaskResponseModel)
async def create_tasks(
    task: schemas.TrainingRequestModel, background_tasks: BackgroundTasks
):
    background_tasks.add_task(train_model, task)  # 將建模任務轉為背景工作
    response = schemas.TaskResponseModel(task_id=task.task_id)
    return response


# Predict classes with a classification model loaded from the FastAPI server
# 模型預測任務請求
@app.post("/predict/{task_id}", response_model=schemas.InferenceResponseModel)
def get_predictions(task_id: str, query: schemas.InferenceRequestModel):
    model_folder = Path.cwd() / "projects" / task_id / "output" / "model-best"
    if model_folder.exists():
        model_nlp = spacy.load(model_folder)
        raw_texts = [article.text for article in query.articles]
        clean_texts = (utils.preprocess_text(raw_text) for raw_text in raw_texts)
        SN = 0
        predictions = []

        for doc in model_nlp.pipe(clean_texts):
            prediction = schemas.Prediction(
                sn=SN,
                text=raw_texts[SN],
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
        data=response_data,
    )
    return response


if __name__ == "__main__":
    uvicorn.run(app)  # 除錯時使用
    # uvicorn.run(app, host="0.0.0.0", port=5000) #部署時使用
