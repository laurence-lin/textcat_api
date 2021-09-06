#-----------------------------------------------------------------------------
# Import libraries 
#-----------------------------------------------------------------------------
# Standard library imports
from typing import List, Optional

# Third-party imports
from pydantic import BaseModel, Field

#-----------------------------------------------------------------------------
# Pydantic classes
#-----------------------------------------------------------------------------
# Request schema for training
class TrainingRequestModel(BaseModel):
    class TrainingData(BaseModel):
        text: str = Field(None, description="文本欄位")
        label: str = Field(None, description="類別欄位")

    task_id: str = Field(None, description="專案代號")
    ngram_size: int = Field(1, description="Ngram大小，建議值1~5")
    min_len: int = Field(20, description="文本長度下限")
    test_size: float = Field(0.2, description="測試資料集比例，建議值0.2~0.5")
    efficiency: bool = Field(True, description="true為效率優先, false為正確率優先")
    dataset: Optional[List[TrainingData]] = Field(None, description="資料集列表，建議至少200筆資料")
    #cpu: bool = Field(True, description="true為使用CPU, false為使用GPU")

#Response schema for training 
class ParamsValue(BaseModel):
    task_id: Optional[str] = Field(None, description="專案代號")
    test_size: Optional[str] = Field(None, description="測試集比例")
    efficiency: Optional[str] = Field(None, description="true為效率優先, false為正確率優先")
    dataset_size: Optional[str] = Field(None, description="資料集位元組大小")
    training_samples: Optional[str] = Field(None, description="訓練資料筆數")
    testing_samples: Optional[str] = Field(None, description="測試資料筆數")
    labels: Optional[str] = Field(None, description="分類標籤")
    ngram_size: Optional[str] = Field(None, description="Ngram大小")
    min_len: Optional[str] = Field(None, description="文本長度下限")
    #cpu: Optional[str] = Field(None, description="true為使用CPU, false為使用GPU")

class Params(BaseModel):
    name: str = Field("建模參數", description="建模參數")
    type: str = Field("value", description="表格資料")
    value: Optional[ParamsValue] = Field(None, description="資料數值")
  
class MetricsValue(BaseModel):
    precision: Optional[float] = Field(None, description="精準率")
    recall: Optional[float] = Field(None, description="召回率")
    F1: Optional[float] = Field(None, description="F1")
    AUC: Optional[float] = Field(None, description="AUC")
    accuracy: Optional[float] = Field(None, description="正確率")
    training_time: Optional[float] = Field(None, description="訓練總秒數")

class Metrics(BaseModel):
    name: str = Field("模型評估指標", description="模型評估指標")
    type: str = Field("value", description="表格資料")
    value: Optional[MetricsValue] = Field(None, description="資料數值")
    #per_category: Optional[Dict[str, Any]] = Field(None, description="分項類別數據")

class PlotCount(BaseModel):
    name: str = Field("類別分布圖", description="圖表標題")
    type: str = Field("file", description="檔案資料")
    value: Optional[str] = Field(None, description="檔案路徑")

class PlotPerCat(BaseModel):
    name: str = Field("各類別評估指標圖", description="圖表標題")
    type: str = Field("file", description="檔案資料")
    value: Optional[str] = Field(None, description="檔案路徑")

class PlotConfusionMatrix(BaseModel):
    name: str = Field("混淆矩陣圖", description="圖表標題")
    type: str = Field("file", description="檔案資料")
    value: Optional[str] = Field(None, description="檔案路徑")

'''
class TrainingResponseData(BaseModel):
    tab1: Optional[Params] = Field(None, description="頁籤1")
    tab2: Optional[Metrics] = Field(None, description="頁籤2")stat
    tab3: Optional[PlotCount] = Field(None, description="頁籤3")
    tab4: Optional[PlotPerCat] = Field(None, description="頁籤4")
    tab5: Optional[PlotConfusionMatrix] = Field(None, description="頁籤5")
'''

class TrainingResponseModel(BaseModel):
    status: bool = Field(None, description="回傳狀態")
    data: Optional[List] = Field(None, description="訓練結果")

class TaskResponseModel(BaseModel):
    status: bool = Field(True, description="回傳狀態")
    task_id: str = Field(None, description="專案代號")

#-----------------------------------------------------------------------------
# Classes for inference
#-----------------------------------------------------------------------------
#Request schema for inferences
class Article(BaseModel):
    # Schema for a single article in a batch of articles to process
    #sn: int = Field(None, description="文本編號")
    text: str = Field(None, description="文本內容")

class InferenceRequestModel(BaseModel):
    articles: List[Article] = Field(None, description="待分類文本列表")

#Response schema for inference
class Prediction(BaseModel):
        sn: int = Field(None, description="文本編號")
        text: str = Field(None, description="文本內容")
        cat: str = Field(None, description="文本類別")
        conf: float = Field(None, description="模型信度")
        #output_proba: Dict[str, float] = Field(None, description="模型輸出類別機率分布")
        #output_max: List[Any] = Field(None, description="模型輸出最大機率與類別")

class InferenceResponseModel(BaseModel):
    status: bool = Field(None, description="回傳狀態")
    feedback: str
    data: Optional[List[Prediction]] = Field(None, description="預測結果列表")
