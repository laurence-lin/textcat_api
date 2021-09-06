#-----------------------------------------------------------------------------
# Import libraries 
#-----------------------------------------------------------------------------
# Standard library imports

# Third-party imports
from fastapi import FastAPI, HTTPException
import uvicorn
from sqlitedict import SqliteDict

# Local application imports
import schemas

#-----------------------------------------------------------------------------
# Initialize the FastAPI app
#-----------------------------------------------------------------------------
app = FastAPI(
  title = "SQLite資料庫API",
  version= "v1.2 2021.07 HamastarAI",
  description = '''端點說明:
  1. /plots: 輸入圖表路徑，輸出HTML字串
  2. /results: 輸入任務代號，輸出訓練結果
  '''
)

#-----------------------------------------------------------------------------
# SQLiteDict databases
#-----------------------------------------------------------------------------
PLOTS_DB = './plots_db.sqlite'
TRAINING_RESULTS_DB = './training_results_db.sqlite'
plots_dict = SqliteDict(PLOTS_DB)
training_results_dict = SqliteDict(TRAINING_RESULTS_DB)

#-----------------------------------------------------------------------------
# API Endpoints
#-----------------------------------------------------------------------------
# Get plotly string representations from the SQLiteDict database
@app.get("/plots", status_code=200)
def render_plots(plot_path: str):
    response = plots_dict.get(plot_path)
    if response:
        return response
    else:
        raise HTTPException(status_code=404, detail="Plot not found")

# Get training results from the SQLiteDict database
@app.get("/results", status_code=200, response_model=schemas.TrainingResponseModel)
def get_results(task_id: str):
    response = training_results_dict.get(task_id)
    if response:
        return response
    else:
        raise HTTPException(status_code=404, detail="Result not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)