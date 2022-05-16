1. The instruction for execute program
pip install -r requirements.txt
python main.py > output.txt 2> error.txt

2. The Structure of Artifacts
• Requirement.txt : install a library for experience setting
• Readme.txt : the instruction for execute artifacts
• Self Assessment Ethics form.doc: Ethics form approved by supervisor
• run.sh : An Execution files for artifacts based on linux OS.
- Python Programs
• main.py – Use functions in experience.py, process.
• src/data_generation.py – Read and preprocess data set for training and testing 
• src/experience.py – Functions to predict a result using models and save its as CSV
• src/metric.py – Function to calculate metrics (WAPE, MAE, and etc.)
• src/model.py - Functions to make models (proposed, DCCNN, LSTM)
• src/visualization.py – Create graph for the timeseries data and results
- Notebook Files
• main_experience.ipynb – Make a model and save results as CSV file
• main_result.ipynb – Create graph for the result using results
• main_visualization.igynb – Create graph for analyzing time series datasets 
- Data Files
• data/time_series_60min_singleindex.csv – CSV file of solar generation time series data
• data/weather_data.csv – CSV file of weather time series data
- Result Files
• result/result_dclstm.csv – CSV file of proposed model results
• result/result_dccnn.csv – CSV file of DCCNN model results
• result/result_lstm.csv – CSV file of LSTM model results
- Model Result Files
• model/ – Model result files of proposed, DCCNN and LSTM models
• model/DC_CN_LSTM_Model#.h5 and DCCNN_LSTM_Model.zip  – Model result files of proposed models
• model/DCCNN_Model#.h5 and DCCNN_Model.zip – Model result files of DCCNN models
• model/LSTM_Model#.h5 and LSTM_Model.zip– Model result files of LSTM models