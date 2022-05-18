# Prediction-of-Solar-Power-Energy-Generation

DataSet
---------
<p>Dataset(Solar energy power generation and weather data) from the UK region of the Open Power System Data Project </p>
*Download : https://data.open-power-system-data.org/time_series/

Installation
---------
Clone the repository
```
git clone https://github.com/juhjoo/Prediction-of-Solar-Power-Energy-Generation.git
```
```
pip install -r requirements.txt
```
```
python main.py > output.txt 2> error.txt
```

Dependencies
---------
* Python 3.8
* Pandas 
* Tensorflow 2.6
* Keras
* Matplotlib
* Seaborn
* Numpy
* tf-nightly
* Scipy = 1.4.1

Model
---------
* Proposed Model(DCCNN+LSTM)
* DCCNN(Dilated Causal Convolutional Nueral Network)
* LSTM

Model Architecture
--------
<p float="left">
  <img src="image/architecture.JPG" alt="drawing" width="600"/>
</p>

Results
-------
<p float="left">
  <img src="image/result1.JPG" alt="drawing" />
    </p>
    <p float="left">
  <img src="image/result2.JPG" alt="drawing" />
</p>


| Model  | Best Score(WAPE)  |
| --------- | -------|
|DCCNN+LSTM	|0.268|
|DCCNN|	0.278|
|LSTM|	0.278|

