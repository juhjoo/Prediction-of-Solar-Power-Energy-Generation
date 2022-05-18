# Prediction-of-Solar-Power-Energy-Generation

#data
    data
      ├── batch_01_vt
      │   ├── 데이터셋 (이미지)
      │   ├── ... 
      │   └── 
      ├── batch_02_vt
      │   ├── 데이터셋 (이미지)
      │   ├── ... 
      │   └── 
      ├── batch_03
      │   ├── 데이터셋 (이미지)
      │   ├── ...
      │   └── 
      ├── train_all.json
      ├── train.json
      ├── val.json
      ├── train_data0.json
      ├── ... 
      ├── valid_data4.json
      ├── train_data_pesudo0.json
      ├── ... 
      ├── valid_data_pesudo4.json
      └── test.json
code
  ├── PyTorch DeepLabv3plus Code.ipynb
  ├── PyTorch EfficientFPN Code.ipynb
  ├── Pesudo Labeling.ipynb
  ├── Ensemble All Models.ipynb
  └── Make A Stratified KFold Json.ipynb
losses
  ├── dice.py
  ├── ...
  └── soft_ce.py
utils.py 
