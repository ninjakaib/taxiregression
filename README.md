# taxiregression
* * *

**Please refer to [this](https://github.com/ninjakaib/taxiregression/blob/main/final%20report.pdf) version of the final report instead of the gradescope one. There are no changes to the report besides the inclusion of a brief note under the table for different models and the figures which I was having an error with Overleaf and could not get them to properly render until a few minutes after the submission closed.**

* * *

## Model Inference
To make predictions with the model, grab the model files here: https://drive.google.com/drive/folders/19CmhXSCVZlvpceNIafC_WRiMCgVImsUS?usp=drive_link

And run the code below

```shell
pip install autogluon
```


```python
from autogluon.tabular import TabularDataset, TabularPredictor
from datetime import datetime
```


```python
def clean(df):
    # Create a copy of the df for preprocessing
    df_processed = df.copy()
    
    def parse_time(x):
      dt = datetime.fromtimestamp(x["TIMESTAMP"])
      return dt.year, dt.month, dt.strftime('%j'), dt.day, dt.hour, dt.weekday()

    # Cyclical encoding of time related features
    df_processed[["YR", "MON", "DYEAR","DAY", "HR", "WK"]] = df_processed[["TIMESTAMP"]].apply(parse_time, axis=1, result_type="expand")
    
    df_processed['MON_SIN'] = np.sin((df_processed['MON']-1)*(2.*np.pi/12))
    df_processed['MON_COS'] = np.cos((df_processed['MON']-1)*(2.*np.pi/12))

    df_processed['DAY_SIN'] = np.sin((df_processed['DAY']-1)*(2.*np.pi/31))
    df_processed['DAY_COS'] = np.cos((df_processed['DAY']-1)*(2.*np.pi/31))

    df_processed['HR_SIN'] = np.sin(df_processed['HR']*(2.*np.pi/24))
    df_processed['HR_COS'] = np.cos(df_processed['HR']*(2.*np.pi/24))

    df_processed['WK_SIN'] = np.sin(df_processed['WK']*(2.*np.pi/7))
    df_processed['WK_COS'] = np.cos(df_processed['WK']*(2.*np.pi/7))

    # One hot encode YR
    df_processed[f'YR_2013'] = (df_processed['YR']==2013).astype(int)
    df_processed[f'YR_2014'] = (df_processed['YR']==2014).astype(int)
  
    return df_processed
```


```python
path_to_data = 'test_public.csv'
path_to_model = 'fullensemble'
data = clean(TabularDataset(path_to_data).set_index('TRIP_ID'))
predictor = predictor.load(path_to_model)
preds = predictor.predict(data).rename('TRAVEL_TIME')
preds.to_csv('predictions.csv')
```


```python

```

