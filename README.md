# Sci-Cheat
Basic analytics for working with a new dataset

## Notes
* Very much a work in progress!
* Works best in a Jupyter Notebook environment
* Install requirements using:
```
pip install -r requirements.txt
```
## Usage
* Import the PrimaryAnalysis module
* Load in a pandas dataframe and specify the target variable
* You can run specific methods or just use run_all()
```
from scicheat import PrimaryAnalysis()

pa = PrimaryAnalysis(df=df, target='y_col')
pa.run_all()
```

## To Do
* Error handling
* Deal with NAs intelligently
* Deal with dates intelligently
* Optionally limit rows for speed
* Hyperparameter tuning
* Unsupervised analysis
