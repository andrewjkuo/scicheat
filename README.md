# Sci-Cheat
Basic analytics for working with a new dataset

## Notes
* See examples folder for demo of functionality
* Very much a work in progress!
* Works best in a Jupyter Notebook environment
## Installation
```
pip install scicheat
```
OR
```
git clone https://github.com/andrewjkuo/scicheat.git
python setup.py install
```
## Usage
* Import the PrimaryAnalysis class
* Load in a pandas dataframe and specify the target variable
* You can run specific methods or just use run_all()
```
from scicheat.dataloader import PrimaryAnalysis

pa = PrimaryAnalysis(df=df, target='y_col')
pa.run_all()
```
## To Do
* Error handling
* Deal with NAs intelligently
* Deal with dates intelligently
* Data normalisation
* Optionally limit rows for speed
* Hyperparameter tuning
* Unsupervised analysis
* Only show "interesting" visualisations if too many columns
