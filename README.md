# Leader Detection
This repo uses Cyclicity Analysis to determine the ranking of each FX Pair.
[explain more][to do]
## Cyclicity Analysis of Time-Series
cyclicity_analysis.py is a working implementation of Cyclicity Analysis, which is a pattern recognition technique for analyzing the leader follower dynamics of multiple time-series.

### Usage

```python
from cyclicity_analysis import OrientedArea, COOM

df = pd.DataFrame([[0, 1], [1, 0], [0, 0]], columns=['0', '1'])


oa = OrientedArea(df)
# Returns the lead lag matrix of df as a dataframe
lead_lag_df = oa.compute_lead_lag_df()

coom = COOM(lead_lag_df)
# Returns leading eigenvector of lead lag matrix as a numpy array
leading_eigenvector = coom.get_leading_eigenvector()
lead_lag_df , leading_eigenvector
 ```
# Requirements
Download [Python >=3.7](https://www.python.org/downloads/)

# Installation

```bash
python -m pip install -r requirements.txt
```
# Running the Code
[to do]
