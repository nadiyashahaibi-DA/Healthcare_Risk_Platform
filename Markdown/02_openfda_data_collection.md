# OpenFDA Data Collection


## Purpose


This notebook extracts adverse drug event information from the OpenFDA API.

Later I will add the data collected and investigate medication safety and potential drug risks into the healthcare analytics platform.


## Load Environment Variables


```python
import pandas as pd
import numpy as np
import requests 
import os
from dotenv import load_dotenv

```


```python
load_dotenv()
```




    True




```python
API_KEY = os.getenv("OPENFDA_API_KEY")
```


```python
from dotenv import load_dotenv
import os
import requests

load_dotenv()

API_KEY = os.getenv("OPENFDA_API_KEY")

print("API Key loaded:", API_KEY is not None)

url = f"https://api.fda.gov/drug/event.json?api_key={API_KEY}&limit=1"

response = requests.get(url)

print("Status:", response.status_code)

print(response.text[:500])
```

    API Key loaded: True
    Status: 200
    {
      "meta": {
        "disclaimer": "Do not rely on openFDA to make decisions regarding medical care. While we make every effort to ensure that data is accurate, you should assume all results are unvalidated. We may limit or otherwise restrict your access to the API in line with our Terms of Service.",
        "terms": "https://open.fda.gov/terms/",
        "license": "https://open.fda.gov/license/",
        "last_updated": "2026-07-30",
        "results": {
          "skip": 0,
          "limit": 1,
          "total": 2069269



```python
from dotenv import load_dotenv

load_dotenv(".env")
```




    True




```python
import os

print(os.getenv("OPENFDA_API_KEY"))
```


```python
import os

from dotenv import load_dotenv

load_dotenv(".env")

API_KEY = os.getenv("OPENFDA_API_KEY")

print(API_KEY[:5])
```

## API Request


```python
url = (
    f"https://api.fda.gov/drug/event.json?"
    f"api_key={API_KEY}&limit=100"
)
```


```python
response = requests.get(url)

response.status_code
```




    200



## Convert To DataFrame


```python
data = response.json()
```


```python
# Explore
data.keys()
```




    dict_keys(['meta', 'results'])




```python
# Create dataframe

events_df = pd.json_normalize(
    data["results"]
)

```


```python
# inspection

events_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>safetyreportid</th>
      <th>transmissiondateformat</th>
      <th>transmissiondate</th>
      <th>serious</th>
      <th>seriousnessdeath</th>
      <th>receivedateformat</th>
      <th>receivedate</th>
      <th>receiptdateformat</th>
      <th>receiptdate</th>
      <th>fulfillexpeditecriteria</th>
      <th>...</th>
      <th>sender.sendertype</th>
      <th>receiver.receivertype</th>
      <th>receiver.receiverorganization</th>
      <th>seriousnessother</th>
      <th>occurcountry</th>
      <th>patient.patientagegroup</th>
      <th>seriousnesshospitalization</th>
      <th>patient.summary.narrativeincludeclinical</th>
      <th>seriousnesslifethreatening</th>
      <th>patient.patientweight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5801206-7</td>
      <td>102</td>
      <td>20090109</td>
      <td>1</td>
      <td>1</td>
      <td>102</td>
      <td>20080707</td>
      <td>102</td>
      <td>20080625</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10003300</td>
      <td>102</td>
      <td>20141002</td>
      <td>1</td>
      <td>NaN</td>
      <td>102</td>
      <td>20140306</td>
      <td>102</td>
      <td>20140306</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>6</td>
      <td>FDA</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10003301</td>
      <td>102</td>
      <td>20141002</td>
      <td>1</td>
      <td>NaN</td>
      <td>102</td>
      <td>20140228</td>
      <td>102</td>
      <td>20140228</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>6</td>
      <td>FDA</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10003302</td>
      <td>102</td>
      <td>20141002</td>
      <td>2</td>
      <td>NaN</td>
      <td>102</td>
      <td>20140312</td>
      <td>102</td>
      <td>20140312</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>6</td>
      <td>FDA</td>
      <td>NaN</td>
      <td>US</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10003304</td>
      <td>102</td>
      <td>20141212</td>
      <td>2</td>
      <td>NaN</td>
      <td>102</td>
      <td>20140312</td>
      <td>102</td>
      <td>20140424</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>6</td>
      <td>FDA</td>
      <td>NaN</td>
      <td>US</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>




```python
# Save Data

events_df.to_csv(
    "../data/raw/openfda_events.csv",
    index=False
)
```

**To be continued on the next notebook**
