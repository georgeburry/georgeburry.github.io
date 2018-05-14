---
title: "Kaggle competition: Porto Seguro’s Safe Driver Prediction"
date: 2018-05-14
tags: [classification]
header:
  image: "/images/machine-learning-header.jpg"
excerpt: "Predicting if drivers will file an insurance claim"
mathjax: ""
---
## Predicting if a driver will file an insurance claim.

[The Jupyter notebook can be accessed here](https://github.com/georgeburry/safe-driver-prediction/blob/master/Kaggle_competition_insurance-claims.ipynb)


# Investigating crashes using API data
A number of factors can increase the chances of a car crash occuring - for instance: weather, drink-driving or time of day. In this in project, I set out to collect data from the web in order to understand the circumstances surrounding car crashes in the state of Maryland, U.S. The crash data was taken from the U.S. website: data.gov.

1. First of all, the Foursquare API is used to collect data about the number of bars within in a certain radius in each County, in order to see if the prevalence of drinking might have an influence on the accident rate.
2. Second of all, the Google Maps API is used to determine the coordinates of each county, then the coordinates are used to request the weather conditions at the time of the accident from the DarkSky API.
3. Finally, the coordinates of the accidents themselves are obtained and plotted on a map using the Google Map Plotter.

## Data

The data for this exercise can be found [here](https://catalog.data.gov/dataset/2012-vehicle-collisions-investigated-by-state-police-4fcd0/resource/d84f79b6-419c-49e0-a74c-01b34a9575f2).

Just run the cells below to get the data ready.


```python
import pandas as pd
mypath = "./"
```


```python
data = pd.read_csv(mypath + "2012_Vehicle_Collisions_Investigated_by_State_Police.csv",
                   parse_dates=[["ACC_DATE", "ACC_TIME"]])
data["MONTH"] = data.ACC_DATE_ACC_TIME.dt.month
data.dropna(subset=["COUNTY_NAME"], inplace=True) #get rid of empty counties
```

Now let's check the length of the data and the names of the counties and the dataset itself.


```python
len(data)
```




    18604




```python
data.COUNTY_NAME.unique()
```




    array(['Montgomery', 'Worcester', 'Calvert', 'St. Marys', 'Baltimore',
           'Prince Georges', 'Anne Arundel', 'Cecil', 'Charles', 'Carroll',
           'Harford', 'Frederick', 'Howard', 'Allegany', 'Garrett', 'Kent',
           'Queen Annes', 'Washington', 'Somerset', 'Wicomico', 'Talbot',
           'Caroline', 'Dorchester', 'Not Applicable', 'Unknown',
           'Baltimore City'], dtype=object)




```python
data.head()
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
      <th>ACC_DATE_ACC_TIME</th>
      <th>CASE_NUMBER</th>
      <th>BARRACK</th>
      <th>ACC_TIME_CODE</th>
      <th>DAY_OF_WEEK</th>
      <th>ROAD</th>
      <th>INTERSECT_ROAD</th>
      <th>DIST_FROM_INTERSECT</th>
      <th>DIST_DIRECTION</th>
      <th>CITY_NAME</th>
      <th>COUNTY_CODE</th>
      <th>COUNTY_NAME</th>
      <th>VEHICLE_COUNT</th>
      <th>PROP_DEST</th>
      <th>INJURY</th>
      <th>COLLISION_WITH_1</th>
      <th>COLLISION_WITH_2</th>
      <th>MONTH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-01-01 02:01:00</td>
      <td>1363000002</td>
      <td>Rockville</td>
      <td>1</td>
      <td>SUNDAY</td>
      <td>IS 00495 CAPITAL BELTWAY</td>
      <td>IS 00270 EISENHOWER MEMORIAL</td>
      <td>0.00</td>
      <td>U</td>
      <td>Not Applicable</td>
      <td>15.0</td>
      <td>Montgomery</td>
      <td>2.0</td>
      <td>YES</td>
      <td>NO</td>
      <td>VEH</td>
      <td>OTHER-COLLISION</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012-01-01 18:01:00</td>
      <td>1296000023</td>
      <td>Berlin</td>
      <td>5</td>
      <td>SUNDAY</td>
      <td>MD 00090 OCEAN CITY EXPWY</td>
      <td>CO 00220 ST MARTINS NECK RD</td>
      <td>0.25</td>
      <td>W</td>
      <td>Not Applicable</td>
      <td>23.0</td>
      <td>Worcester</td>
      <td>1.0</td>
      <td>YES</td>
      <td>NO</td>
      <td>FIXED OBJ</td>
      <td>OTHER-COLLISION</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-01-01 07:01:00</td>
      <td>1283000016</td>
      <td>Prince Frederick</td>
      <td>2</td>
      <td>SUNDAY</td>
      <td>MD 00765 MAIN ST</td>
      <td>CO 00208 DUKE ST</td>
      <td>100.00</td>
      <td>S</td>
      <td>Not Applicable</td>
      <td>4.0</td>
      <td>Calvert</td>
      <td>1.0</td>
      <td>YES</td>
      <td>NO</td>
      <td>FIXED OBJ</td>
      <td>FIXED OBJ</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-01-01 00:01:00</td>
      <td>1282000006</td>
      <td>Leonardtown</td>
      <td>1</td>
      <td>SUNDAY</td>
      <td>MD 00944 MERVELL DEAN RD</td>
      <td>MD 00235 THREE NOTCH RD</td>
      <td>10.00</td>
      <td>E</td>
      <td>Not Applicable</td>
      <td>18.0</td>
      <td>St. Marys</td>
      <td>1.0</td>
      <td>YES</td>
      <td>NO</td>
      <td>FIXED OBJ</td>
      <td>OTHER-COLLISION</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012-01-01 01:01:00</td>
      <td>1267000007</td>
      <td>Essex</td>
      <td>1</td>
      <td>SUNDAY</td>
      <td>IS 00695 BALTO BELTWAY</td>
      <td>IS 00083 HARRISBURG EXPWY</td>
      <td>100.00</td>
      <td>S</td>
      <td>Not Applicable</td>
      <td>3.0</td>
      <td>Baltimore</td>
      <td>2.0</td>
      <td>YES</td>
      <td>NO</td>
      <td>VEH</td>
      <td>OTHER-COLLISION</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Google Maps API
Now we will use the Google Maps API.


```python
from googlemaps.exceptions import TransportError
import googlemaps
from googlemaps.exceptions import HTTPError
```


```python
gmaps = googlemaps.Client(key=os.environ["GOOGLE_API_BASECAMP_KEY"])
```

This function is used to get the official latitude and longitude for each county, by making calls to the API.


```python
county_names = list(set(data.COUNTY_NAME.unique()))
```

## Foursquare API

Foursquare API documentation is [here](https://developer.foursquare.com/)

1. Start a foursquare application and get your keys.
2. For each crash, pull number of of bars (category "Nightlife") in 5km radius.
3. Find a relationship between number of bars in the area and severity of the crash.
4. (optional) Try to come up with other approaches to get more information out of the data.
5. (optional) Think about the most generic way to approach the problem.

Hints:

* check out python package "foursquare"
* what happens if the code fails?
* what if you run out of requests? (check out [time](https://docs.python.org/2/library/time.html) package)


```python
#set the keys
foursquare_id = '0NQCHENU5SVZ2HOIHUCXKSW3Y55VJPDJ3FYL4VPHG5MINZPV'
foursquare_secret = "L5QWH4QUK0NTE3LPFVTAZIV43GHINZ0C4ZZ3GVPXAFUWEI3U"
```


```python
#install and load the library
from foursquare import Foursquare
```

Now we need to set up the client in order to make calls to the API.


```python
client = Foursquare(client_id = foursquare_id,
                   client_secret = foursquare_secret)
```

We will loop through the counties and obtain the number of bars within a 5km radius (up to a maximum of 50 bars). If the call quota is exceeded, then the code the operation will be paused for one hour to allow it to reset.


```python
number_of_bars = {}
for county in county_names:
    try:
        response = client.venues.search({'near': county,
                                         'limit': 50,
                                         'intent': 'browse',
                                        'radius': 5000,
                                        'units': 'si',
                                        'categoryId': '4d4b7105d754a06376d81259'})
        number_of_bars[county] = len(response['venues'])
    except Exception as e:
        print (e)
        if e == "Quota exceeded":
            print ("exceeded quota: waiting for an hour")
            time.sleep(3600)
        number_of_bars[county] = -1
```

    Couldn't geocode param near: Not Applicable



```python
number_of_bars
```




    {'Allegany': 30,
     'Anne Arundel': 49,
     'Baltimore': 50,
     'Baltimore City': 50,
     'Calvert': 0,
     'Caroline': 6,
     'Carroll': 22,
     'Cecil': 28,
     'Charles': 13,
     'Dorchester': 22,
     'Frederick': 35,
     'Garrett': 12,
     'Harford': 2,
     'Howard': 9,
     'Kent': 19,
     'Montgomery': 50,
     'Not Applicable': -1,
     'Prince Georges': 29,
     'Queen Annes': 8,
     'Somerset': 41,
     'St. Marys': 13,
     'Talbot': 3,
     'Unknown': 0,
     'Washington': 50,
     'Wicomico': 50,
     'Worcester': 36}




```python
# Adding a new column for the number of bars
number_of_bars = pd.DataFrame({'county': list(number_of_bars.keys()), 'num_bars': list(number_of_bars.values())})
data_df = pd.merge(data, number_of_bars, left_on='COUNTY_NAME', right_on='county', how='left')
data_df.drop(columns=['county'], inplace=True)
```

We need to select a target variable. I will choose the 'INJURY' column, because this is a straightforward way to judge the severity of the crash.


```python
# Converting injuries to a binary mapping to judge severity
data_df['severity'] = data_df['INJURY'].map({'YES':1, 'NO':0})
```


```python
data_df.head()
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
      <th>ACC_DATE_ACC_TIME</th>
      <th>CASE_NUMBER</th>
      <th>BARRACK</th>
      <th>ACC_TIME_CODE</th>
      <th>DAY_OF_WEEK</th>
      <th>ROAD</th>
      <th>INTERSECT_ROAD</th>
      <th>DIST_FROM_INTERSECT</th>
      <th>DIST_DIRECTION</th>
      <th>CITY_NAME</th>
      <th>COUNTY_CODE</th>
      <th>COUNTY_NAME</th>
      <th>VEHICLE_COUNT</th>
      <th>PROP_DEST</th>
      <th>INJURY</th>
      <th>COLLISION_WITH_1</th>
      <th>COLLISION_WITH_2</th>
      <th>MONTH</th>
      <th>num_bars</th>
      <th>severity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-01-01 02:01:00</td>
      <td>1363000002</td>
      <td>Rockville</td>
      <td>1</td>
      <td>SUNDAY</td>
      <td>IS 00495 CAPITAL BELTWAY</td>
      <td>IS 00270 EISENHOWER MEMORIAL</td>
      <td>0.00</td>
      <td>U</td>
      <td>Not Applicable</td>
      <td>15.0</td>
      <td>Montgomery</td>
      <td>2.0</td>
      <td>YES</td>
      <td>NO</td>
      <td>VEH</td>
      <td>OTHER-COLLISION</td>
      <td>1</td>
      <td>50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012-01-01 18:01:00</td>
      <td>1296000023</td>
      <td>Berlin</td>
      <td>5</td>
      <td>SUNDAY</td>
      <td>MD 00090 OCEAN CITY EXPWY</td>
      <td>CO 00220 ST MARTINS NECK RD</td>
      <td>0.25</td>
      <td>W</td>
      <td>Not Applicable</td>
      <td>23.0</td>
      <td>Worcester</td>
      <td>1.0</td>
      <td>YES</td>
      <td>NO</td>
      <td>FIXED OBJ</td>
      <td>OTHER-COLLISION</td>
      <td>1</td>
      <td>36</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-01-01 07:01:00</td>
      <td>1283000016</td>
      <td>Prince Frederick</td>
      <td>2</td>
      <td>SUNDAY</td>
      <td>MD 00765 MAIN ST</td>
      <td>CO 00208 DUKE ST</td>
      <td>100.00</td>
      <td>S</td>
      <td>Not Applicable</td>
      <td>4.0</td>
      <td>Calvert</td>
      <td>1.0</td>
      <td>YES</td>
      <td>NO</td>
      <td>FIXED OBJ</td>
      <td>FIXED OBJ</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-01-01 00:01:00</td>
      <td>1282000006</td>
      <td>Leonardtown</td>
      <td>1</td>
      <td>SUNDAY</td>
      <td>MD 00944 MERVELL DEAN RD</td>
      <td>MD 00235 THREE NOTCH RD</td>
      <td>10.00</td>
      <td>E</td>
      <td>Not Applicable</td>
      <td>18.0</td>
      <td>St. Marys</td>
      <td>1.0</td>
      <td>YES</td>
      <td>NO</td>
      <td>FIXED OBJ</td>
      <td>OTHER-COLLISION</td>
      <td>1</td>
      <td>13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012-01-01 01:01:00</td>
      <td>1267000007</td>
      <td>Essex</td>
      <td>1</td>
      <td>SUNDAY</td>
      <td>IS 00695 BALTO BELTWAY</td>
      <td>IS 00083 HARRISBURG EXPWY</td>
      <td>100.00</td>
      <td>S</td>
      <td>Not Applicable</td>
      <td>3.0</td>
      <td>Baltimore</td>
      <td>2.0</td>
      <td>YES</td>
      <td>NO</td>
      <td>VEH</td>
      <td>OTHER-COLLISION</td>
      <td>1</td>
      <td>50</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



I will now create a new dataframe for my features and encode each feature into categoric variables.


```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# crash severity
feature_df = pd.DataFrame()
feature_df['id'] = data_df['CASE_NUMBER']
feature_df['Time'] = data_df['ACC_TIME_CODE']
feature_df['Day'] = le.fit_transform(data_df['DAY_OF_WEEK'])
feature_df['Vehicles'] = data_df['VEHICLE_COUNT'].fillna(0)
feature_df['One hit'] = le.fit_transform(data_df['COLLISION_WITH_1'])
feature_df['Tws hits'] = le.fit_transform(data_df['COLLISION_WITH_2'])
feature_df['Bars'] = data_df['num_bars']

feature_df.head()
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
      <th>id</th>
      <th>Time</th>
      <th>Day</th>
      <th>Vehicles</th>
      <th>One hit</th>
      <th>Tws hits</th>
      <th>Bars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1363000002</td>
      <td>1</td>
      <td>3</td>
      <td>2.0</td>
      <td>6</td>
      <td>4</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1296000023</td>
      <td>5</td>
      <td>3</td>
      <td>1.0</td>
      <td>2</td>
      <td>4</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1283000016</td>
      <td>2</td>
      <td>3</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1282000006</td>
      <td>1</td>
      <td>3</td>
      <td>1.0</td>
      <td>2</td>
      <td>4</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1267000007</td>
      <td>1</td>
      <td>3</td>
      <td>2.0</td>
      <td>6</td>
      <td>4</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's make sure that there are no missing values because they will cause the algorithm to crash
feature_df.isna().sum()
```

I will now use the Scikit-learn random forest classifier to fit a model and use that model to determine the importance of each feature.


```python
from sklearn.ensemble import RandomForestClassifier
# Sets up a classifier and fits a model to all features of the dataset
clf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
clf.fit(feature_df.drop(['id'],axis=1), data_df['severity'])
# We need a list of features as well
features = feature_df.drop(['id'],axis=1).columns.values
print("--- COMPLETE ---")
```

    --- COMPLETE ---


Using the following code from Anisotropic's kernal (https://www.kaggle.com/arthurtok/interactive-porto-insights-a-plot-ly-tutorial), we can use Plotly to create a nice horizontal bar chart for visualising the ranking of the most important features for determing the severity of crash.


```python
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

x, y = (list(x) for x in zip(*sorted(zip(clf.feature_importances_, features),
                                                            reverse = False)))
trace2 = go.Bar(
    x=x ,
    y=y,
    marker=dict(
        color=x,
        colorscale = 'Viridis',
        reversescale = True
    ),
    name='Random Forest Feature importance',
    orientation='v',
)

layout = dict(
    title='Ranking of most influential features',
     width = 900, height = 1500,
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
    ))

fig1 = go.Figure(data=[trace2])
fig1['layout'].update(layout)
py.iplot(fig1, filename='plots')
```

## DarkSky API

DarkSky API documentation is [here](https://darksky.net/dev/docs/time-machine)

1. Sign up for FREE api key.
2. For each crush, get the weather for the location and time.
3. Find a relationship between the weather and severity of the crash.

Hints:

* There is an API limit (perhaps 1000 calls)
* use "Time Machine" request in DarkSky API
* for sending HTTP requests check out "requests" library [here](http://docs.python-requests.org/en/master/)



The time needs to be converted to unix in order for the DarkSky API to recognise it.


```python
def time_to_unix(t):
    return time.mktime(t.timetuple())

# changing to unix time
data_sample_df = data_df.sample(n=1000, random_state=0)
data_sample_df['UNIX_TIME'] = data_sample_df['ACC_DATE_ACC_TIME'].apply(time_to_unix).astype(int)
```

We now need to get the latitude and longitude for each county in order for the API to provide local weather for each crash.


```python
def get_lat_lng(data, state, place_col):
    places = list(set(data[place_col].unique()))

    geo_dict = {'place':[], 'lat':[], 'lng':[]}
    for i in range(len(places)):
        place = places[i] + ', ' + state
        try:    
            lat = gmaps.geocode(place)[0]['geometry']['location']['lat']
            lng = gmaps.geocode(place)[0]['geometry']['location']['lng']
            geo_dict['place'].append(places[i])
            geo_dict['lat'].append(lat)
            geo_dict['lng'].append(lng)
        except:
            geo_dict['place'].append(None)
            geo_dict['lat'].append(None)
            geo_dict['lng'].append(None)

    geo_df = pd.DataFrame(geo_dict)
    return pd.merge(data, geo_df, left_on='COUNTY_NAME', right_on='place', how='left')

data_sample_geo_df = get_lat_lng(data_sample_df, 'Maryland', 'COUNTY_NAME')
```


```python
# Once you have signed up on DarkSky.net, you will be given an API key, which you need to insert here.
api_key = "[YOUR API KEY HERE]"
```

The next step is to make requests to the API for each of the crash instances in our sample. I am simply appending each entry into a dictionary with the place, coordinates, time and returned results.


```python
import requests
import time

weather_data = {'place': [], 'lat': [], 'lng': [], 'time': [], 'result': []}

for crash in data_sample_geo_df.iterrows():
    place = crash[1]['COUNTY_NAME']
    lat = crash[1]['lat']
    lng = crash[1]['lng']
    t = crash[1]['UNIX_TIME']
    # https://api.darksky.net/forecast/[key]/[latitude],[longitude],[time]
    request = 'https://api.darksky.net/forecast/' + api_key + '/' + str(lat) + ',' + str(lng) + ',' + str(t)

    try:
        result = requests.get(request).content

        weather_data['place'].append(place)
        weather_data['lat'].append(lat)
        weather_data['lng'].append(lng)
        weather_data['time'].append(t)
        weather_data['result'].append(result)
    except Exception as e:
        print(e)
        weather_data['place'].append('')
        weather_data['lat'].append('')
        weather_data['lng'].append('')
        weather_data['time'].append('')
        weather_data['result'].append('')
```


```python
weather_df = pd.concat([data_sample_geo_df[['CASE_NUMBER']], pd.DataFrame(weather_data).reset_index(drop=True)], axis=1)
# I like to save results like these in a CSV file, because there is a limit on the number of API calls
weather_df.to_csv('weather-data.csv')
```

After going through one of the JSON files that was returned to me, I found the section that I want ("currently"). It contains the overall weather conditions, precipitation type (if any) and most importantly, the chance of rain. If the chance is greater than 50%, then I am satisfied that if was raining at the time for the sake of this exercise.


```python
print(d['currently'])
```

    {'time': 1351069800, 'summary': 'Clear', 'icon': 'clear-night', 'precipIntensity': 0, 'precipProbability': 0, 'temperature': 53.74, 'apparentTemperature': 53.74, 'dewPoint': 51.68, 'humidity': 0.93, 'pressure': 1018.53, 'windSpeed': 2.68, 'windBearing': 293, 'cloudCover': 0.12, 'visibility': 6.09}



```python
import json

def extract_data(x):
    try:
        d = json.loads(x)
        res = d['currently']['precipProbability']
    except Exception as e:
        print (e)
        res = ''
    return res

weather_df['precipProb'] = weather_df['result'].apply(extract_data)
```

    Expecting value: line 1 column 1 (char 0)



```python
df_final = data_sample_df.merge(weather_df, left_on='CASE_NUMBER', right_on='CASE_NUMBER', how='outer')
```


```python
df_final[['severity', 'precipProb']].corr()
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
      <th>severity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>severity</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_sample_df.head()
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
      <th>ACC_DATE_ACC_TIME</th>
      <th>CASE_NUMBER</th>
      <th>BARRACK</th>
      <th>ACC_TIME_CODE</th>
      <th>DAY_OF_WEEK</th>
      <th>ROAD</th>
      <th>INTERSECT_ROAD</th>
      <th>DIST_FROM_INTERSECT</th>
      <th>DIST_DIRECTION</th>
      <th>CITY_NAME</th>
      <th>COUNTY_CODE</th>
      <th>COUNTY_NAME</th>
      <th>VEHICLE_COUNT</th>
      <th>PROP_DEST</th>
      <th>INJURY</th>
      <th>COLLISION_WITH_1</th>
      <th>COLLISION_WITH_2</th>
      <th>MONTH</th>
      <th>UNIX_TIME</th>
      <th>severity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14950</th>
      <td>2012-10-24 11:10:00</td>
      <td>1251037421</td>
      <td>Frederick</td>
      <td>3</td>
      <td>WEDNESDAY</td>
      <td>IS 00270 EISEN MEM HWY</td>
      <td>MD 00080 FINGERBOARD RD</td>
      <td>500.0</td>
      <td>N</td>
      <td>Not Applicable</td>
      <td>10.0</td>
      <td>Frederick</td>
      <td>2.0</td>
      <td>NO</td>
      <td>YES</td>
      <td>VEH</td>
      <td>OTHER-COLLISION</td>
      <td>10</td>
      <td>1351069800</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11152</th>
      <td>2012-08-13 20:08:00</td>
      <td>1251029176</td>
      <td>Frederick</td>
      <td>6</td>
      <td>MONDAY</td>
      <td>MD 00026 Liberty Rd</td>
      <td>MU 00998 Monocacy Blvd</td>
      <td>0.0</td>
      <td>U</td>
      <td>Not Applicable</td>
      <td>10.0</td>
      <td>Frederick</td>
      <td>2.0</td>
      <td>YES</td>
      <td>NO</td>
      <td>VEH</td>
      <td>OTHER-COLLISION</td>
      <td>8</td>
      <td>1344881280</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13808</th>
      <td>2012-10-05 17:10:00</td>
      <td>1280008862</td>
      <td>Centreville</td>
      <td>5</td>
      <td>FRIDAY</td>
      <td>US 00301 BLUE STAR MEMORIAL</td>
      <td>CO 00151 JOHN BROWN RD</td>
      <td>60.0</td>
      <td>S</td>
      <td>Not Applicable</td>
      <td>17.0</td>
      <td>Queen Annes</td>
      <td>2.0</td>
      <td>YES</td>
      <td>NO</td>
      <td>VEH</td>
      <td>VEH</td>
      <td>10</td>
      <td>1349449800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11261</th>
      <td>2012-08-16 05:08:00</td>
      <td>1294006103</td>
      <td>McHenry</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>CO 00172 BUMBLE BEE RD</td>
      <td>CO 00173 SPEAR RD</td>
      <td>300.0</td>
      <td>W</td>
      <td>Not Applicable</td>
      <td>11.0</td>
      <td>Garrett</td>
      <td>1.0</td>
      <td>NO</td>
      <td>YES</td>
      <td>OTHER-COLLISION</td>
      <td>FIXED OBJ</td>
      <td>8</td>
      <td>1345086480</td>
      <td>1</td>
    </tr>
    <tr>
      <th>875</th>
      <td>2012-01-20 22:01:00</td>
      <td>1283001115</td>
      <td>Prince Frederick</td>
      <td>6</td>
      <td>FRIDAY</td>
      <td>CO 00058 GRAYS RD</td>
      <td>MD 00506 SIXES RD</td>
      <td>1.0</td>
      <td>E</td>
      <td>Not Applicable</td>
      <td>4.0</td>
      <td>Calvert</td>
      <td>1.0</td>
      <td>YES</td>
      <td>NO</td>
      <td>FIXED OBJ</td>
      <td>NON-COLLISION</td>
      <td>1</td>
      <td>1327093260</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_sample_geo_df['crash_lat'] = gmaps.geocode(data_sample_geo_df['ROAD'].values[0] + ' ' + data_sample_geo_df['INTERSECT_ROAD'].values[0])[0]['geometry']['location']['lat']


gmaps.geocode(data_sample_geo_df['ROAD'] + ' ' + data_sample_geo_df['INTERSECT_ROAD'])[0]['geometry']['location']['lat']
```


```python
df_intersections = data_sample_geo_df[['ROAD','INTERSECT_ROAD']]
```


```python
def get_coords(x):    
    return x[0] + x[1]  # gmaps.geocode(x[0] + ' ' + x[1])[0]['geometry']['location']['lat']

data_sample_geo_df['intersections'] = data_sample_geo_df[['ROAD','INTERSECT_ROAD']].apply(get_coords, axis=1)
```


```python
data_sample_geo_df.shape
```




    (1000, 24)




```python
def get_lat(x):
    try:
        return gmaps.geocode(x)[0]['geometry']['location']['lat']
    except:
        return ''

def get_lng(x):
    try:
        return gmaps.geocode(x)[0]['geometry']['location']['lng']
    except:
        return ''

data_sample_geo_df['crash_lat'] = data_sample_geo_df['intersections'].apply(get_lat)
data_sample_geo_df['crash_lng'] = data_sample_geo_df['intersections'].apply(get_lng)
```


```python
from gmplot import gmplot

# Coordinates for Maryland
state_lat = gmaps.geocode('Maryland')[0]['geometry']['location']['lat']
state_lng = gmaps.geocode('Maryland')[0]['geometry']['location']['lng']

# Place map
gmap = gmplot.GoogleMapPlotter(state_lat, state_lng, 7)

# Coordinates for Counties
county_lats = data_sample_geo_df['lat'].values.tolist()
county_lngs = data_sample_geo_df['lng'].values.tolist()

# Coordinates for crashes
crash_lats = data_sample_geo_df['crash_lat'].values.tolist()
crash_lngs = data_sample_geo_df['crash_lng'].values.tolist()

# Scatter points
gmap.scatter(county_lats, county_lngs, 'blue', size=1000, marker=False)
gmap.scatter(crash_lats, crash_lngs, 'red', size=2500, marker=True)

# Draw
gmap.draw("my_map.html")
```
