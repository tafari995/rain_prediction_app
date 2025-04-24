

import openmeteo_requests
import requests_cache
from retry_requests import retry

import pandas as pd
from pandas import read_sql
from numpy import array, flip
from sqlalchemy import create_engine 
from rainpred import rpred

mysql_conn_engine = create_engine('mysql+mysqldb://<username>:<password>@localhost/weather')
connection = mysql_conn_engine.connect()

#Dates are YYYY-MM-DD for open meteo

def get_weather(city_lat, city_long, s_date, e_date):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below    
    url = "https://api.open-meteo.com/v1/forecast"
    parameters = {
        "latitude": float(city_lat),
        "longitude": float(city_long),
        "start_date": s_date,
        "end_date": e_date,        
        "hourly": ["temperature_2m", 
                   "relative_humidity_2m", 
                   "surface_pressure", 
                   "wind_speed_10m", 
                   "rain"
                   ],
        "temperature_unit":"fahrenheit",  
        "timezone":"auto"
    }
    responses = openmeteo.weather_api(url, params=parameters)
    weather = responses[0]
    elevation = weather.Elevation()
    timezone = weather.Timezone()
    hourly = weather.Hourly()    
    hourly_temperature_2m =hourly.Variables(0).ValuesAsNumpy()
    hourly_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(2).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()
    hourly_rain = hourly.Variables(4).ValuesAsNumpy()
    
    hourly_data={}   
    hourly_data["dates"] = pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval() ),
        inclusive = "left"
    )  
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["humidity_2m"] = hourly_humidity_2m
    hourly_data["surface_pressure"] = hourly_surface_pressure
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["rain"] = hourly_rain
    
    hourly_df=pd.DataFrame(data=hourly_data)
    hourly_df["dates"] = hourly_df["dates"].dt.strftime('%Y-%m-%d %H')
    hourly_df.dropna()
    hourly_df.to_sql(
        name = 'hourly_weather',
        con = connection,
        schema = None,
        if_exists = 'replace',
        index = False
    )
    print("requested data saved to SQL table :D")
    return elevation, timezone
    
    
'''
You can likely replace the two functions below 
by simply reading sql directly in the Flask application.
Working this way requires significantly less pandas,
but we will at some point either have to save or immediately
return the rain prediction 
'''


def get_plottable_hourly_weather():
    hourly_df = read_sql(
        sql = f""" 
               SELECT
               ROW_NUMBER()  OVER (ORDER BY dates) AS counter,
               temperature_2m, 
               humidity_2m, 
               surface_pressure, 
               wind_speed_10m 
               FROM hourly_weather
               ORDER BY dates ASC              
               """,
        con = connection
    )  
    hourly_df.index = hourly_df["counter"]  
    return hourly_df
    

    
def get_rain_prediction():
    hourly_df = read_sql(
        sql = f""" 
               SELECT
               temperature_2m, 
               humidity_2m, 
               surface_pressure, 
               wind_speed_10m,
               rain 
               FROM hourly_weather
               ORDER BY dates ASC              
               """,
        con = connection
    )  
    hrs_forward = 72
    wind_sz = 8
    batch_sz = 33
    homemade_rain_prediction = pd.DataFrame(
        data = rpred(
            hourly_df[:hourly_df.shape[0]-hrs_forward],
            wind_sz,
            hrs_forward,
            batch_sz
        )
    )
    pred_length = homemade_rain_prediction.shape[0]
    resized_rain = hourly_df["rain"].iloc[-pred_length:]
    resized_rain.index = range(pred_length)
    return resized_rain, homemade_rain_prediction
     

'''
weather=get_weather(city_lat, city_long, s_date, e_date)[0]
hourly=weather.Hourly()

hourly_temperature_2m=hourly.Variables(0).ValuesAsNumpy()
hourly_humidity_2m=hourly.Variables(1).ValuesAsNumpy()
hourly_surface_pressure=hourly.Variables(2).ValuesAsNumpy()
hourly_wind_speed_10m=hourly.Variables(3).ValuesAsNumpy()

L=hourly_wind_speed_10m.size    
hour_counter=array([i for i in range(L)])    
hourly_data={}


hourly_data["counter"] = hour_counter 
hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["humidity_2m"] = hourly_humidity_2m
hourly_data["surface_pressure"] = hourly_surface_pressure
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m

hourly_df=pd.DataFrame(data=hourly_data, index=hourly_data["counter"] )
'''




