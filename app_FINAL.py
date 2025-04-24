

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
#import matplotlib.dates as mdates

import io
from flask import Flask, render_template, request, Response

from weather_FINAL import get_weather, get_plottable_hourly_weather,get_rain_prediction
from CityDataStructure_FINAL import CityDataEntry

'''
Use 'python app_FINAL.py' in weather app folder to run
'''

app = Flask(__name__)

city_data = CityDataEntry()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/weather', methods=['GET','POST'])
def weather():
    if request.method == 'GET':
        city_lat = city_data.data["city_lat"]
        city_long = city_data.data["city_long"]
        elevation = city_data.data["elevation"]
        timezone = city_data.data["timezone"]        
        return render_template(
            'weather.html', 
            lat=city_lat, 
            long=city_long, 
            elevation=elevation, 
            timezone=timezone
        )
    city_lat = request.form['latitude']
    city_long = request.form['longitude']
    s_date = request.form['start date'] 
    e_date = request.form['end date']
    
    elevation, timezone = get_weather(city_lat, city_long, s_date, e_date)
    city_data.populate((elevation, timezone, city_lat, city_long, s_date, e_date))
    try:
        return render_template(
            'weather.html', 
            lat=city_lat, 
            long=city_long, 
            elevation=elevation, 
            timezone=timezone) 
    except Error:
        return render_template('index.html', error="Location not found!")
        


@app.route('/weather/rain_prediction', methods=['GET','POST'])        
def rain_prediction():
    city_lat, city_long = city_data.data["city_lat"], city_data.data["city_long"]
    e_date = city_data.data["e_date"]
    
    try:
        return render_template('rain_prediction.html', 
                        city_lat=city_lat, 
                        city_long=city_long,
                        e_date=e_date
                        )
    except Error:
        return render_template('index.html', error="City not found!")
    
    
@app.route('/weather/rain_prediction/r_pred_plot')
def r_pred_plot():
    resized_rain, home_rain_pred = get_rain_prediction()
    fig = Figure()
    axis=fig.add_subplot(1, 1, 1)
    axis.plot(resized_rain.index.values,resized_rain)
    axis.plot(resized_rain.index.values,home_rain_pred)    
    output=io.BytesIO()
    FigureCanvas(fig).print_png(output)        
    return Response(output.getvalue(), mimetype='image/png')
    

@app.route('/weather/weather_plots', methods=['GET','POST'])
def weather_plots():
    city_lat, city_long = city_data.data["city_lat"], city_data.data["city_long"]
    s_date, e_date = city_data.data["s_date"], city_data.data["e_date"]
    try:
        return render_template('weather_plots.html', 
                               city_lat=city_lat, 
                               city_long=city_long, 
                               s_date=s_date, 
                               e_date=e_date
                               )        
    except Error:
        return render_template('index.html', error="City not found!")

@app.route('/weather/weather_plots/w_plot/<city_lat>/<city_long>/<s_date>/<e_date>')        
def w_plot(city_lat, city_long, s_date, e_date):
    hourly_df = get_plottable_hourly_weather()
    
    fig = Figure()
    axis1 = fig.add_subplot(2, 2, 1)
    axis1.plot(hourly_df.index.values,hourly_df["temperature_2m"],label="temperature_2m")
    axis2 = fig.add_subplot(2, 2, 2)
    axis2.plot(hourly_df.index.values,hourly_df["humidity_2m"],label="humidity_2m", color="green")
    axis3 = fig.add_subplot(2, 2, 3)
    axis3.plot(hourly_df.index.values,hourly_df["surface_pressure"],label="surface_pressure", color="purple")
    axis4 = fig.add_subplot(2, 2, 4)
    axis4.plot(hourly_df.index.values,hourly_df["wind_speed_10m"],label="wind_speed_10m", color="red")
    
    axis1.set_title("Temp_2m")
    axis2.set_title("Hum_2m")
    axis3.set_title("S_Presh")
    axis4.set_title("Wind_10m")
    
    axis1.set_ylabel("F")
    axis2.set_ylabel("relative %")
    axis3.set_ylabel("hPa")
    axis4.set_ylabel("km/h")
    
    fig.subplots_adjust(hspace=0.35, wspace=0.4)
        
    output=io.BytesIO()
    FigureCanvas(fig).print_png(output)
        
    return Response(output.getvalue(), mimetype='image/png')
        
    


if __name__ == '__main__':
    app.run(debug=True)






