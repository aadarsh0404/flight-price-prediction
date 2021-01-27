from flask import Flask, request, render_template
import jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
data = pd.read_excel('train.xlsx')
model = pickle.load(open('model.pkl','rb'))
data.Additional_Info = np.where(data.Additional_Info=='No Info','No info',data.Additional_Info)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        """Airlines"""
        airline = list(data.Airline.unique())
        airline.remove('Air Asia')
        airline_onehot = [0]*len(airline)
        airlines = dict(zip(airline, airline_onehot))
        Airline = request.form['Airline']
        if Airline in airlines:
            airlines[Airline] = 1
            
        """Source"""
        source = list(data.Source.unique())
        source.remove('Banglore')
        source_onehot = [0]*len(source)
        sources = dict(zip(source, source_onehot))
        Source = request.form['Source']
        if Source in sources:
            sources[Source] = 1
        
        """Destination"""
        destination = list(data.Destination.unique())
        destination.remove('Banglore')
        destination_onehot = [0]*len(destination)
        destinations = dict(zip(destination, destination_onehot))
        Destination = request.form['Destination']
        if Destination in destinations:
            destinations[Destination] = 1
        
        """Arrival Time"""
        Arrival_Time = request.form['Arrival_Time']
        Arrival_hour = pd.to_datetime(Arrival_Time).hour
        Arrival_min = pd.to_datetime(Arrival_Time).minute
        
        """Departure Time"""
        Dep_Time = request.form['Dep_Time']
        Dep_hr = pd.to_datetime(Dep_Time).hour
        Dep_min = pd.to_datetime(Dep_Time).minute
        
        """Date of Journey"""
        Date_of_Journey = request.form['Date_of_Journey']
        Journey_Day = pd.to_datetime(Date_of_Journey).day
        Journey_Month = pd.to_datetime(Date_of_Journey).month
        
        """Duration"""
        Duration = request.form['Duration']
        hr = 0
        Duration = Duration.split(' ')
        if len(Duration)==2:
            hr = hr + float(Duration[0][:-1])+(float(Duration[1][:-1])/60)
        else:
            hr = float(Duration[0][:-1])
            
        """Total Stops"""
        Total_Stops = request.form['Total_Stops']
        stops = 0
        Total_Stops = Total_Stops.split(' ')
        if len(Total_Stops)==2:
            stops = int(Total_Stops[0])
        else:
            stops = 0
            
        """Additional Info"""
        additional_info = list(data.Additional_Info.unique())
        additional_info.remove('1 Long layover')
        additional_info_onehot = [0]*len(additional_info)
        additional_infos = dict(zip(additional_info, additional_info_onehot))
        Additional_Info = request.form['Additional_Info']
        if Additional_Info in additional_infos:
            additional_infos[Additional_Info] = 1
        cols = np.array(model.get_booster().feature_names)
        values = np.array([hr, stops, Journey_Day, Journey_Month, sources['Chennai'], sources['Delhi'], sources['Kolkata'], sources['Mumbai'], destinations['Cochin'],destinations['Delhi'], destinations['Hyderabad'], destinations['Kolkata'], Dep_hr, Dep_min, Arrival_hour, Arrival_min, airlines['Air India'], airlines['GoAir'], airlines['IndiGo'], airlines['Jet Airways'], airlines['Jet Airways Business'], airlines['Multiple carriers'], airlines['Multiple carriers Premium economy'], airlines['SpiceJet'], airlines['Trujet'], airlines['Vistara'], airlines['Vistara Premium economy'], additional_infos['1 Short layover'], additional_infos['2 Long layover'], additional_infos['Business class'], additional_infos['Change airports'], additional_infos['In-flight meal not included'], additional_infos['No check-in baggage included'], additional_infos['No info'], additional_infos['Red-eye flight']])
        df = pd.DataFrame(values.reshape(1, 35), columns = cols)
        pred = model.predict(df)
        return render_template('index.html', prediction_text=pred[0])
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run()
