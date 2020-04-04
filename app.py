from flask import Flask, redirect, url_for, render_template, request,jsonify
import numpy as np
import json
import os
os.environ["FLASK_ENV"]="deployment"

app = Flask(__name__, static_folder='static', static_url_path='')

def get_num_points(num_batch):
    return (num_batch+1)*2

def dumb_model():
   
    # dictionary={}
    # num_batch=80
    # num_points=3
    # for i in range(num_batch):
    #     dictionary["Num_Batch_"+str(i)]={}
    #     num_points=get_num_points(i)
    #     for j in range(num_points):
    #         dictionary["Num_Batch_"+str(i)]["Num_Point_"+str(j)]={}
    #         dictionary["Num_Batch_"+str(i)]["Num_Point_"+str(j)]={}
    # for i in range(num_batch):
    #     num_points=get_num_points(i)
    #     for j in range(num_points):
    #         point=np.random.rand(2,1)
    #         dictionary["Num_Batch_"+str(i)]["Num_Point_"+str(j)]={'x1':point[0][0],'x2':point[1][0]}
    # json_data=json.dumps(dictionary)
    with open('sim2.json') as json_file:
        json_data = json.load(json_file)
    return json_data
    



@app.route("/", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        user = request.form["nm"]
        percentage_aux = request.form["percentage"]
        capacity_aux=request.form["capacity"]
        min_infected_aux=request.form["min_infected"]
        json_data=dumb_model()
        Susceptible=json_data['0']['S']


        return render_template('index.html',data_aux=json_data,susceptible_cat=Susceptible)
    else:
        return render_template("inputs.html")




if __name__ == "__main__":
    app.run(debug=False)