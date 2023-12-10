from flask import Flask, render_template, request
from RNNPredict import LSTM_Model

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def hoge():
    if request.method == "GET":
        return render_template("index2.html")

    if request.method == "POST":
        parameter = {}
        

        parameter["tau"] = request.form["tau"]
        t = int(parameter["tau"])

        parameter["predict"] = request.form["predict"]
        stp = int(parameter["predict"])
        

        csv_file = request.files["csv_file"].read().decode("SHIFT-JIS").split(",")
        
        Y = list(map(float,csv_file))

        calculator = LSTM_Model(Y,t,stp)
        calculator.model_predict()


        return render_template("result_2.html",parameter=parameter)
    else:
        return "ERROR"


    

