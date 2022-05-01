from flask import Flask, render_template, request
import pickle
import sklearn

app = Flask(__name__)

pickle_in = open('RandomForest.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        national_inv = request.form['national_inv']
        lead_time = request.form['lead_time']
        sales_1_month = request.form['sales_1_month']
        pieces_past_due = request.form['pieces_past_due']
        perf_6_month_avg = request.form['perf_6_month_avg']
        local_bo_qty = request.form['local_bo_qty']
        deck_risk = request.form['deck_risk']
        oe_constraint = request.form['oe_constraint']
        ppap_risk = request.form['ppap_risk']
        stop_auto_buy = request.form['stop_auto_buy']
        rev_stop = request.form['rev_stop']

        prediction = classifier.predict([[national_inv, lead_time, sales_1_month, pieces_past_due, perf_6_month_avg, local_bo_qty, deck_risk, oe_constraint, ppap_risk, stop_auto_buy, rev_stop]])

        return render_template('success.html', prediction=prediction)

if __name__ == '__main__':
    app.debug = True
    app.run()