from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__, static_folder='static', template_folder='templates')

model_pipeline = joblib.load('mma_predictor/models/mma_fight_predictor.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        fighter_1_stats = [
            float(request.form['height_1']),
            float(request.form['weight_1']),
            float(request.form['reach_1']),
            float(request.form['age_1']),
            float(request.form['strikes_1']),
            float(request.form['takedowns_1']),
            float(request.form['ratio_1']),
            float(request.form['experience_1']),
            1 if request.form['stance_1'] == 'Orthodox' else 0,
            0 if request.form['stance_1'] == 'Orthodox' else 1
        ]
        
        fighter_2_stats = [
            float(request.form['height_2']),
            float(request.form['weight_2']),
            float(request.form['reach_2']),
            float(request.form['age_2']),
            float(request.form['strikes_2']),
            float(request.form['takedowns_2']),
            float(request.form['ratio_2']),
            float(request.form['experience_2']),
            1 if request.form['stance_2'] == 'Orthodox' else 0,
            0 if request.form['stance_2'] == 'Orthodox' else 1
        ]
        
        fighter_1_odds, fighter_2_odds = predict_fight(fighter_1_stats, fighter_2_stats)
        return render_template('index.html', fighter_1_odds=fighter_1_odds, fighter_2_odds=fighter_2_odds)
    return render_template('index.html', fighter_1_odds=None, fighter_2_odds=None)

def predict_fight(fighter_1_stats, fighter_2_stats):
    features = [
        'height', 'Weight', 'reach', 'age',
        'significant_strikes_landed_per_minute', 
        'average_takedowns_landed_per_15_minutes', 
        'win_loss_ratio', 'experience',
        'stance_Orthodox', 'stance_Southpaw'
    ]

    fighter_1_df = pd.DataFrame([fighter_1_stats], columns=features)
    fighter_2_df = pd.DataFrame([fighter_2_stats], columns=features)

    fighter_1_poly = model_pipeline.named_steps['poly'].transform(fighter_1_df)
    fighter_2_poly = model_pipeline.named_steps['poly'].transform(fighter_2_df)

    fighter_1_scaled = model_pipeline.named_steps['scaler'].transform(fighter_1_poly)
    fighter_2_scaled = model_pipeline.named_steps['scaler'].transform(fighter_2_poly)

    fighter_1_prediction = model_pipeline.named_steps['model'].predict_proba(fighter_1_scaled)[0][1]
    fighter_2_prediction = model_pipeline.named_steps['model'].predict_proba(fighter_2_scaled)[0][1]

    fighter_1_odds = fighter_1_prediction / (fighter_1_prediction + fighter_2_prediction)
    fighter_2_odds = fighter_2_prediction / (fighter_1_prediction + fighter_2_prediction)

    return fighter_1_odds, fighter_2_odds

if __name__ == '__main__':
    app.run(debug=True)
