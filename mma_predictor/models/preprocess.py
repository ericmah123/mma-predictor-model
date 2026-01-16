from datetime import datetime, timezone
import numpy as np
import pandas as pd

def load_and_clean_data(filepath, simulate_outcomes=False):
    data = pd.read_csv(filepath)
    
    # Drop rows with missing values in essential columns
    data = data.dropna(subset=['height_cm', 'reach_in_cm', 'significant_strikes_landed_per_minute', 'average_takedowns_landed_per_15_minutes'])

    # Convert height and reach from cm to inches (1 cm = 0.393701 inches)
    data['height'] = data['height_cm'] * 0.393701
    data['reach'] = data['reach_in_cm'] * 0.393701
    
    # Convert weight from kg to pounds (1 kg = 2.20462 pounds)
    data['Weight'] = data['weight_in_kg'] * 2.20462

    # Calculate age based on current date
    current_year = datetime.now(timezone.utc).year
    data['age'] = current_year - pd.to_datetime(data['date_of_birth']).dt.year

    # Encode stance using one-hot encoding
    data = pd.get_dummies(data, columns=['stance'], drop_first=False)
    for stance_column in ['stance_Orthodox', 'stance_Southpaw']:
        if stance_column not in data.columns:
            data[stance_column] = 0

    # Create new features
    data['win_loss_ratio'] = data['wins'] / (data['losses'] + 1)
    data['experience'] = data['wins'] + data['losses'] + data['draws']
    
    # Select relevant columns
    relevant_columns = [
        'height', 'Weight', 'reach', 'age',
        'significant_strikes_landed_per_minute', 
        'average_takedowns_landed_per_15_minutes', 
        'win_loss_ratio', 'experience'
    ] + [col for col in data.columns if col.startswith('stance_')]
    data = data[relevant_columns]

    # Drop rows with any remaining missing values
    data = data.dropna()

    if 'Outcome' in data.columns:
        fight_df = data[relevant_columns + ['Outcome']].dropna()
        return fight_df

    if not simulate_outcomes:
        raise ValueError(
            "Outcome column not found. Provide labeled fight outcomes or set "
            "simulate_outcomes=True for synthetic labels."
        )

    # Simulate fight outcomes by pairing fighters with a noisy strength score
    rng = np.random.default_rng(42)
    shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
    score_columns = [
        'win_loss_ratio',
        'significant_strikes_landed_per_minute',
        'average_takedowns_landed_per_15_minutes',
        'experience'
    ]
    scaled_scores = (shuffled[score_columns] - shuffled[score_columns].min()) / (
        shuffled[score_columns].max() - shuffled[score_columns].min()
    ).replace(0, 1)
    shuffled['strength_score'] = scaled_scores.mean(axis=1)

    fight_data = []
    for i in range(0, len(shuffled) - 1, 2):
        fighter_1 = shuffled.iloc[i]
        fighter_2 = shuffled.iloc[i + 1]

        total_strength = fighter_1['strength_score'] + fighter_2['strength_score']
        win_prob = fighter_1['strength_score'] / total_strength if total_strength else 0.5
        outcome_1 = int(rng.random() < win_prob)
        outcome_2 = 1 - outcome_1

        fight_data.append(fighter_1.drop(labels=['strength_score']).to_dict())
        fight_data[-1]['Outcome'] = outcome_1
        fight_data.append(fighter_2.drop(labels=['strength_score']).to_dict())
        fight_data[-1]['Outcome'] = outcome_2
    
    fight_df = pd.DataFrame(fight_data)
    
    return fight_df
