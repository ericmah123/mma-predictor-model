import pandas as pd

def load_and_clean_data(filepath):
    data = pd.read_csv(filepath)
    
    # Drop rows with missing values in essential columns
    data = data.dropna(subset=['height_cm', 'reach_in_cm', 'significant_strikes_landed_per_minute', 'average_takedowns_landed_per_15_minutes'])

    # Convert height and reach from cm to inches (1 cm = 0.393701 inches)
    data['height'] = data['height_cm'] * 0.393701
    data['reach'] = data['reach_in_cm'] * 0.393701
    
    # Convert weight from kg to pounds (1 kg = 2.20462 pounds)
    data['Weight'] = data['weight_in_kg'] * 2.20462

    # Calculate age
    data['age'] = 2024 - pd.to_datetime(data['date_of_birth']).dt.year

    # Encode stance using one-hot encoding
    data = pd.get_dummies(data, columns=['stance'], drop_first=True)

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

    # Simulate fight outcomes by pairing fighters
    fight_data = []
    for i in range(0, len(data) - 1, 2):
        fighter_1 = data.iloc[i]
        fighter_2 = data.iloc[i + 1]
        
        # Simple rule-based outcome: higher win/loss ratio wins
        if fighter_1['win_loss_ratio'] > fighter_2['win_loss_ratio']:
            outcome_1 = 1
            outcome_2 = 0
        else:
            outcome_1 = 0
            outcome_2 = 1
        
        fight_data.append(fighter_1.to_dict())
        fight_data[-1]['Outcome'] = outcome_1
        fight_data.append(fighter_2.to_dict())
        fight_data[-1]['Outcome'] = outcome_2
    
    fight_df = pd.DataFrame(fight_data)
    
    return fight_df
