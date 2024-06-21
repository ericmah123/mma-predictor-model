import tkinter as tk
from tkinter import messagebox, ttk
import joblib
import pandas as pd
import os

class FightPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MMA Fight Predictor")

        # Load the trained model and scaler
        model_path = os.path.join(os.path.dirname(__file__), 'mma_fight_predictor.pkl')
        self.model_pipeline = joblib.load(model_path)

        # Create input fields for Fighter 1
        self.create_input_fields("Fighter 1", 0)
        
        # Create input fields for Fighter 2
        self.create_input_fields("Fighter 2", 11)

        # Create predict button
        predict_button = tk.Button(root, text="Predict", command=self.predict)
        predict_button.grid(row=22, column=0, columnspan=2, pady=10)

    def create_input_fields(self, label, start_row):
        tk.Label(self.root, text=label).grid(row=start_row, column=0, columnspan=2, pady=5)

        attributes = [
            "Height (in inches):", "Weight (in pounds):", "Reach (in inches):", 
            "Age:", "Significant Strikes Landed Per Minute:", 
            "Average Takedowns Landed Per 15 Minutes:", "Win/Loss Ratio:", 
            "Experience:", "Stance (Orthodox=0, Southpaw=1):"
        ]
        
        self.entries = getattr(self, 'entries', [])
        current_entries = []
        
        for i, attr in enumerate(attributes):
            tk.Label(self.root, text=attr).grid(row=start_row + i + 1, column=0, sticky='e', padx=5, pady=2)
            entry = tk.Entry(self.root)
            entry.grid(row=start_row + i + 1, column=1, padx=5, pady=2)
            current_entries.append(entry)
        
        self.entries.append(current_entries)

    def predict(self):
        try:
            # Retrieve the inputs
            fighter_1_stats = []
            fighter_2_stats = []

            # Collect inputs for Fighter 1
            for entry in self.entries[0]:
                value = entry.get()
                if not value:
                    raise ValueError("All fields must be filled.")
                fighter_1_stats.append(float(value))

            # Collect inputs for Fighter 2
            for entry in self.entries[1]:
                value = entry.get()
                if not value:
                    raise ValueError("All fields must be filled.")
                fighter_2_stats.append(float(value))

            # Debugging: Print the stats to ensure they're being collected correctly
            print("Fighter 1 Stats:", fighter_1_stats)
            print("Fighter 2 Stats:", fighter_2_stats)

            # Split stance into two columns: stance_Orthodox and stance_Southpaw
            fighter_1_stats = fighter_1_stats[:-1] + [1, 0] if fighter_1_stats[-1] == 0 else fighter_1_stats[:-1] + [0, 1]
            fighter_2_stats = fighter_2_stats[:-1] + [1, 0] if fighter_2_stats[-1] == 0 else fighter_2_stats[:-1] + [0, 1]

            # Debugging: Print the updated stats to ensure the stance is correctly split
            print("Updated Fighter 1 Stats:", fighter_1_stats)
            print("Updated Fighter 2 Stats:", fighter_2_stats)

            # Define feature names
            features = [
                'height', 'Weight', 'reach', 'age',
                'significant_strikes_landed_per_minute', 
                'average_takedowns_landed_per_15_minutes', 
                'win_loss_ratio', 'experience',
                'stance_Orthodox', 'stance_Southpaw'
            ]
            
            # Create the input DataFrame for the prediction
            fighter_1_df = pd.DataFrame([fighter_1_stats], columns=features)
            fighter_2_df = pd.DataFrame([fighter_2_stats], columns=features)

            # Predict the outcome for both fighters using the entire pipeline
            fighter_1_prediction = self.model_pipeline.predict_proba(fighter_1_df)[0][1]
            fighter_2_prediction = self.model_pipeline.predict_proba(fighter_2_df)[0][1]

            # Calculate the prediction odds
            fighter_1_odds = fighter_1_prediction / (fighter_1_prediction + fighter_2_prediction)
            fighter_2_odds = fighter_2_prediction / (fighter_1_prediction + fighter_2_prediction)

            # Display the result
            result = f"Fighter 1 Win Probability: {fighter_1_odds * 100:.2f}%\nFighter 2 Win Probability: {fighter_2_odds * 100:.2f}%"
            messagebox.showinfo("Prediction Result", result)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = FightPredictorApp(root)
    root.mainloop()
