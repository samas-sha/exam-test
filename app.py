from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load voting data
VOTING_DATA_PATH = "voting_data.csv"
if os.path.exists(VOTING_DATA_PATH):
    voting_df = pd.read_csv(VOTING_DATA_PATH)
else:
    voting_df = None

MODEL_PATH = "model.pkl"

# Train and save the model if it doesn't exist
def train_and_save_model():
    """Train a GradientBoostingRegressor model and save it"""
    # Create synthetic training data (5 features: energy, danceability, tempo, acousticness, valence)
    X_train, y_train = make_regression(n_samples=100, n_features=5, random_state=42)
    
    # Train the model
    model = GradientBoostingRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    
    # Save the trained model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model trained and saved to {MODEL_PATH}")
    return model

# Load the model on startup
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {MODEL_PATH}")
else:
    print("Model not found. Training new model...")
    model = train_and_save_model()

@app.route('/', methods=['GET'])
def index():
    """Serve the main frontend page"""
    return render_template('index.html')

@app.route('/api', methods=['GET'])
def api_docs():
    """API documentation endpoint"""
    return jsonify({
        "message": "Flask Prediction API",
        "endpoints": {
            "GET /": "Frontend application",
            "POST /predict": "Make predictions with 5 features",
            "GET /health": "Health check",
            "GET /api": "API documentation"
        },
        "predict_example": {
            "energy": 0.8,
            "danceability": 0.7,
            "tempo": 120,
            "acousticness": 0.3,
            "valence": 0.6
        }
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint that accepts JSON input with 5 features.
    
    Expected JSON input:
    {
        "energy": float,
        "danceability": float,
        "tempo": float,
        "acousticness": float,
        "valence": float
    }
    
    Returns:
    {
        "predicted_points": float
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Extract features in the correct order
        features = [
            data.get("energy", 0),
            data.get("danceability", 0),
            data.get("tempo", 0),
            data.get("acousticness", 0),
            data.get("valence", 0)
        ]
        
        # Validate that we have valid numeric inputs
        features_array = np.array(features, dtype=float).reshape(1, -1)
        
        # Make prediction using the loaded model
        prediction = model.predict(features_array)[0]
        
        # Return prediction as JSON
        return jsonify({"predicted_points": float(prediction)})
    
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": "loaded"}), 200

@app.route('/voting-analysis')
def voting_analysis():
    """Voting analysis page"""
    if voting_df is None:
        countries = []
    else:
        countries = sorted(voting_df['Receiving_Country'].unique().tolist())
    
    return render_template('voting.html', countries=countries)

@app.route('/api/voting-analysis', methods=['POST'])
def api_voting_analysis():
    """API endpoint for voting analysis"""
    try:
        data = request.get_json()
        country = data.get('country')
        
        if voting_df is None:
            return jsonify({"error": "Voting data not available"}), 500
        
        if not country:
            return jsonify({"error": "Country parameter required"}), 400
        
        # Filter votes received by the selected country
        country_votes = voting_df[voting_df['Receiving_Country'] == country]
        
        if country_votes.empty:
            return jsonify({"error": f"No voting data found for {country}"}), 404
        
        # Group by voting country and sum points
        votes_by_country = country_votes.groupby('Voting_Country')['Points'].sum().reset_index()
        votes_by_country = votes_by_country.sort_values('Points', ascending=False)
        
        # Get the country with most points
        if not votes_by_country.empty:
            top_country = votes_by_country.iloc[0]
            result = {
                "selected_country": country,
                "top_country": top_country['Voting_Country'],
                "points": int(top_country['Points']),
                "message": f"The country that has given the most points to {country} is {top_country['Voting_Country']}!",
                "all_votes": votes_by_country.to_dict('records')
            }
            return jsonify(result), 200
        else:
            return jsonify({"error": "No voting data found"}), 404
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)