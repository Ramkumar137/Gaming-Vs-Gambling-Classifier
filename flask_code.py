from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
import typing_extensions as typing
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure Gemini AI
genai.configure(api_key="Api-key-here")
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(
    history=[
        {"role": "user", "parts": "Hello"},
        {"role": "model", "parts": "Great to meet you. What would you like to know?"},
    ]
)
@app.route('/')
def home():
    return render_template('index.html')

# Define schema for evaluation response
class EvalSchema(typing.TypedDict):
    game_data: str
    feedback: str


# Global variables
rf_model = None
le_dict = {}
fill_values = {}
categorical_features = ['Genre', 'Platform', 'Player_Interact', 'Betting_Features']
bool_features = ['Contains_InApp_Purchase', 'Game_Reward_Real_Cash']
numeric_features = ['Release_Year', 'Player_Base (in millions)', 'Rating (out of 5)', 'Age_Rating', 'Luck_Factor']


def initialize_model():
    """Initialize and train the model"""
    global rf_model, le_dict, fill_values

    # Load and preprocess the data
    data = pd.read_csv('Realistic_Trending_Gaming_Gambling_Dataset.csv')
    data["Luck_Factor"] = pd.to_numeric(data["Luck_Factor"], errors='coerce')
    data["Luck_Factor"] = np.clip(data["Luck_Factor"], 0, 100)

    # Age Rating mapping
    age_rating_map = {
        'E': 3, 'E10+': 10, 'T': 13, 'M': 17, 'A': 18
    }
    data["Age_Rating"] = data["Age_Rating"].map(age_rating_map)

    # Select features
    features = [
        'Genre', 'Platform', 'Contains_InApp_Purchase', 'Release_Year',
        'Player_Base (in millions)', 'Rating (out of 5)', 'Age_Rating',
        'Game_Reward_Real_Cash', 'Luck_Factor', 'Player_Interact', 'Betting_Features'
    ]

    X = data[features].copy()
    y = data['Category'].copy()

    # Set fill values
    fill_values.update({
        'Contains_InApp_Purchase': False,
        'Game_Reward_Real_Cash': False,
        'Betting_Features': 'No',
        'Release_Year': X['Release_Year'].median(),
        'Player_Base (in millions)': X['Player_Base (in millions)'].median(),
        'Rating (out of 5)': X['Rating (out of 5)'].median(),
        'Age_Rating': X['Age_Rating'].median(),
        'Luck_Factor': X['Luck_Factor'].median(),
        'Player_Interact': 'Single-player'
    })
    X = X.fillna(fill_values)

    # Initialize and fit label encoders
    for feature in categorical_features:
        le_dict[feature] = LabelEncoder()
        X[feature] = le_dict[feature].fit_transform(X[feature])

    # Convert boolean features
    X = convert_bool_values(X, bool_features)
    X[numeric_features] = X[numeric_features].astype(float)

    # Train model
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)


def convert_bool_values(df, bool_features):
    """Convert boolean features to integers."""
    df_copy = df.copy()
    for feature in bool_features:
        df_copy[feature] = df_copy[feature].astype(int)
    return df_copy


def get_gemini_feedback(game_data, predicted_category, user_type):
    """Get feedback from Gemini AI"""
    try:
        if user_type == "developer":
            if predicted_category == "Gaming":
                prompt = "To better align your game with the gaming category, would you consider replacing gambling-related elements with engaging storylines, character progression, or strategy-driven gameplay? How could the game encourage more meaningful interaction and skill development among players?"
            else:
                prompt = "The game has been predicted as falling into the gambling category. Could you consider minimizing or removing betting mechanics, real-cash rewards, or luck-heavy features? How might the game benefit from adding skill-based challenges, immersive narratives, or creative multiplayer interactions?"
        elif user_type == "gamer":
            if predicted_category == "Gambling":
                prompt = "What would make the game more enjoyable for you as a player? Would removing cash rewards or betting mechanics and replacing them with in-game achievements, collectibles, or skill-based progression enhance your experience?"
            else:
                return "This game appears to focus on gaming elements rather than gambling mechanics. Enjoy the gameplay experience!"

        if prompt:
            response = chat.send_message(prompt, generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=EvalSchema
            ))
            return str(response)
        return "No specific feedback available for this combination."

    except Exception as e:
        return f"Error generating feedback: {str(e)}"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        game_data = data.get('game_data')
        user_type = data.get('user_type')

        if not game_data or not user_type:
            return jsonify({'error': 'Missing required data'}), 400

        # Create DataFrame for prediction
        input_data = pd.DataFrame([game_data])
        input_data = input_data.fillna(fill_values)
        input_data = convert_bool_values(input_data, bool_features)

        # Encode categorical variables
        for feature in categorical_features:
            if feature in le_dict:
                if input_data[feature].iloc[0] not in le_dict[feature].classes_:
                    return jsonify({
                        'error': f"Invalid value for {feature}. Allowed values: {list(le_dict[feature].classes_)}"
                    }), 400
                input_data[feature] = le_dict[feature].transform(input_data[feature])

        # Convert numeric features
        input_data[numeric_features] = input_data[numeric_features].astype(float)

        # Make prediction
        prediction = rf_model.predict(input_data)
        probabilities = rf_model.predict_proba(input_data)[0]

        # Get Gemini feedback
        feedback = get_gemini_feedback(game_data, prediction[0], user_type)

        return jsonify({
            'prediction': prediction[0],
            'probabilities': {
                'Gaming': abs(probabilities[0] - 1),
                'Gambling': abs(probabilities[1] - 1)
            },
            'feedback': feedback
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_allowed_values', methods=['GET'])
def get_allowed_values():
    """Return allowed values for categorical features"""
    try:
        allowed_values = {
            feature: list(le_dict[feature].classes_) for feature in categorical_features
        }
        return jsonify(allowed_values)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Return model information and feature importance"""
    try:
        feature_importance = pd.DataFrame({
            'feature': categorical_features + bool_features + numeric_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        return jsonify({
            'feature_importance': feature_importance.to_dict(orient='records'),
            'categorical_features': categorical_features,
            'bool_features': bool_features,
            'numeric_features': numeric_features
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    initialize_model()
    app.run(debug=True, port=5000)
