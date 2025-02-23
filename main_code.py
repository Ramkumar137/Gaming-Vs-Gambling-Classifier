import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import google.generativeai as genai
import PyPDF2
import os
import json
import typing_extensions as typing

genai.configure(api_key="Api-key-here")
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(
    history=[
        {"role": "user", "parts": "Hello"},
        {"role": "model", "parts": "Great to meet you. What would you like to know?"},
    ]
)


# Define schema for evaluation response
class EvalSchema(typing.TypedDict):
    game_data: str
    feedback: str


# Load and preprocess the data
data = pd.read_csv('Realistic_Trending_Gaming_Gambling_Dataset.csv')

# Convert Luck_Factor to numeric, coercing errors to NaN
data["Luck_Factor"] = pd.to_numeric(data["Luck_Factor"], errors='coerce')

# Clip Luck_Factor values between 0 and 100
data["Luck_Factor"] = np.clip(data["Luck_Factor"], 0, 100)

# Age Rating mapping
age_rating_map = {
    'E': 3,  # Everyone
    'E10+': 10,  # Everyone 10 and older
    'T': 13,  # Teen
    'M': 17,  # Mature
    'A': 18  # Adults only
}
data["Age_Rating"] = data["Age_Rating"].map(age_rating_map)

# Select features for the model
features = [
    'Genre', 'Platform', 'Contains_InApp_Purchase', 'Release_Year',
    'Player_Base (in millions)', 'Rating (out of 5)', 'Age_Rating',
    'Game_Reward_Real_Cash', 'Luck_Factor', 'Player_Interact', 'Betting_Features'
]

# Create clean copies of features and target
X = data[features].copy()
y = data['Category'].copy()

# Define feature types
categorical_features = ['Genre', 'Platform', 'Player_Interact', 'Betting_Features']
bool_features = ['Contains_InApp_Purchase', 'Game_Reward_Real_Cash']
numeric_features = ['Release_Year', 'Player_Base (in millions)', 'Rating (out of 5)', 'Age_Rating', 'Luck_Factor']

# Fill NaN values
fill_values = {
    'Contains_InApp_Purchase': False,
    'Game_Reward_Real_Cash': False,
    'Betting_Features': 'No',
    'Release_Year': X['Release_Year'].median(),
    'Player_Base (in millions)': X['Player_Base (in millions)'].median(),
    'Rating (out of 5)': X['Rating (out of 5)'].median(),
    'Age_Rating': X['Age_Rating'].median(),
    'Luck_Factor': X['Luck_Factor'].median(),
    'Player_Interact': 'Single-player'
}
X = X.fillna(fill_values)


# Function to convert boolean values
def convert_bool_values(df, bool_features):
    """Convert boolean features to integers."""
    df_copy = df.copy()
    for feature in bool_features:
        df_copy[feature] = df_copy[feature].astype(int)
    return df_copy


# Initialize label encoders for categorical variables
le_dict = {}
for feature in categorical_features:
    le_dict[feature] = LabelEncoder()
    X[feature] = le_dict[feature].fit_transform(X[feature])

# Convert boolean features
X = convert_bool_values(X, bool_features)

# Ensure numeric columns are float
X[numeric_features] = X[numeric_features].astype(float)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Print model evaluation
print("accuracy score :", accuracy_score(y_true=y_test, y_pred=y_pred), end='-' * 25)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)


def predict_category(new_game_data):
    """
    Predict the category (Gaming/Gambling) for a new game.

    Parameters:
    new_game_data (dict): Dictionary containing game features

    Returns:
    tuple: (predicted_category, probability_distribution)
    """
    try:
        # Create a DataFrame with the new game data
        input_data = pd.DataFrame([new_game_data])

        # Fill NaN values if any
        input_data = input_data.fillna(fill_values)

        # Handle boolean features
        input_data = convert_bool_values(input_data, bool_features)

        # Encode categorical variables
        for feature in categorical_features:
            if feature in le_dict:
                # Check if input value exists in training data
                if input_data[feature].iloc[0] not in le_dict[feature].classes_:
                    raise ValueError(
                        f"Unknown category '{input_data[feature].iloc[0]}' in feature '{feature}'. "
                        f"Allowed values are: {list(le_dict[feature].classes_)}"
                    )
                input_data[feature] = le_dict[feature].transform(input_data[feature])

        # Ensure numeric columns are float
        input_data[numeric_features] = input_data[numeric_features].astype(float)

        # Make prediction
        prediction = rf_model.predict(input_data)
        probability = rf_model.predict_proba(input_data)

        return prediction[0], probability[0]

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None


def get_user_input():
    """Get game data from user input."""
    print("\nPlease enter the game details:")

    # Genre input
    print("\nAvailable Genres:", list(le_dict['Genre'].classes_))
    genre = input("Enter Genre: ").strip().lower()

    # Platform input
    print("\nAvailable Platforms:", list(le_dict['Platform'].classes_))
    platform = input("Enter Platform: ").strip()

    # Player Interaction input
    print("\nAvailable Player Interaction types:", list(le_dict['Player_Interact'].classes_))
    player_interact = input("Enter Player Interaction type: ").strip()

    # Betting Features input
    print("\nAvailable Betting Features options:", list(le_dict['Betting_Features'].classes_))
    betting_features = input("Enter Betting Features option: ").strip()

    # Boolean inputs
    contains_iap = input("\nContains In-App Purchases? (yes/no): ").lower() == 'yes'
    game_reward_cash = input("Game Rewards Real Cash? (yes/no): ").lower() == 'yes'

    # Numeric inputs
    while True:
        try:
            release_year = int(input("\nEnter Release Year (e.g., 2023): "))
            if 1970 <= release_year <= 2030:
                break
            print("Please enter a valid year between 1970 and 2030")
        except ValueError:
            print("Please enter a valid year")

    while True:
        try:
            player_base = float(input("Enter Player Base in millions (e.g., 200): "))
            if player_base >= 0:
                break
            print("Please enter a non-negative number")
        except ValueError:
            print("Please enter a valid number")

    while True:
        try:
            rating = float(input("Enter Rating (0-5): "))
            if 0 <= rating <= 5:
                break
            print("Please enter a rating between 0 and 5")
        except ValueError:
            print("Please enter a valid rating")

    while True:
        try:
            age_rating = int(input("Enter Age Rating (3=E, 10=E10+, 13=T, 17=M, 18=A): "))
            if age_rating in [3, 10, 13, 17, 18]:
                break
            print("Please enter a valid age rating (3, 10, 13, 17, or 18)")
        except ValueError:
            print("Please enter a valid age rating")

    while True:
        try:
            luck_factor = float(input("Enter Luck Factor (0-100): "))
            if 0 <= luck_factor <= 100:
                break
            print("Please enter a luck factor between 0 and 100")
        except ValueError:
            print("Please enter a valid number")

    # Create game data dictionary
    game_data = {
        'Genre': genre,
        'Platform': platform,
        'Contains_InApp_Purchase': contains_iap,
        'Release_Year': release_year,
        'Player_Base (in millions)': player_base,
        'Rating (out of 5)': rating,
        'Age_Rating': age_rating,
        'Game_Reward_Real_Cash': game_reward_cash,
        'Luck_Factor': luck_factor,
        'Player_Interact': player_interact,
        'Betting_Features': betting_features
    }

    return game_data


def main():
    """Main function to run the game classification program."""
    global prompt
    print("\nWelcome to the Game Classification System!")
    print("This system will help you determine if a game is likely to be classified as Gaming or Gambling.")

    while True:
        # Get user input
        user = input("[developer/gamer]\nEnter the role : ")
        game_data = get_user_input()

        # Make prediction
        predicted_category, probabilities = predict_category(game_data)

        if predicted_category is not None:
            print("\nPrediction Results:")
            print(f"Predicted Category: {predicted_category}")
            print(f"Probability Distribution:")
            probabilities[0] = abs(probabilities[0] - 1)
            probabilities[1] = abs(probabilities[1] - 1)
            print(f"Gaming: {probabilities[0]:.2f}")
            print(f"Gambling: {probabilities[1]:.2f}")

            evaluation_prompt = (
                f"These are the game_data:{game_data}\n"
                f"Predicted Category : {predicted_category}\n"
                f"Give the feedback for this. give advise for the user like how can they improve."
                f"if the predicated category is Gambling then give the feedback like which game data is lead to raise Gambling."
            )
            prompt=""
            if user == "developer":
                if predicted_category == "Gaming":
                    prompt = "To better align your game with the gaming category, would you consider replacing gambling-related elements with engaging storylines, character progression, or strategy-driven gameplay? How could the game encourage more meaningful interaction and skill development among players?"

                elif predicted_category == "Gambling":
                    prompt = "The game has been predicted as falling into the gambling category. Could you consider minimizing or removing betting mechanics, real-cash rewards, or luck-heavy features? How might the game benefit from adding skill-based challenges, immersive narratives, or creative multiplayer interactions?"
            if user == "gamer":
                if predicted_category == "Gambling":
                    prompt = "What would make the game more enjoyable for you as a player? Would removing cash rewards or betting mechanics and replacing them with in-game achievements, collectibles, or skill-based progression enhance your experience?"


            if not prompt:
                print("empty prompt")


            response = chat.send_message(prompt, generation_config=genai.GenerationConfig(
                response_mime_type="application/json", response_schema=EvalSchema
            ), )

            print(f"Feedback : {response}")

        if input("\nWould you like to classify another game? (yes/no): ").lower() != 'yes':
            break

    print("\nThank you for using the Game Classification System!")

if __name__ == "__main__":
    main()
