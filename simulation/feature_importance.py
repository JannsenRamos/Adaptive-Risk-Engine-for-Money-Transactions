import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def generate_feature_importance(df: pd.DataFrame):

    features = ['drift', 'preamble_score', 'vel_penalty']
    X = df[features]
    y = df['blocked']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 3. Extract and Rank Importance
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Signal': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    return feature_importance_df