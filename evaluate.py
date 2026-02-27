import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate

FEEDBACK_PATH = 'user_feedback.csv'

AUDIO_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature', 'duration_ms'
]

def main():
    try:
        fb = pd.read_csv(FEEDBACK_PATH)
    except FileNotFoundError:
        print("No feedback file found. Rate some songs first.")
        return

    likes = int(fb['label'].sum())
    dislikes = len(fb) - likes
    print(f"\nFeedback: {len(fb)} songs rated ({likes} likes, {dislikes} dislikes)")

    if len(fb) < 20:
        print("Warning: less than 20 ratings, results may not be reliable.")
    print()

    X = fb[AUDIO_FEATURES].values
    y = fb['label'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_splits = min(5, likes, dislikes)
    if n_splits < 2:
        print("Not enough likes or dislikes for cross-validation.")
        return

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    k = min(10, max(3, len(y) // 5))

    models = {
        'Random Forest':       (RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42), X),
        'Logistic Regression': (LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42), X_scaled),
        'Gradient Boosting':   (GradientBoostingClassifier(n_estimators=100, random_state=42), X),
        'KNN':                 (KNeighborsClassifier(n_neighbors=k, weights='distance'), X_scaled),
    }

    scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1'}
    results = {}

    for name, (model, features) in models.items():
        scores = cross_validate(model, features, y, cv=cv, scoring=scoring)
        results[name] = {
            'Accuracy':  scores['test_accuracy'].mean(),
            'Precision': scores['test_precision'].mean(),
            'Recall':    scores['test_recall'].mean(),
            'F1':        scores['test_f1'].mean(),
        }

    print(f"{'Metric':<20} {'Random Forest':>14} {'Logistic Reg':>13} {'Grad Boost':>11} {'KNN':>8}")
    print("-" * 70)

    for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
        vals = [results[m][metric] for m in models]
        best_idx = vals.index(max(vals))
        labels = ['RF', 'LR', 'GB', 'KNN']
        winner = f"<- {labels[best_idx]}"
        print(f"{metric:<20} {vals[0]:>13.1%} {vals[1]:>12.1%} {vals[2]:>11.1%} {vals[3]:>8.1%}  {winner}")

    print("-" * 70)
    avgs = {name: np.mean(list(v.values())) for name, v in results.items()}
    print(f"\nBest model: {max(avgs, key=avgs.get)}")

    # feature importance
    print("\nTop features (Random Forest):")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=AUDIO_FEATURES)
    for feat, imp in importances.nlargest(8).items():
        bar = '#' * int(imp * 200)
        print(f"  {feat:<20} {imp:.3f}  {bar}")

    # taste profile
    print("\nTaste profile:")
    liked = fb[fb['label'] == 1]
    disliked = fb[fb['label'] == 0]
    print(f"  {'Feature':<20} {'Liked avg':>10} {'Disliked avg':>13}")
    print(f"  {'-'*20} {'-'*10} {'-'*13}")
    for feat in ['energy', 'danceability', 'acousticness', 'valence', 'tempo', 'speechiness']:
        l = liked[feat].mean()
        d = disliked[feat].mean()
        diff = l - d
        note = ''
        if abs(diff) > 0.05:
            note = f"  ({'higher' if diff > 0 else 'lower'} when liked)"
        print(f"  {feat:<20} {l:>10.2f} {d:>13.2f}{note}")

    # genre breakdown
    print("\nGenre breakdown:")
    genre_stats = fb.groupby('track_genre')['label'].agg(['sum', 'count'])
    genre_stats['like_rate'] = genre_stats['sum'] / genre_stats['count']
    genre_stats = genre_stats.sort_values('like_rate', ascending=False)
    for genre, row in genre_stats.iterrows():
        likes_str = 'L ' * int(row['sum'])
        dislikes_str = 'D ' * int(row['count'] - row['sum'])
        print(f"  {genre:<20} {int(row['sum'])}/{int(row['count'])}  {likes_str}{dislikes_str}")

if __name__ == '__main__':
    main()