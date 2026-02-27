# ML Music Recommender

A music recommendation system that learns your taste in real time. It plays 30-second song previews and asks you to rate them. The model retrains after every rating and gets better over time.

Two versions are included:
- `music_recommender.py` — ensemble model (Random Forest + Logistic Regression + Gradient Boosting)
- `knn.py` — KNN-only version (finds songs most similar to your liked songs by audio features)
- `evaluate.py` — runs cross-validation on your feedback and compares all four models

---

## Setup

**Python 3.8+ required**

Install dependencies:
```
pip install pandas numpy scikit-learn requests
```

Download the dataset from Kaggle and place it in the same folder as the scripts:
https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

Rename the file to `dataset.csv` if it isn't already.

---

## How to Run

**Ensemble recommender:**
```
python music_recommender.py
```

**KNN recommender:**
```
python knn.py
```

**Rating scale:**
- `1` = love
- `2` = like
- `3` = dislike
- `4` = hate
- `s` = skip
- `r` = replay the preview
- `q` = quit

Ratings are saved to `user_feedback.csv` automatically. The model starts learning after 6 ratings and retrains after every new rating. If you quit and reopen, it picks up where you left off.

**Evaluate models:**
```
python evaluate.py
```

---

## Notes

- Previews are fetched from the iTunes API — no API key needed
- Songs with foreign characters or non-English iTunes genre labels are automatically skipped. Some will still sneak through so you can manually skip as well
