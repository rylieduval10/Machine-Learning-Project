import re
import random
import requests
import pandas as pd
import numpy as np
import os
import subprocess
import threading
import sys
import tempfile
import shutil
from datetime import datetime

# audio playback globals
_audio_process = None
_audio_lock = threading.Lock()
_temp_audio_file = None

def _get_play_command(filepath):
    if sys.platform == "darwin":
        return ["afplay", filepath]
    elif sys.platform == "win32":
        ps_cmd = f"(New-Object Media.SoundPlayer \"{filepath}\").PlaySync()"
        return ["powershell", "-c", ps_cmd]
    else:
        for player in ["ffplay", "aplay", "mpg123", "cvlc"]:
            if shutil.which(player):
                if player == "ffplay":
                    return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath]
                return [player, filepath]
    return None

def play_preview(url):
    global _audio_process, _temp_audio_file
    stop_preview()
    try:
        response = requests.get(url, timeout=10)
        suffix = ".wav" if sys.platform == "win32" else ".m4a"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(response.content)
        tmp.close()
        _temp_audio_file = tmp.name
        cmd = _get_play_command(_temp_audio_file)
        if not cmd:
            print("  No audio player found.")
            return
        with _audio_lock:
            _audio_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"  Playback error: {e}")

def stop_preview():
    global _audio_process, _temp_audio_file
    with _audio_lock:
        if _audio_process and _audio_process.poll() is None:
            _audio_process.terminate()
            try:
                _audio_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _audio_process.kill()
        _audio_process = None
    if _temp_audio_file and os.path.exists(_temp_audio_file):
        try:
            os.remove(_temp_audio_file)
        except Exception:
            pass
        _temp_audio_file = None

DATASET_PATH = 'dataset.csv'
FEEDBACK_PATH = 'user_feedback.csv'
MIN_SONGS_TO_TRAIN = 6
MIN_POPULARITY = 20

AUDIO_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature', 'duration_ms'
]

GENRES = [
    'acoustic', 'pop', 'rock', 'indie', 'alternative', 'alt-rock', 'indie-pop',
    'singer-songwriter', 'songwriter', 'folk', 'country', 'bluegrass', 'honky-tonk',
    'hip-hop', 'r-n-b', 'soul', 'funk', 'groove', 'trip-hop',
    'edm', 'electronic', 'house', 'deep-house', 'electro', 'dance',
    'dubstep', 'drum-and-bass', 'breakbeat', 'chicago-house',
    'hard-rock', 'grunge', 'metal', 'heavy-metal', 'black-metal', 'death-metal',
    'metalcore', 'punk', 'punk-rock', 'hardcore', 'emo', 'goth', 'garage',
    'psych-rock', 'rock-n-roll', 'british', 'synth-pop', 'power-pop',
    'chill', 'sad', 'study', 'ambient', 'new-age', 'piano', 'guitar',
    'blues', 'jazz', 'gospel', 'reggae', 'dancehall', 'ska', 'dub',
    'disco', 'club', 'party', 'happy', 'show-tunes', 'pop-film', 'disney',
    'idm', 'trance', 'hardstyle', 'progressive-house', 'minimal-techno', 'detroit-techno', 'afrobeat',
]

FOREIGN_ITUNES_GENRES = {
    'Latin', 'Latino', 'Reggaeton', 'World', 'World Music', 'Brazilian',
    'Samba', 'Bossa Nova', 'Tango', 'Flamenco', 'Bollywood', 'Indian',
    'Afrobeat', 'Afropop', 'J-Pop', 'J-Rock', 'K-Pop', 'Anime',
    'Cantopop', 'Mandopop', 'Turkish', 'Arabic', 'French', 'German', 'Spanish',
}

def get_itunes_preview(artist, track_name):
    search_term = f"{artist} {track_name}".replace(' ', '+')
    url = f"https://itunes.apple.com/search?term={search_term}&entity=song&limit=1"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if data['resultCount'] > 0:
            result = data['results'][0]
            return {
                'preview_url': result.get('previewUrl'),
                'itunes_artist': result.get('artistName'),
                'itunes_genre': result.get('primaryGenreName', '')
            }
    except Exception:
        pass
    return None

def load_feedback():
    if os.path.exists(FEEDBACK_PATH):
        return pd.read_csv(FEEDBACK_PATH)
    return pd.DataFrame()

def save_feedback(track_id, song_row, label, weight=1):
    record = {'track_id': track_id, 'label': label, 'weight': weight, 'timestamp': datetime.now().isoformat()}
    for feat in AUDIO_FEATURES:
        record[feat] = song_row[feat]
    record['track_name'] = song_row['track_name']
    record['artists'] = song_row['artists']
    record['track_genre'] = song_row['track_genre']
    feedback_df = load_feedback()
    feedback_df = pd.concat([feedback_df, pd.DataFrame([record])], ignore_index=True)
    feedback_df.to_csv(FEEDBACK_PATH, index=False)
    return feedback_df

def train_model(feedback_df):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = feedback_df[AUDIO_FEATURES].values
    y = feedback_df['label'].values
    weights = feedback_df['weight'].values if 'weight' in feedback_df.columns else np.ones(len(y))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X, y, sample_weight=weights)

    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_scaled, y, sample_weight=weights)

    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X, y, sample_weight=weights)

    return rf, lr, gb, scaler

def get_recommendations(df, feedback_df, rf, lr, gb, scaler, top_n=10):
    heard_ids = set(feedback_df['track_id'].tolist())
    unseen = df[~df['track_id'].isin(heard_ids)].copy()

    X = unseen[AUDIO_FEATURES].values
    X_scaled = scaler.transform(X)

    rf_probs = rf.predict_proba(X)[:, 1]
    lr_probs = lr.predict_proba(X_scaled)[:, 1]
    gb_probs = gb.predict_proba(X)[:, 1]
    unseen['score'] = ((rf_probs + lr_probs + gb_probs) / 3).clip(0, 1)

    genre_stats = feedback_df.groupby('track_genre')['label'].agg(['sum', 'count'])
    genre_stats['like_rate'] = genre_stats['sum'] / genre_stats['count']

    if 'weight' in feedback_df.columns:
        hated = feedback_df[(feedback_df['weight'] == 3) & (feedback_df['label'] == 0)]['track_genre'].unique()
        unseen.loc[unseen['track_genre'].isin(hated), 'score'] *= 0.05

    bad_genres = genre_stats[(genre_stats['count'] >= 2) & (genre_stats['like_rate'] < 0.5)].index
    unseen.loc[unseen['track_genre'].isin(bad_genres), 'score'] *= 0.3

    loved_genres = genre_stats[(genre_stats['count'] >= 2) & (genre_stats['like_rate'] >= 0.75)].index
    unseen.loc[unseen['track_genre'].isin(loved_genres), 'score'] *= 3.0

    unseen = unseen.drop_duplicates(subset='track_id')

    def smart_cap(group):
        cap = 20 if group['track_genre'].iloc[0] in loved_genres else 5
        return group.nlargest(cap, 'score')

    balanced = unseen.sort_values('score', ascending=False).groupby('track_genre', group_keys=False).apply(smart_cap)
    return balanced.nlargest(top_n, 'score')

def explain_recommendation(song, feedback_df):
    likes = feedback_df[feedback_df['label'] == 1]
    dislikes = feedback_df[feedback_df['label'] == 0]
    if likes.empty or dislikes.empty:
        return
    reasons = []
    for feat in ['energy', 'acousticness', 'danceability', 'valence', 'tempo', 'instrumentalness']:
        song_val = song[feat]
        like_avg = likes[feat].mean()
        dislike_avg = dislikes[feat].mean()
        if abs(like_avg - dislike_avg) > 0.1 and abs(song_val - like_avg) < abs(song_val - dislike_avg):
            direction = 'high' if like_avg > dislike_avg else 'low'
            reasons.append(f'{feat}: you like {direction} (liked avg: {like_avg:.2f}, this song: {song_val:.2f})')
    if reasons:
        print("  Why this song:")
        for r in reasons[:3]:
            print(f"    - {r}")

def get_user_choice(prompt, valid_options):
    while True:
        val = input(prompt).strip().lower()
        if val in valid_options:
            return val
        print(f"  Enter one of: {', '.join(valid_options)}")

def main():
    print("Music Recommender\n")

    df = pd.read_csv(DATASET_PATH).dropna(subset=['artists', 'track_name'])
    df = df[df['track_genre'].isin(GENRES)]
    df = df[df['popularity'] >= MIN_POPULARITY]
    df = df.drop_duplicates(subset='track_id').reset_index(drop=True)

    def is_latin(text):
        return not bool(re.search(r'[\u0400-\u04FF\u0600-\u06FF\u0900-\u097F\u4E00-\u9FFF\u3040-\u30FF\u0E00-\u0E7F]', str(text)))
    df = df[df['track_name'].apply(is_latin) & df['artists'].apply(is_latin)].reset_index(drop=True)

    foreign_words = r'\b(los|las|del|por|para|como|cuando|amor|corazon|noche|esta|este|pero|porque|han|les|des|je|nous|vous|ils|elles|mon|ton|dans|sur|avec|qui|lui|leur)\b'
    df = df[~df['track_name'].apply(lambda x: bool(re.search(foreign_words, str(x).lower())))].reset_index(drop=True)

    print(f"Loaded {len(df):,} songs")

    feedback_df = load_feedback()
    if not feedback_df.empty:
        likes = int(feedback_df['label'].sum())
        print(f"Resuming - {likes} likes, {len(feedback_df) - likes} dislikes\n")

    model_active = False
    rf = lr = gb = scaler = None
    skipped_ids = set()
    last_genre = None

    if not feedback_df.empty and len(feedback_df) >= MIN_SONGS_TO_TRAIN:
        likes = int(feedback_df['label'].sum())
        dislikes = len(feedback_df) - likes
        if likes > 0 and dislikes > 0:
            print("Loading model...")
            rf, lr, gb, scaler = train_model(feedback_df)
            model_active = True
            print("Top picks for you:")
            top5 = get_recommendations(df, feedback_df, rf, lr, gb, scaler, top_n=5)
            for i, (_, s) in enumerate(top5.iterrows(), 1):
                print(f"  {i}. {s['track_name']} - {s['artists'].split(';')[0]} ({s['track_genre']})")
            print()

    try:
        while True:
            print("-" * 50)

            heard_ids = (set(feedback_df['track_id'].tolist()) if not feedback_df.empty else set()) | skipped_ids
            explore = model_active and random.random() < 0.25

            if model_active and not explore:
                recs = get_recommendations(df, feedback_df, rf, lr, gb, scaler, top_n=50)
                if recs.empty:
                    available = df[~df['track_id'].isin(heard_ids)]
                    song = available.sample(n=1).iloc[0] if not available.empty else df.sample(n=1).iloc[0]
                    score = None
                else:
                    diverse_recs = recs[recs['track_genre'] != last_genre] if last_genre and len(recs[recs['track_genre'] != last_genre]) > 3 else recs
                    diverse_recs = diverse_recs.groupby('track_genre').head(10).reset_index(drop=True)
                    song = diverse_recs.sample(n=1, weights=diverse_recs['score'] ** 3).iloc[0]
                    score = song['score']
            else:
                available = df[~df['track_id'].isin(heard_ids)]
                if available.empty:
                    print("You've heard every song!")
                    break
                popular = available[available['popularity'] >= 75]
                if not popular.empty and random.random() < 0.30:
                    song = popular.sample(n=1).iloc[0]
                else:
                    genre = available['track_genre'].sample(n=1).iloc[0]
                    song = available[available['track_genre'] == genre].sample(n=1).iloc[0]
                score = None

            current_id = song['track_id']
            last_genre = song['track_genre']
            skipped_ids.add(current_id)

            print(f"\n{song['track_name']} - {song['artists']}")
            print(f"  Genre: {song['track_genre']}  |  Popularity: {song.get('popularity', '?')}")
            print(f"  Energy: {song['energy']:.2f}  |  Acousticness: {song['acousticness']:.2f}  |  Valence: {song['valence']:.2f}  |  Tempo: {song['tempo']:.0f} BPM")
            if score is not None:
                print(f"  Confidence: {score:.0%}")

            if score is not None and not feedback_df.empty:
                explain_recommendation(song, feedback_df)

            if not all(ord(c) < 128 for c in song['track_name'] + song['artists']):
                print("  Skipping (non-English characters)")
                continue

            artist = song['artists'].split(';')[0].split(',')[0]
            result = get_itunes_preview(artist, song['track_name'])

            if result and result['preview_url']:
                if result.get('itunes_genre') in FOREIGN_ITUNES_GENRES:
                    print(f"  Skipping (iTunes genre: {result['itunes_genre']})")
                    continue
                itunes_artist = (result.get('itunes_artist') or '').lower()
                dataset_artist = artist.lower()
                a_words = set(dataset_artist.split()) - {'the', 'a', 'and', 'of', 'feat', 'ft'}
                i_words = set(itunes_artist.split()) - {'the', 'a', 'and', 'of', 'feat', 'ft'}
                if not (a_words & i_words) and a_words and i_words:
                    print("  Skipping (artist mismatch)")
                    continue
                print("\nPlaying...")
                play_preview(result['preview_url'])
            else:
                print("  No preview found, skipping.")
                continue

            print()
            while True:
                choice = get_user_choice(
                    "  [1=love / 2=like / 3=dislike / 4=hate / s=skip / r=replay / q=quit]: ",
                    ['1', '2', '3', '4', 's', 'r', 'q']
                )
                if choice == 'r':
                    play_preview(result['preview_url'])
                else:
                    break
            stop_preview()

            if choice == 'q':
                break
            elif choice == 's':
                print("  Skipped.")
                continue
            else:
                skipped_ids.discard(current_id)
                if choice == '1':
                    label, weight, msg = 1, 3, "Loved!"
                elif choice == '2':
                    label, weight, msg = 1, 1, "Liked!"
                elif choice == '3':
                    label, weight, msg = 0, 1, "Disliked."
                else:
                    label, weight, msg = 0, 3, "Hated."

                feedback_df = save_feedback(song['track_id'], song, label, weight)
                likes = int(feedback_df['label'].sum())
                total = len(feedback_df)
                print(f"  {msg} [{total} rated - {likes} likes, {total - likes} dislikes]")

                if total >= MIN_SONGS_TO_TRAIN and likes > 0 and (total - likes) > 0:
                    rf, lr, gb, scaler = train_model(feedback_df)
                    model_active = True
                    print("  Model updated.")

    finally:
        stop_preview()

    print("\nDone! Feedback saved.")

if __name__ == '__main__':
    main()