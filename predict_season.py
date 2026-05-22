"""
Football Season Predictor 2025/26
===================================
GradientBoosting (25 features) + Dixon-Coles Poisson Monte Carlo (10,000 sims)
Supports: Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Brasileirão

Run: python3.12 predict_season.py
Requires: pip install pandas numpy scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import os

BASE_PATH = r'C:\Users\theocout\Downloads\PP\SPO'

LEAGUES = {
    1: {
        'name': 'Premier League',
        'file': 'premier-league-full.csv',
        'teams': [
            'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
            'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham',
            'Leeds', 'Liverpool', 'Man City', 'Man United', 'Newcastle',
            "Nott'm Forest", 'Sunderland', 'Tottenham', 'West Ham', 'Wolves',
        ],
        'promoted': ['Burnley', 'Leeds', 'Sunderland'],
        'n_teams': 20,
    },
    2: {
        'name': 'La Liga',
        'file': 'la-liga-full.csv',
        'teams': [
            'Alaves', 'Ath Bilbao', 'Ath Madrid', 'Barcelona', 'Betis',
            'Celta', 'Elche', 'Espanol', 'Getafe', 'Girona',
            'Levante', 'Mallorca', 'Osasuna', 'Oviedo', 'Real Madrid',
            'Sevilla', 'Sociedad', 'Valencia', 'Vallecano', 'Villarreal',
        ],
        'promoted': ['Elche', 'Levante', 'Oviedo'],
        'n_teams': 20,
    },
    3: {
        'name': 'Serie A',
        'file': 'serie-a-full.csv',
        'teams': [
            'Atalanta', 'Bologna', 'Cagliari', 'Como', 'Cremonese',
            'Fiorentina', 'Genoa', 'Inter', 'Juventus', 'Lazio',
            'Lecce', 'Milan', 'Napoli', 'Parma', 'Pisa',
            'Roma', 'Sassuolo', 'Torino', 'Udinese', 'Verona',
        ],
        'promoted': ['Cremonese', 'Pisa', 'Sassuolo'],
        'n_teams': 20,
    },
    4: {
        'name': 'Bundesliga',
        'file': 'bundesliga-full.csv',
        'teams': [
            'Augsburg', 'Bayern Munich', 'Dortmund', 'Ein Frankfurt',
            'FC Koln', 'Freiburg', 'Hamburg', 'Heidenheim', 'Hoffenheim',
            'Leverkusen', "M'gladbach", 'Mainz', 'RB Leipzig', 'St Pauli',
            'Stuttgart', 'Union Berlin', 'Werder Bremen', 'Wolfsburg',
        ],
        'promoted': ['FC Koln', 'Hamburg'],
        'n_teams': 18,
    },
    5: {
        'name': 'Ligue 1',
        'file': 'ligue-1-full.csv',
        'teams': [
            'Angers', 'Auxerre', 'Brest', 'Le Havre', 'Lens',
            'Lille', 'Lorient', 'Lyon', 'Marseille', 'Metz',
            'Monaco', 'Nantes', 'Nice', 'Paris FC', 'Paris SG',
            'Rennes', 'Strasbourg', 'Toulouse',
        ],
        'promoted': ['Lorient', 'Metz', 'Paris FC'],
        'n_teams': 18,
    },
    6: {
        'name': 'Brasileirão',
        'file': 'brasileirao-full.csv',
        'teams': [
            'Athletico-PR', 'Atletico-MG', 'Bahia', 'Botafogo RJ', 'Bragantino',
            'Chapecoense-SC', 'Corinthians', 'Coritiba', 'Cruzeiro', 'Flamengo RJ',
            'Fluminense', 'Gremio', 'Internacional', 'Mirassol', 'Palmeiras',
            'Remo', 'Santos', 'Sao Paulo', 'Vasco', 'Vitoria',
        ],
        'promoted': ['Chapecoense-SC', 'Coritiba', 'Athletico-PR', 'Remo'],
        'n_teams': 20,
    },
}

FEATURES = [
    'imp_home', 'imp_draw', 'imp_away',
    'home_elo', 'away_elo', 'elo_diff',
    'home_goals_avg', 'away_goals_avg',
    'home_conceded_avg', 'away_conceded_avg',
    'home_goals_avg_10', 'away_goals_avg_10',
    'home_gd_avg', 'away_gd_avg',
    'home_points_form', 'away_points_form',
    'home_season_ppg', 'away_season_ppg',
    'home_win_rate_5', 'away_win_rate_5',
    'strength_gap',
    'home_draw_rate_5', 'away_draw_rate_5', 'combined_draw_rate', 'elo_closeness',
]


def build_features(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    # Implied probabilities — fallback to Elo-derived if odds missing
    has_odds = 'B365H' in df.columns and df['B365H'].notna().sum() > len(df) * 0.5
    if has_odds:
        df['imp_home'] = 1 / df['B365H']
        df['imp_draw'] = 1 / df['B365D']
        df['imp_away'] = 1 / df['B365A']
        total = df['imp_home'] + df['imp_draw'] + df['imp_away']
        df['imp_home'] /= total
        df['imp_draw'] /= total
        df['imp_away'] /= total
    else:
        print("  ⚠ No betting odds found — using Elo-derived implied probabilities")
        df['imp_home'] = 0.45
        df['imp_draw'] = 0.27
        df['imp_away'] = 0.28

    # Elo ratings
    K, HOME_ADV = 20, 50
    elo_ratings = {}
    elo_home_list, elo_away_list = [], []

    for _, row in df.iterrows():
        home, away = row['Home'], row['Away']
        if home not in elo_ratings: elo_ratings[home] = 1500
        if away not in elo_ratings: elo_ratings[away] = 1500
        h_elo, a_elo = elo_ratings[home], elo_ratings[away]
        elo_home_list.append(h_elo)
        elo_away_list.append(a_elo)
        exp_h = 1 / (1 + 10 ** ((a_elo - (h_elo + HOME_ADV)) / 400))
        s_h = 1.0 if row['FTR'] == 'H' else (0.5 if row['FTR'] == 'D' else 0.0)
        elo_ratings[home] = h_elo + K * (s_h - exp_h)
        elo_ratings[away] = a_elo + K * ((1 - s_h) - (1 - exp_h))

    df['home_elo'] = elo_home_list
    df['away_elo'] = elo_away_list
    df['elo_diff'] = df['home_elo'] - df['away_elo']

    # Fill rows missing odds with Elo-derived probs (per-row fallback)
    if has_odds:
        missing_odds = df['imp_home'].isna()
        if missing_odds.any():
            elo_prob = 1 / (1 + 10 ** (-(df.loc[missing_odds, 'elo_diff'] + HOME_ADV) / 400))
            df.loc[missing_odds, 'imp_home'] = elo_prob * 0.74  # normalize with draw share
            df.loc[missing_odds, 'imp_draw'] = 0.26
            df.loc[missing_odds, 'imp_away'] = (1 - elo_prob) * 0.74

    # Rolling 5-game
    df['home_goals_avg'] = df.groupby('Home')['HomeGoals'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['away_goals_avg'] = df.groupby('Away')['AwayGoals'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['home_conceded_avg'] = df.groupby('Home')['AwayGoals'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['away_conceded_avg'] = df.groupby('Away')['HomeGoals'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    # Rolling 10-game
    df['home_goals_avg_10'] = df.groupby('Home')['HomeGoals'].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    df['away_goals_avg_10'] = df.groupby('Away')['AwayGoals'].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())

    # Goal difference
    df['home_gd_avg'] = df['home_goals_avg'] - df['home_conceded_avg']
    df['away_gd_avg'] = df['away_goals_avg'] - df['away_conceded_avg']

    # Points form
    df['home_pts'] = (df['FTR'] == 'H').astype(int) * 3 + (df['FTR'] == 'D').astype(int)
    df['away_pts'] = (df['FTR'] == 'A').astype(int) * 3 + (df['FTR'] == 'D').astype(int)
    df['home_points_form'] = df.groupby('Home')['home_pts'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['away_points_form'] = df.groupby('Away')['away_pts'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    # Season PPG
    df['home_season_ppg'] = df.groupby(['Season_End_Year', 'Home'])['home_pts'].transform(lambda x: x.shift(1).expanding().mean())
    df['away_season_ppg'] = df.groupby(['Season_End_Year', 'Away'])['away_pts'].transform(lambda x: x.shift(1).expanding().mean())

    # Win rates
    df['home_win_flag'] = (df['FTR'] == 'H').astype(int)
    df['away_win_flag'] = (df['FTR'] == 'A').astype(int)
    df['home_win_rate_5'] = df.groupby('Home')['home_win_flag'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['away_win_rate_5'] = df.groupby('Away')['away_win_flag'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    # Draw features
    df['draw_flag'] = (df['FTR'] == 'D').astype(int)
    df['home_draw_rate_5'] = df.groupby('Home')['draw_flag'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['away_draw_rate_5'] = df.groupby('Away')['draw_flag'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['combined_draw_rate'] = df['home_draw_rate_5'] + df['away_draw_rate_5']
    df['elo_closeness'] = 1 / (1 + abs(df['elo_diff']) / 100)

    # Derived
    df['strength_gap'] = df['home_season_ppg'] - df['away_season_ppg']

    fill_defaults = {
        'home_season_ppg': 0.9, 'away_season_ppg': 0.6,
        'home_points_form': 1.0, 'away_points_form': 0.7,
        'home_win_rate_5': 0.35, 'away_win_rate_5': 0.20,
        'strength_gap': 0, 'home_draw_rate_5': 0.25,
        'away_draw_rate_5': 0.25, 'combined_draw_rate': 0.50, 'elo_closeness': 0.5,
    }
    for col, val in fill_defaults.items():
        df[col] = df[col].fillna(val)

    df = df.dropna(subset=FEATURES)
    return df, elo_ratings


def train_model(df):
    le = LabelEncoder()
    df['result'] = df['FTR'].map({'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'})
    df['result_encoded'] = le.fit_transform(df['result'])

    X = df[FEATURES]
    y = df['result_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    class_counts = np.bincount(y_train)
    weights = np.sqrt(len(y_train) / (len(class_counts) * class_counts))
    weights[list(le.classes_).index('Draw')] *= 1.15
    sample_weights = np.array([weights[label] for label in y_train])

    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print(f"\n{'='*60}")
    print(f"  Model Accuracy: {acc*100:.1f}%")
    print(f"{'='*60}")
    print(classification_report(y_test, predictions, target_names=le.classes_))

    return model, le, acc


def simulate_season(model, le, df, elo_ratings, league_config, n_sims=10000):
    """
    Monte Carlo simulation using proper independent Poisson goal draws.
    
    Approach: For each fixture, sample home_goals ~ Poisson(λ_home) and
    away_goals ~ Poisson(λ_away) independently. The scoreline determines
    the match outcome directly. The classifier probabilities are used to
    WEIGHT which Poisson regime to draw from (stronger/weaker λ scaling),
    not to override the scoreline.
    """
    teams = league_config['teams']
    promoted = league_config['promoted']
    HOME_ADV = 50

    league_avg_home = df['HomeGoals'].mean()
    league_avg_away = df['AwayGoals'].mean()

    team_stats = {}
    for team in teams:
        home_data = df[df['Home'] == team]
        away_data = df[df['Away'] == team]

        if len(home_data) < 5 or len(away_data) < 5 or team in promoted:
            if len(home_data) >= 5 and len(away_data) >= 5:
                th = home_data.iloc[-1]
                ta = away_data.iloc[-1]
                team_stats[team] = {
                    'home_goals_avg': th.get('home_goals_avg', 1.2) * 0.85,
                    'away_goals_avg': ta.get('away_goals_avg', 0.9) * 0.85,
                    'home_conceded_avg': th.get('home_conceded_avg', 1.3) * 1.1,
                    'away_conceded_avg': ta.get('away_conceded_avg', 1.5) * 1.1,
                    'home_goals_avg_10': th.get('home_goals_avg_10', 1.2) * 0.85,
                    'away_goals_avg_10': ta.get('away_goals_avg_10', 0.9) * 0.85,
                    'home_gd_avg': -0.1, 'away_gd_avg': -0.6,
                    'home_points_form': 1.1, 'away_points_form': 0.8,
                    'home_season_ppg': 1.1, 'away_season_ppg': 0.7,
                    'home_elo': elo_ratings.get(team, 1450), 'away_elo': elo_ratings.get(team, 1450),
                    'home_win_rate_5': 0.40, 'away_win_rate_5': 0.25,
                    'home_draw_rate_5': 0.25, 'away_draw_rate_5': 0.25,
                    'home_attack': th.get('home_goals_avg', 1.2) * 0.85 / max(league_avg_home, 0.5),
                    'away_attack': ta.get('away_goals_avg', 0.9) * 0.85 / max(league_avg_away, 0.5),
                    'home_defense': th.get('home_conceded_avg', 1.3) * 1.1 / max(league_avg_away, 0.5),
                    'away_defense': ta.get('away_conceded_avg', 1.5) * 1.1 / max(league_avg_home, 0.5),
                }
            else:
                team_stats[team] = {
                    'home_goals_avg': 1.2, 'away_goals_avg': 0.9,
                    'home_conceded_avg': 1.4, 'away_conceded_avg': 1.7,
                    'home_goals_avg_10': 1.2, 'away_goals_avg_10': 0.9,
                    'home_gd_avg': -0.2, 'away_gd_avg': -0.8,
                    'home_points_form': 1.0, 'away_points_form': 0.7,
                    'home_season_ppg': 1.0, 'away_season_ppg': 0.6,
                    'home_elo': 1400, 'away_elo': 1400,
                    'home_win_rate_5': 0.35, 'away_win_rate_5': 0.20,
                    'home_draw_rate_5': 0.25, 'away_draw_rate_5': 0.25,
                    'home_attack': 0.80, 'away_attack': 0.75,
                    'home_defense': 1.20, 'away_defense': 1.40,
                }
            print(f"  ℹ {team}: seeded as promoted team")
        else:
            th = home_data.iloc[-1]
            ta = away_data.iloc[-1]
            team_stats[team] = {
                'home_goals_avg': th['home_goals_avg'], 'away_goals_avg': ta['away_goals_avg'],
                'home_conceded_avg': th['home_conceded_avg'], 'away_conceded_avg': ta['away_conceded_avg'],
                'home_goals_avg_10': th['home_goals_avg_10'], 'away_goals_avg_10': ta['away_goals_avg_10'],
                'home_gd_avg': th['home_gd_avg'], 'away_gd_avg': ta['away_gd_avg'],
                'home_points_form': th['home_points_form'], 'away_points_form': ta['away_points_form'],
                'home_season_ppg': th['home_season_ppg'], 'away_season_ppg': ta['away_season_ppg'],
                'home_elo': elo_ratings.get(team, 1500), 'away_elo': elo_ratings.get(team, 1500),
                'home_win_rate_5': th['home_win_rate_5'], 'away_win_rate_5': ta['away_win_rate_5'],
                'home_draw_rate_5': th['home_draw_rate_5'], 'away_draw_rate_5': ta['away_draw_rate_5'],
                'home_attack': th['home_goals_avg'] / max(league_avg_home, 0.5),
                'away_attack': ta['away_goals_avg'] / max(league_avg_away, 0.5),
                'home_defense': th['home_conceded_avg'] / max(league_avg_away, 0.5),
                'away_defense': ta['away_conceded_avg'] / max(league_avg_home, 0.5),
            }

    # Generate fixtures and predict classifier probabilities
    fixtures = [{'Home': h, 'Away': a} for h in teams for a in teams if h != a]
    fixtures_df = pd.DataFrame(fixtures)
    n_fixtures = len(fixtures)

    for col in [f for f in FEATURES if f.startswith('home_')]:
        fixtures_df[col] = fixtures_df['Home'].map({t: s.get(col, 0) for t, s in team_stats.items()})
    for col in [f for f in FEATURES if f.startswith('away_')]:
        fixtures_df[col] = fixtures_df['Away'].map({t: s.get(col, 0) for t, s in team_stats.items()})

    fixtures_df['elo_diff'] = fixtures_df['Home'].map({t: s['home_elo'] for t, s in team_stats.items()}) - \
                              fixtures_df['Away'].map({t: s['away_elo'] for t, s in team_stats.items()})
    fixtures_df['elo_closeness'] = 1 / (1 + abs(fixtures_df['elo_diff']) / 100)
    fixtures_df['strength_gap'] = fixtures_df.get('home_season_ppg', 0) - fixtures_df.get('away_season_ppg', 0)
    fixtures_df['combined_draw_rate'] = fixtures_df.get('home_draw_rate_5', 0.25) + fixtures_df.get('away_draw_rate_5', 0.25)

    fixtures_df['imp_home'] = 1 / (1 + 10 ** (-(fixtures_df['elo_diff'] + HOME_ADV) / 400))
    fixtures_df['imp_away'] = 1 - fixtures_df['imp_home']
    fixtures_df['imp_draw'] = 0.26
    total = fixtures_df['imp_home'] + fixtures_df['imp_draw'] + fixtures_df['imp_away']
    fixtures_df['imp_home'] /= total
    fixtures_df['imp_draw'] /= total
    fixtures_df['imp_away'] /= total

    for col in FEATURES:
        if col in fixtures_df.columns:
            fixtures_df[col] = fixtures_df[col].fillna(fixtures_df[col].median())

    # Get classifier probabilities (used to calibrate Poisson lambdas)
    probs = model.predict_proba(fixtures_df[FEATURES])
    class_list = list(le.classes_)
    prob_home_arr = probs[:, class_list.index('Home Win')]
    prob_draw_arr = probs[:, class_list.index('Draw')]
    prob_away_arr = probs[:, class_list.index('Away Win')]

    # Dixon-Coles Poisson lambdas (independent for each side)
    # λ_home = home_attack × away_defense × league_avg_home_goals
    # λ_away = away_attack × home_defense × league_avg_away_goals
    home_attack_arr = np.array([team_stats[t]['home_attack'] for t in fixtures_df['Home']])
    away_attack_arr = np.array([team_stats[t]['away_attack'] for t in fixtures_df['Away']])
    home_defense_arr = np.array([team_stats[t]['home_defense'] for t in fixtures_df['Home']])
    away_defense_arr = np.array([team_stats[t]['away_defense'] for t in fixtures_df['Away']])

    lambda_home = np.clip(home_attack_arr * away_defense_arr * league_avg_home, 0.3, 4.0)
    lambda_away = np.clip(away_attack_arr * home_defense_arr * league_avg_away, 0.2, 3.5)

    # Simulate using hybrid approach:
    # 1. Draw independent Poisson goals for each fixture
    # 2. Use classifier probs to accept/reject: if Poisson scoreline disagrees
    #    with the classifier's sampled outcome, resample goals from a
    #    constrained Poisson that respects the outcome.
    # This preserves realistic goal distributions while respecting the
    # classifier's learned match probabilities.

    print(f"\n  Simulating {n_sims:,} seasons...", end="", flush=True)
    home_arr = fixtures_df['Home'].values
    away_arr = fixtures_df['Away'].values

    team_points = {t: [] for t in teams}
    team_wins = {t: [] for t in teams}
    team_draws_list = {t: [] for t in teams}
    team_losses = {t: [] for t in teams}
    team_gf = {t: [] for t in teams}
    team_ga = {t: [] for t in teams}
    team_positions = {t: [] for t in teams}

    np.random.seed(42)

    for sim in range(n_sims):
        if sim % 2500 == 0 and sim > 0:
            print(f" {sim//1000}k...", end="", flush=True)

        pts = {t: 0 for t in teams}
        w = {t: 0 for t in teams}
        d = {t: 0 for t in teams}
        l = {t: 0 for t in teams}
        gf = {t: 0 for t in teams}
        ga = {t: 0 for t in teams}

        # Sample all goals independently via Poisson
        home_goals = np.random.poisson(lambda_home)
        away_goals = np.random.poisson(lambda_away)

        # Determine outcome from classifier probabilities
        rands = np.random.random(n_fixtures)

        for i in range(n_fixtures):
            home, away_team = home_arr[i], away_arr[i]
            hg, ag = int(home_goals[i]), int(away_goals[i])

            # Classifier decides outcome
            if rands[i] < prob_home_arr[i]:
                # Home win — if Poisson disagrees, resample until home > away
                if hg <= ag:
                    # Resample from same lambdas, accept first home win scoreline
                    for _ in range(10):
                        hg = np.random.poisson(lambda_home[i])
                        ag = np.random.poisson(lambda_away[i])
                        if hg > ag:
                            break
                    else:
                        hg, ag = max(hg, 1), 0  # fallback
                pts[home] += 3; w[home] += 1; l[away_team] += 1

            elif rands[i] < prob_home_arr[i] + prob_draw_arr[i]:
                # Draw — resample independently until equal
                for _ in range(20):
                    hg = np.random.poisson(lambda_home[i])
                    ag = np.random.poisson(lambda_away[i])
                    if hg == ag:
                        break
                else:
                    # Fallback: use average of both lambdas
                    avg_lambda = (lambda_home[i] + lambda_away[i]) / 2
                    g = np.random.poisson(avg_lambda * 0.85)
                    hg, ag = g, g
                pts[home] += 1; pts[away_team] += 1; d[home] += 1; d[away_team] += 1

            else:
                # Away win — if Poisson disagrees, resample until away > home
                if ag <= hg:
                    for _ in range(10):
                        hg = np.random.poisson(lambda_home[i])
                        ag = np.random.poisson(lambda_away[i])
                        if ag > hg:
                            break
                    else:
                        hg, ag = 0, max(ag, 1)  # fallback
                pts[away_team] += 3; w[away_team] += 1; l[home] += 1

            gf[home] += hg; ga[home] += ag
            gf[away_team] += ag; ga[away_team] += hg

        for t in teams:
            team_points[t].append(pts[t])
            team_wins[t].append(w[t])
            team_draws_list[t].append(d[t])
            team_losses[t].append(l[t])
            team_gf[t].append(gf[t])
            team_ga[t].append(ga[t])

        ranked = sorted(teams, key=lambda t: (pts[t], gf[t]-ga[t], gf[t]), reverse=True)
        for pos, t in enumerate(ranked, 1):
            team_positions[t].append(pos)

    print(" Done!")

    # Build table
    n_teams = league_config['n_teams']
    releg_zone = n_teams - 2  # bottom 3: positions 18,19,20 for 20-team or 16,17,18 for 18-team

    table_data = []
    for team in teams:
        p = np.array(team_points[team])
        pos = np.array(team_positions[team])
        table_data.append({
            'Team': team, 'Avg_Pos': pos.mean(),
            'Pts': round(p.mean()), 'Pts_Low': int(np.percentile(p, 10)), 'Pts_High': int(np.percentile(p, 90)),
            'W': round(np.mean(team_wins[team])), 'D': round(np.mean(team_draws_list[team])), 'L': round(np.mean(team_losses[team])),
            'GF': round(np.mean(team_gf[team])), 'GA': round(np.mean(team_ga[team])),
            'GD': round(np.mean(team_gf[team]) - np.mean(team_ga[team])),
            'Title%': (pos == 1).mean() * 100,
            'Top4%': (pos <= 4).mean() * 100,
            'Releg%': (pos >= releg_zone).mean() * 100,
        })

    table = pd.DataFrame(table_data).sort_values('Avg_Pos').reset_index(drop=True)
    table.insert(0, 'Pos', range(1, len(table) + 1))
    return table


def print_table(table, league_name, acc, n_sims):
    print(f"\n{'='*95}")
    print(f"  PREDICTED {league_name.upper()} TABLE — 2025/26")
    print(f"  Accuracy: {acc*100:.1f}% | {n_sims:,} sims | 25 features | Dixon-Coles goals")
    print(f"{'='*95}")
    print(f"\n{'Pos':<4}{'Team':<22}{'Pts':<6}{'W':<5}{'D':<5}{'L':<5}{'GF':<5}{'GA':<5}{'GD':<6}{'80% Range':<12}{'Title':<8}{'Top4':<8}{'Releg'}")
    print(f"{'-'*95}")

    for _, row in table.iterrows():
        pts_range = f"{row['Pts_Low']}-{row['Pts_High']}"
        title = f"{row['Title%']:.1f}%" if row['Title%'] >= 0.1 else "—"
        top4 = f"{row['Top4%']:.1f}%" if row['Top4%'] >= 0.5 else "—"
        releg = f"{row['Releg%']:.1f}%" if row['Releg%'] >= 0.5 else "—"
        print(f"{row['Pos']:<4}{row['Team']:<22}{row['Pts']:<6}{row['W']:<5}{row['D']:<5}{row['L']:<5}"
              f"{row['GF']:<5}{row['GA']:<5}{row['GD']:<6}{pts_range:<12}{title:<8}{top4:<8}{releg}")

    print(f"\n{'-'*95}")
    print(f"  Win = 3 pts | Draw = 1 pt | Loss = 0 pts")
    print(f"  Goals: Independent Poisson (Dixon-Coles) | 80% Range = 10th-90th percentile")
    print(f"{'='*95}")


def main():
    print(f"\n{'═'*50}")
    print(f"  FOOTBALL SEASON PREDICTOR 2025/26")
    print(f"  GradientBoosting + Monte Carlo (10,000 sims)")
    print(f"{'═'*50}")
    print(f"\n  1. Premier League")
    print(f"  2. La Liga")
    print(f"  3. Serie A")
    print(f"  4. Bundesliga")
    print(f"  5. Ligue 1")
    print(f"  6. Brasileirão")
    print()

    choice = input("  Choose league (1-6): ").strip()
    if choice not in ['1', '2', '3', '4', '5', '6']:
        print("  Invalid choice. Exiting.")
        return

    league = LEAGUES[int(choice)]
    print(f"\n  → {league['name']} selected")
    print(f"  → {league['n_teams']} teams, {league['n_teams'] * (league['n_teams'] - 1)} fixtures")

    filepath = os.path.join(BASE_PATH, league['file'])
    print(f"\n  Loading {league['file']}...")
    df = pd.read_csv(filepath)

    max_year = df['Season_End_Year'].max()
    df = df[df['Season_End_Year'] < max_year].reset_index(drop=True)
    print(f"  {len(df):,} historical matches loaded (up to {max_year - 1})")

    print("  Building 25 features (Elo, form, odds, draws)...")
    df, elo_ratings = build_features(df)
    print(f"  {len(df):,} matches with complete features")

    print("  Training GradientBoosting model...")
    model, le, acc = train_model(df)

    print(f"\n{'='*60}")
    print(f"  MONTE CARLO — {league['name'].upper()} 2025/26")
    print(f"{'='*60}")
    table = simulate_season(model, le, df, elo_ratings, league, n_sims=10000)

    print_table(table, league['name'], acc, 10000)

    output_file = f"predicted_{league['name'].lower().replace(' ', '_').replace('ã', 'a')}_2026.csv"
    output_path = os.path.join(BASE_PATH, output_file)
    table.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved: {output_file}")


if __name__ == '__main__':
    main()


