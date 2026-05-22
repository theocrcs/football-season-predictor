# 🏆 Football Season Predictor 2025/26

Predicts full league tables for the 2025/26 season using machine learning and Monte Carlo simulation.

🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League • 🇪🇸 La Liga • 🇮🇹 Serie A • 🇩🇪 Bundesliga • 🇫🇷 Ligue 1 • 🇧🇷 Brasileirão

---

## How It Works

The predictor combines a **GradientBoosting classifier** with a **Dixon-Coles Poisson goal model** inside a **Monte Carlo simulation** (10,000 seasons) to generate realistic league table predictions with uncertainty ranges.

### Pipeline

```
Historical Data → Feature Engineering (25 features) → Train Classifier → Monte Carlo Simulation → Predicted Table
```

### Three-Phase Approach

1. **Feature Engineering** — Builds 25 match-level features from historical data (betting odds, Elo ratings, rolling form, goal stats, draw indicators)
2. **Classification** — GradientBoosting model trained to predict match outcomes (Home Win / Draw / Away Win) with ~52-55% accuracy
3. **Simulation** — 10,000 full seasons simulated using classifier probabilities + independent Poisson goal draws to produce a final table with confidence intervals

---

## Features Used (25)

| Category | Features | Purpose |
|----------|----------|---------|
| Betting odds | `imp_home`, `imp_draw`, `imp_away` | Market-derived win probabilities |
| Elo ratings | `home_elo`, `away_elo`, `elo_diff` | Long-term team strength |
| Goals (5-game) | `home_goals_avg`, `away_goals_avg`, `home_conceded_avg`, `away_conceded_avg` | Recent attacking/defensive form |
| Goals (10-game) | `home_goals_avg_10`, `away_goals_avg_10` | Stable form indicator |
| Goal difference | `home_gd_avg`, `away_gd_avg` | Overall dominance |
| Points form | `home_points_form`, `away_points_form` | Recent results |
| Season PPG | `home_season_ppg`, `away_season_ppg` | Season-long performance |
| Win rates | `home_win_rate_5`, `away_win_rate_5` | Win frequency |
| Draw indicators | `home_draw_rate_5`, `away_draw_rate_5`, `combined_draw_rate`, `elo_closeness` | Draw-prone matchups |
| Derived | `strength_gap` | PPG difference between teams |

---

## Model Details

- **Algorithm**: GradientBoostingClassifier (300 trees, depth 3, lr 0.05)
- **Class balancing**: Partial sqrt weights + 1.15× draw nudge
- **Train/test split**: 80/20 (random_state=42)
- **Goal model**: Dixon-Coles Poisson (independent λ for each team)
- **Simulations**: 10,000 full seasons per prediction
- **Odds fallback**: When B365 odds unavailable, uses Elo-derived implied probabilities

---

## Output

For each league, the predictor outputs:

| Metric | Description |
|--------|-------------|
| Pts | Average predicted points |
| W / D / L | Average wins, draws, losses |
| GF / GA / GD | Goals for, against, difference |
| 80% Range | 10th–90th percentile points range |
| Title% | Probability of winning the league |
| Top4% | Probability of finishing top 4 |
| Releg% | Probability of relegation |

---

## How to Run

### Requirements

```bash
pip install pandas numpy scikit-learn
```

### Run

```bash
python predict_season.py
```

```
══════════════════════════════════════════════════
  FOOTBALL SEASON PREDICTOR 2025/26
  GradientBoosting + Monte Carlo (10,000 sims)
══════════════════════════════════════════════════

  1. Premier League
  2. La Liga
  3. Serie A
  4. Bundesliga
  5. Ligue 1
  6. Brasileirão

  Choose league (1-6): _
```

---

## Data Sources

- Match results and betting odds: [football-data.co.uk](https://www.football-data.co.uk/)
- Seasons: 2010–2025 (varies by league)
- Odds: Bet365 (B365H/D/A), with average closing odds fallback for Brasileirão

---

## Project Structure

```
├── predict_season.py          # Main predictor script (all leagues)
├── premier-league-full.csv    # 🏴󠁧󠁢󠁥󠁮󠁧󠁿 6,448 matches (2010-2025)
├── la-liga-full.csv           # 🇪🇸 6,070 matches (2011-2025)
├── serie-a-full.csv           # 🇮🇹 6,065 matches (2011-2025)
├── bundesliga-full.csv        # 🇩🇪 4,896 matches (2011-2025)
├── ligue-1-full.csv           # 🇫🇷 5,753 matches (2011-2025)
├── brasileirao-full.csv       # 🇧🇷 5,476 matches (2012-2025)
└── README.md
```

---

## License

MIT
