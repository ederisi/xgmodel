# 🏒 NHL Expected Goals (xG) & Expected Win (xWin) Analytics Suite

Modulaarinen koneoppimis- ja syväoppimisjärjestelmä NHL-laukausten maaliodottaman (**xG**), ottelukohtaisten voittotodennäköisyyksien (**xWin%**) ja maalivahtien torjuntatehokkuuden (**GSAX**) mallintamiseen. Sovellus sisältää interaktiivisen **Streamlit**-käyttöliittymän ja vertaa **8 eri tekoälymallia + Ensemble-yhdistelmämallia** suoraan MoneyPuckin viralliseen `xGoal`-malliin.

---

## ✨ Keskeiset ominaisuudet

* **🤖 8 ML/DL-mallin arkkitehtuuri + Ensemble:**
  * **Puupohjaiset & Gradient Boosting -mallit:** `XGBoost`, `HistGradientBoosting`, `Random Forest`.
  * **Aikasarja- ja syväoppimismallit (PyTorch):** `1D-CNN` (konvoluutioverkko BatchNorm- ja Logit-kalibroinnilla) sekä aidot sekvenssipohjaiset `Sequence LSTM`- ja `Sequence GRU`-mallit, jotka analysoivat ottelun 4 peräkkäisen laukauksen ketjuja (`seq_len=4`).
  * **Klassiset & Neuroverkkomallit (Scikit-Learn):** `Logistic Regression`, `MLP Neural Network`.
  * **Log-Odds Bias -kalibrointi:** PyTorch-mallien ulostulot kalibroidaan automaattisesti vastaamaan koulutusdatan todellista maalikeskiarvoa.
* **⚡ Edistynyt piirteiden erottelu (Feature Engineering):**
  * Hyödyntää laukaisuetäisyyttä, koordinaatteja (`X`, `Y`), automaattisesti laskettua laukaisukulmaa (`shotAngleAdjusted`), kätisyyttä (`offWing`), laukaustyyppiä (`shotType`), irtokiekkoja (`shotRebound`), vastahyökkäyksiä (`shotRush`), joukkuetta ja maalivahtitasoa (`goalieTier`).
* **🎯 Interaktiivinen Laukaisukartta & Tilanneanalyysi (`app.py`):**
  * Säädä laukauksen paikkaa ja pelitilannetta reaaliajassa hyökkäysalueen kenttäkartalla ja vertaa kaikkien 8 mallin antamia xG-arvoja sekä pelitilanteen (Rush / Rebound) vaikutusta.
* **🎲 Monte Carlo xWin -ottelusimulaattori (`pages/xwin_simulator.py` & `xwin_simul.py`):**
  * Simuloi ottelun lopputulokset 10 000 kertaa yksittäisten laukausten xG-arvojen perusteella ja laskee molemmille joukkueille voittotodennäköisyyden (`xWin%`).
* **📈 Uuden kauden validointi vs. MoneyPuck (`pages/season_valid.py`):**
  * Lataa uuden kauden CSV-tiedosto (esim. `shots_2026.csv`) ja vertaa mallien tarkkuutta (**Log Loss**, **Brier Score**, **Predicted Total xG** ja **xWin Winner Accuracy**) MoneyPuckin malliin.
* **🥅 Maalivahtien GSAX-analyysi (`pages/goalie_gsax.py`):**
  * Laskee maalivahtikohtaisen **Goals Saved Above Expected (GSAX = Kohdattu xG − Päästetyt maalit)** -tilaston valitulla mallilla ja suodattaa tulokset kohdattujen laukausten määrän mukaan.

---

## 📁 Project structure

```text
xgmodel/
│
├── app.py                        # Pääsivu: Interaktiivinen laukaisukartta, 8 mallin xG-vertailu & Leaderboard
├── main.py                       # Käynnistysskripti Streamlit-palvelimelle
├── model_loader.py               # Alustaa, kouluttaa ja välimuistittaa (@st.cache_resource) kaikki 8 mallia
├── xwin_simul.py                 # Monte Carlo -ottelusimulaattorin laskentalogiikka (GameWinSimulator)
│
├── pages/                        # Streamlitin monisivuiset analyysinäkymät
│   ├── goalie_gsax.py            # Maalivahtien GSAX-analyysi ja Top/Bottom-visualisointi
│   ├── season_valid.py           # Uuden kauden CSV-validointi (Log Loss, Brier Score & xWin vs. MoneyPuck)
│   └── xwin_simulator.py         # Ottelukohtainen xWin-simulaattori käyttöliittymällä
│
├── trainer/                      # Modulaariset koneoppimis- ja syväoppimismallit
│   ├── __init__.py               # Paketin alustus
│   ├── xg_trainer.py             # XGBoost-malli, datan lataus ja piirteiden valmistelu
│   ├── sklearn_pre.py            # StandardScaler, One-Hot-koodaus & 3D-sekvenssien (seq_len=4) rakennus
│   ├── cnn_trainer.py            # PyTorch 1D-CNN -malli + Log-Odds-kalibrointi
│   ├── lstm_trainer.py           # PyTorch True Sequence LSTM -malli
│   ├── gru_trainer.py            # PyTorch True Sequence GRU -malli
│   ├── hgb_trainer.py            # Scikit-Learn HistGradientBoosting -malli
│   ├── rf_trainer.py             # Scikit-Learn Random Forest -malli
│   ├── logreg_trainer.py         # Scikit-Learn Logistic Regression -malli
│   └── mlp_trainer.py            # Scikit-Learn Multi-Layer Perceptron (MLP) -neuroverkko
│
├── shots.csv                     # Koulutusdata (MoneyPuck NHL shot data)
├── shots_2026.csv                # Testi-/validointikauden laukausdata
└── team_x_goals_summary.csv      # Joukkuetason yhteenvetodata

## How to setup
- Ensure you have Python 3.10+ version
- download all needed libraries (with command): pip install streamlit pandas numpy scikit-learn xgboost torch matplotlib
- Load data, I utilized Moneypuck's shot data. Place this inside the project.
- If you want to utilize another season's data for validation, you can upload it straight to either the season_val or goalie_gsax tab

## How to run
- Either by py(can be python or similar depends on your pc configurations) py main.py or streamlit app.py
- (The app opens on your browser)
- The system evaluates the models based on log loss and brier score. The models located in the trainer file,
- to find suitable performance.