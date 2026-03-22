# Dashboard Streamlit — Climat & Agriculture au Sénégal

Ce projet est une version dashboard de ton notebook geospatial Sénégal.

## Contenu
- visualisation des données climatiques et agricoles
- récupération météo via Open-Meteo
- modèle Random Forest de démonstration
- matrice de confusion, ROC-AUC, importance des variables
- simulateur de risque agricole
- carte simple par région

## Lancer le projet
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dépôt GitHub
```bash
git init
git add .
git commit -m "Initial commit - dashboard Streamlit Senegal"
git branch -M main
git remote add origin TON_URL_GITHUB
git push -u origin main
```

## Remarque
Si tu veux utiliser tes propres données, importe un CSV contenant au minimum :
- `rainfall_mm`
- `temperature_c`
- `ndvi`
- `humidity_pct`
- `soil_moisture_proxy`
- `risk`
