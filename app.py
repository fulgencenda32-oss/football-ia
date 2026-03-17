
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Chargement des modèles ────────────────────────────────
MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DATA_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

print("Chargement des modèles...")
m_res     = joblib.load(f"{MODELS_PATH}/modele_resultat.pkl")
m_o25     = joblib.load(f"{MODELS_PATH}/modele_over25.pkl")
m_o15     = joblib.load(f"{MODELS_PATH}/modele_over15.pkl")
m_btts    = joblib.load(f"{MODELS_PATH}/modele_btts.pkl")
m_buts    = joblib.load(f"{MODELS_PATH}/modele_nb_buts.pkl")
m_bdom    = joblib.load(f"{MODELS_PATH}/modele_buts_dom.pkl")
m_bext    = joblib.load(f"{MODELS_PATH}/modele_buts_ext.pkl")
m_score   = joblib.load(f"{MODELS_PATH}/modele_score.pkl")
m_le      = joblib.load(f"{MODELS_PATH}/encodeur_score.pkl")
m_corners = joblib.load(f"{MODELS_PATH}/modele_corners.pkl")
m_oc      = joblib.load(f"{MODELS_PATH}/modele_over_corners.pkl")
m_cartons = joblib.load(f"{MODELS_PATH}/modele_cartons.pkl")
m_fautes  = joblib.load(f"{MODELS_PATH}/modele_fautes.pkl")
FEATURES  = joblib.load(f"{MODELS_PATH}/features.pkl")
print("Modèles chargés !")
gc.collect()

# ── Chargement données historiques ───────────────────────
df_hist = pd.read_csv(
    f"{DATA_PATH}/processed/football_complet.csv",
    parse_dates=["date"],
    low_memory=False
)
df_hist = df_hist.sort_values("date").reset_index(drop=True)

# ELO actuel de chaque équipe
elo_ratings = {}
for _, row in df_hist.iterrows():
    ed, ee = row["equipe_dom"], row["equipe_ext"]
    for e in [ed, ee]:
        if e not in elo_ratings: elo_ratings[e] = 1500
    elo_d = elo_ratings[ed]; elo_e = elo_ratings[ee]
    prob = 1 / (1 + 10**((elo_e-(elo_d+50))/400))
    sd   = 1 if row["resultat"]=="H" else (0.5 if row["resultat"]=="D" else 0)
    diff = abs(row["buts_dom"]-row["buts_ext"])
    fac  = np.log1p(diff)+1
    elo_ratings[ed] = elo_d + 32*fac*(sd-prob)
    elo_ratings[ee] = elo_e + 32*fac*((1-sd)-(1-prob))

# Historiques pour les features
historique      = {}
hist_forme_dom  = {}
hist_forme_ext  = {}
hist_fatigue    = {}
hist_dom_dict   = {}
hist_stats      = {}
hist_h2h        = {}
classements     = {}

for _, row in df_hist.iterrows():
    ed, ee = row["equipe_dom"], row["equipe_ext"]
    date   = row["date"]
    for e in [ed,ee]:
        for d in [historique,hist_forme_dom,hist_forme_ext,
                  hist_fatigue,hist_dom_dict,hist_stats]:
            if e not in d: d[e]=[]

    cle = tuple(sorted([ed,ee]))
    if cle not in hist_h2h: hist_h2h[cle]=[]

    ptsd=3 if row["resultat"]=="H" else(1 if row["resultat"]=="D" else 0)
    ptse=3 if row["resultat"]=="A" else(1 if row["resultat"]=="D" else 0)

    historique[ed].append({"pts":ptsd,"bm":row["buts_dom"],"be":row["buts_ext"]})
    historique[ee].append({"pts":ptse,"bm":row["buts_ext"],"be":row["buts_dom"]})
    hist_forme_dom[ed].append({"pts":ptsd,"bm":row["buts_dom"],"be":row["buts_ext"]})
    hist_forme_ext[ee].append({"pts":ptse,"bm":row["buts_ext"],"be":row["buts_dom"]})
    hist_fatigue[ed].append(date); hist_fatigue[ee].append(date)
    hist_dom_dict[ed].append(row["resultat"])

    for eq,bm,be in [(ed,row["buts_dom"],row["buts_ext"]),
                     (ee,row["buts_ext"],row["buts_dom"])]:
        entry={"bm":bm,"be":be}
        if eq==ed:
            entry["corners"]=row.get("corners_dom",np.nan)
            entry["fautes"]=row.get("fautes_dom",np.nan)
            entry["cartons"]=row.get("cartons_j_dom",np.nan)
            entry["tirs"]=row.get("tirs_dom",np.nan)
        else:
            entry["corners"]=row.get("corners_ext",np.nan)
            entry["fautes"]=row.get("fautes_ext",np.nan)
            entry["cartons"]=row.get("cartons_j_ext",np.nan)
            entry["tirs"]=row.get("tirs_ext",np.nan)
        hist_stats[eq].append(entry)

    hist_h2h[cle].append({
        "dom":ed,"ext":ee,
        "bm_dom":row["buts_dom"],"bm_ext":row["buts_ext"],
        "pts_dom":ptsd,"pts_ext":ptse
    })

    cle_cl = f"{row[chr(68)+'ivision']}_{row['Saison']}"
    if cle_cl not in classements: classements[cle_cl]={}
    cl=classements[cle_cl]
    for e in [ed,ee]:
        if e not in cl: cl[e]={"pts":0,"mj":0,"bp":0,"bc":0}
    cl[ed]["pts"]+=ptsd; cl[ed]["mj"]+=1
    cl[ed]["bp"]+=row["buts_dom"]; cl[ed]["bc"]+=row["buts_ext"]
    cl[ee]["pts"]+=ptse; cl[ee]["mj"]+=1
    cl[ee]["bp"]+=row["buts_ext"]; cl[ee]["bc"]+=row["buts_dom"]

print("Historiques calculés !")

# Liste équipes
toutes_equipes = sorted(set(
    df_hist["equipe_dom"].tolist()+df_hist["equipe_ext"].tolist()
))

# Performances modèle
PERFORMANCES = {
    "Résultat": 50.3,
    "Over 2.5": 52.8,
    "Over 1.5": 74.2,
    "BTTS": 52.8,
    "Score exact": 11.5,
    "Corners": 66.8
}

# Fichier historique prédictions
HISTO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historique", "predictions.json")
if not os.path.exists(HISTO_FILE):
    with open(HISTO_FILE, "w") as f:
        json.dump([], f)

def get_features(eq_dom, eq_ext, division):
    f = {}
    elo_d = elo_ratings.get(eq_dom, 1500)
    elo_e = elo_ratings.get(eq_ext, 1500)
    f["elo_dom"]=elo_d; f["elo_ext"]=elo_e
    f["elo_diff"]=elo_d-elo_e; f["elo_gap"]=abs(elo_d-elo_e)
    f["prob_elo_dom"]=1/(1+10**(-(f["elo_diff"]+50)/400))
    f["prob_bk_dom"]=f["prob_elo_dom"]
    f["prob_bk_nul"]=0.26
    f["prob_bk_ext"]=max(0,1-f["prob_elo_dom"]-0.26)
    f["ratio_elo_bk"]=1.0

    def gf(eq,n=10):
        h=historique.get(eq,[])[-n:]
        if not h: return 0,1.3,1.3,0.35,0.25,0
        pts=sum(x["pts"] for x in h)
        bm=np.mean([x["bm"] for x in h])
        be=np.mean([x["be"] for x in h])
        vic=sum(1 for x in h if x["pts"]==3)/len(h)
        cs=sum(1 for x in h if x["be"]==0)/len(h)
        s=0
        for x in reversed(h):
            if x["pts"]==3: s+=1
            elif x["pts"]==0: s-=1
            else: break
        return pts,bm,be,vic,cs,s

    fd=gf(eq_dom); fe=gf(eq_ext)
    f["forme_pts_dom"]=fd[0]; f["forme_pts_ext"]=fe[0]
    f["forme_bm_dom"]=fd[1];  f["forme_bm_ext"]=fe[1]
    f["forme_be_dom"]=fd[2];  f["forme_be_ext"]=fe[2]
    f["forme_vic_dom"]=fd[3]; f["forme_vic_ext"]=fe[3]
    f["forme_cs_dom"]=fd[4];  f["forme_cs_ext"]=fe[4]
    f["serie_dom"]=fd[5];     f["serie_ext"]=fe[5]
    f["diff_forme_pts"]=fd[0]-fe[0]

    def fl(h,n=5):
        x=h[-n:]
        if not x: return 0,1.3,1.3
        return sum(i["pts"] for i in x)/len(x),np.mean([i["bm"] for i in x]),np.mean([i["be"] for i in x])

    r1=fl(hist_forme_dom.get(eq_dom,[])); r2=fl(hist_forme_ext.get(eq_ext,[]))
    f["forme_dom_a_dom"]=r1[0]; f["forme_bm_dom_a_dom"]=r1[1]; f["forme_be_dom_a_dom"]=r1[2]
    f["forme_ext_a_ext"]=r2[0]; f["forme_bm_ext_a_ext"]=r2[1]; f["forme_be_ext_a_ext"]=r2[2]

    def mom(eq):
        h=historique.get(eq,[])
        if len(h)<3: return 0
        p3=np.mean([x["pts"] for x in h[-3:]])
        p10=np.mean([x["pts"] for x in h[-10:]]) if len(h)>=10 else p3
        return p3-p10

    f["momentum_dom"]=mom(eq_dom); f["momentum_ext"]=mom(eq_ext)
    f["diff_momentum"]=f["momentum_dom"]-f["momentum_ext"]

    from datetime import datetime
    now=pd.Timestamp(datetime.now())
    f["fatigue_dom"]=sum(1 for d in hist_fatigue.get(eq_dom,[]) if d>=now-pd.Timedelta(days=7))
    f["fatigue_ext"]=sum(1 for d in hist_fatigue.get(eq_ext,[]) if d>=now-pd.Timedelta(days=7))

    cle=tuple(sorted([eq_dom,eq_ext]))
    h=hist_h2h.get(cle,[])[-5:]
    if h:
        pl,bml,bel=[],[],[]
        for x in h:
            if x["dom"]==eq_dom:
                pl.append(x["pts_dom"]); bml.append(x["bm_dom"]); bel.append(x["bm_ext"])
            else:
                pl.append(x["pts_ext"]); bml.append(x["bm_ext"]); bel.append(x["bm_dom"])
        f["h2h_pts_dom"]=np.mean(pl)
        f["h2h_vic_dom"]=sum(1 for p in pl if p==3)/len(pl)
        f["h2h_nuls"]=sum(1 for p in pl if p==1)/len(pl)
        f["h2h_bm_dom"]=np.mean(bml); f["h2h_bm_ext"]=np.mean(bel)
    else:
        f["h2h_pts_dom"]=1.5; f["h2h_vic_dom"]=0.4
        f["h2h_nuls"]=0.25; f["h2h_bm_dom"]=1.3; f["h2h_bm_ext"]=1.1

    def st(eq,col,defaut,n=20):
        h=hist_stats.get(eq,[])[-n:]
        v=[x[col] for x in h if x.get(col) is not None and not np.isnan(x[col])]
        return np.mean(v) if v else defaut

    f["att_dom"]=st(eq_dom,"bm",1.3); f["att_ext"]=st(eq_ext,"bm",1.3)
    f["def_dom"]=st(eq_dom,"be",1.3); f["def_ext"]=st(eq_ext,"be",1.3)
    f["diff_att"]=f["att_dom"]-f["att_ext"]; f["diff_def"]=f["def_dom"]-f["def_ext"]
    f["xg_dom"]=f["att_dom"]*f["def_ext"]/1.3
    f["xg_ext"]=f["att_ext"]*f["def_dom"]/1.3
    f["xg_total"]=f["xg_dom"]+f["xg_ext"]
    f["moy_corners_dom"]=st(eq_dom,"corners",5.0); f["moy_corners_ext"]=st(eq_ext,"corners",4.5)
    f["moy_fautes_dom"]=st(eq_dom,"fautes",11.0);  f["moy_fautes_ext"]=st(eq_ext,"fautes",11.0)
    f["moy_cartons_dom"]=st(eq_dom,"cartons",1.5); f["moy_cartons_ext"]=st(eq_ext,"cartons",1.7)
    f["moy_tirs_dom"]=st(eq_dom,"tirs",12.0);      f["moy_tirs_ext"]=st(eq_ext,"tirs",11.0)

    div_map={"Premier_League":1,"Championship":2,"League_One":3}
    cle_cl=f"{division}_2024-25"
    cl=classements.get(cle_cl,{})
    if cl:
        et=sorted(cl.keys(),key=lambda e:(cl[e]["pts"],cl[e]["bp"]-cl[e]["bc"]),reverse=True)
        f["rang_dom"]=et.index(eq_dom)+1 if eq_dom in et else 10
        f["rang_ext"]=et.index(eq_ext)+1 if eq_ext in et else 10
        f["pts_classe_dom"]=cl.get(eq_dom,{}).get("pts",0)
        f["pts_classe_ext"]=cl.get(eq_ext,{}).get("pts",0)
        f["mj_dom"]=cl.get(eq_dom,{}).get("mj",0)
        f["mj_ext"]=cl.get(eq_ext,{}).get("mj",0)
    else:
        f["rang_dom"]=10; f["rang_ext"]=10
        f["pts_classe_dom"]=40; f["pts_classe_ext"]=40
        f["mj_dom"]=30; f["mj_ext"]=30
    f["diff_classement"]=f["pts_classe_dom"]-f["pts_classe_ext"]
    f["enjeu_dom"]=1 if f["rang_dom"]<=4 or f["rang_dom"]>=18 else 0
    f["enjeu_ext"]=1 if f["rang_ext"]<=4 or f["rang_ext"]>=18 else 0

    hd=hist_dom_dict.get(eq_dom,[])[-20:]
    f["avantage_dom"]=sum(1 for r in hd if r=="H")/max(len(hd),1)
    f["division_rank"]=div_map.get(division,1)
    f["mois"]=now.month; f["jour_semaine"]=now.weekday()
    f["periode_saison"]=1 if now.month in[8,9,10] else(2 if now.month in[11,12,1] else 3)

    X=np.array([[f.get(feat,0) for feat in FEATURES]])
    return X, f

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/equipes")
def get_equipes():
    return jsonify({"equipes": toutes_equipes})

@app.route("/api/predire", methods=["POST"])
def predire():
    data        = request.json
    eq_dom      = data.get("equipe_dom")
    eq_ext      = data.get("equipe_ext")
    division    = data.get("division","Premier_League")
    cote_dom    = data.get("cote_dom")
    cote_nul    = data.get("cote_nul")
    cote_ext    = data.get("cote_ext")

    if not eq_dom or not eq_ext:
        return jsonify({"error":"Équipes manquantes"}), 400

    X, feats = get_features(eq_dom, eq_ext, division)

    proba_res  = m_res.predict_proba(X)[0]
    buts_dom_p = max(0, float(m_bdom.predict(X)[0]))
    buts_ext_p = max(0, float(m_bext.predict(X)[0]))
    nb_buts    = max(0, float(m_buts.predict(X)[0]))
    prob_o25   = float(m_o25.predict_proba(X)[0][1])
    prob_o15   = float(m_o15.predict_proba(X)[0][1])
    prob_btts  = float(m_btts.predict_proba(X)[0][1])
    corners_p  = max(0, float(m_corners.predict(X)[0]))
    prob_oc    = float(m_oc.predict_proba(X)[0][1])
    cartons_p  = max(0, float(m_cartons.predict(X)[0]))
    fautes_p   = max(0, float(m_fautes.predict(X)[0]))
    proba_s    = m_score.predict_proba(X)[0]
    top5_idx   = np.argsort(proba_s)[::-1][:5]
    top5       = [(m_le.classes_[i], float(proba_s[i])) for i in top5_idx]

    # Fusion avec cotes si disponibles
    source = "Modèle IA"
    if cote_dom and cote_nul and cote_ext:
        try:
            p_d=1/float(cote_dom); p_n=1/float(cote_nul); p_e=1/float(cote_ext)
            tot=p_d+p_n+p_e; p_d/=tot; p_n/=tot; p_e/=tot
            proba_res[0]=proba_res[0]*0.55+p_d*0.45
            proba_res[1]=proba_res[1]*0.55+p_n*0.45
            proba_res[2]=proba_res[2]*0.55+p_e*0.45
            proba_res/=proba_res.sum()
            source="Modèle IA + Cotes"
        except: pass

    result = {
        "equipe_dom": eq_dom,
        "equipe_ext": eq_ext,
        "division":   division,
        "source":     source,
        "date":       datetime.now().strftime("%d/%m/%Y %H:%M"),
        "resultat": {
            "prediction": ["Victoire dom","Match nul","Victoire ext"][int(np.argmax(proba_res))],
            "prob_dom": round(float(proba_res[0])*100,1),
            "prob_nul": round(float(proba_res[1])*100,1),
            "prob_ext": round(float(proba_res[2])*100,1),
        },
        "buts": {
            "score_dom":  int(round(buts_dom_p)),
            "score_ext":  int(round(buts_ext_p)),
            "total":      round(nb_buts,1),
            "over_15":    round(prob_o15*100,1),
            "over_25":    round(prob_o25*100,1),
            "btts":       round(prob_btts*100,1),
        },
        "scores_exacts": [{"score":s,"proba":round(p*100,1)} for s,p in top5],
        "corners": {
            "total":   round(corners_p,1),
            "over_11": round(prob_oc*100,1),
        },
        "cartons": {
            "jaunes": round(cartons_p,1),
            "fautes": round(fautes_p,1),
        },
        "contexte": {
            "elo_dom":      round(feats["elo_dom"]),
            "elo_ext":      round(feats["elo_ext"]),
            "xg_dom":       round(feats["xg_dom"],2),
            "xg_ext":       round(feats["xg_ext"],2),
            "forme_dom":    int(feats["forme_pts_dom"]),
            "forme_ext":    int(feats["forme_pts_ext"]),
            "rang_dom":     int(feats["rang_dom"]),
            "rang_ext":     int(feats["rang_ext"]),
            "momentum_dom": round(feats["momentum_dom"],2),
            "momentum_ext": round(feats["momentum_ext"],2),
            "fatigue_dom":  int(feats["fatigue_dom"]),
            "fatigue_ext":  int(feats["fatigue_ext"]),
        }
    }

    # Sauvegarder dans historique
    try:
        with open(HISTO_FILE,"r") as f_h: histo=json.load(f_h)
        histo.insert(0, result)
        histo = histo[:100]  # Garder max 100
        with open(HISTO_FILE,"w") as f_h: json.dump(histo,f_h,ensure_ascii=False)
    except: pass

    return jsonify(result)

@app.route("/api/historique")
def get_historique():
    try:
        with open(HISTO_FILE,"r") as f: return jsonify(json.load(f))
    except: return jsonify([])

@app.route("/api/performances")
def get_performances():
    return jsonify(PERFORMANCES)

@app.route("/api/equipes_list")
def get_equipes_list():
    pl = sorted(df_hist[df_hist["Division"]=="Premier_League"]["equipe_dom"].unique().tolist())
    ch = sorted(df_hist[df_hist["Division"]=="Championship"]["equipe_dom"].unique().tolist())
    lo = sorted(df_hist[df_hist["Division"]=="League_One"]["equipe_dom"].unique().tolist())
    return jsonify({"Premier_League":pl,"Championship":ch,"League_One":lo})


@app.route("/api/sauver_resultat", methods=["POST"])
def sauver_resultat():
    try:
        data  = request.json
        index = data.get("index")
        real  = data.get("real_result")
        with open(HISTO_FILE,"r") as f: histo = json.load(f)
        if 0 <= index < len(histo):
            histo[index]["real_result"] = real
            with open(HISTO_FILE,"w") as f:
                json.dump(histo, f, ensure_ascii=False)
        return jsonify({"status":"ok"})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/mise_a_jour", methods=["POST"])
def mise_a_jour():
    try:
        import time
        API_KEY = "240c6fcac50b49aab377222fb903f12e"
        headers = {"X-Auth-Token": API_KEY}
        comps   = {
            "PL":  {"id":2021,"nom":"Premier_League"},
            "ELC": {"id":2016,"nom":"Championship"}
        }

        # Charger matchs existants
        try:
            df_ex = pd.read_csv(f"{DATA_PATH}/matchs_2025_2026.csv",
                                parse_dates=["date"])
            ids_ex = set(df_ex["id_match"].astype(str).tolist())
        except:
            df_ex  = pd.DataFrame()
            ids_ex = set()

        nouveaux = []
        for code, info in comps.items():
            url = (f"https://api.football-data.org/v4/competitions/"
                   f"{info[chr(105)+'d']}/matches?season=2025")
            r = requests.get(url, headers=headers)
            if r.status_code != 200:
                time.sleep(7); continue
            for m in r.json().get("matches",[]):
                if m["status"] != "FINISHED": continue
                if str(m["id"]) in ids_ex:    continue
                ft = m.get("score",{}).get("fullTime",{})
                bd, be = ft.get("home"), ft.get("away")
                if bd is None or be is None:  continue
                nouveaux.append({
                    "date":       m["utcDate"][:10],
                    "Saison":     "2025-26",
                    "Division":   info["nom"],
                    "equipe_dom": m["homeTeam"]["name"],
                    "equipe_ext": m["awayTeam"]["name"],
                    "buts_dom":   int(bd),
                    "buts_ext":   int(be),
                    "buts_total": int(bd)+int(be),
                    "resultat":   "H" if bd>be else("A" if bd<be else "D"),
                    "id_match":   m["id"],
                })
            time.sleep(7)

        if not nouveaux:
            return jsonify({
                "status":  "ok",
                "message": "Données déjà à jour !"
            })

        df_new = pd.DataFrame(nouveaux)
        df_new["date"] = pd.to_datetime(df_new["date"])
        if len(df_ex) > 0:
            df_maj = pd.concat([df_ex, df_new], ignore_index=True
                    ).drop_duplicates(
                        subset=["date","equipe_dom","equipe_ext"]
                    ).sort_values("date").reset_index(drop=True)
        else:
            df_maj = df_new

        df_maj.to_csv(f"{DATA_PATH}/matchs_2025_2026.csv", index=False)

        # Màj Drive
        try:
            df_maj.to_csv(
                "/content/drive/MyDrive/Football_IA/data/matchs_2025_2026.csv",
                index=False
            )
        except: pass

        return jsonify({
            "status":  "ok",
            "message": f"{len(nouveaux)} nouveaux matchs ajoutés ! "
                       f"Total : {len(df_maj)} matchs 2025/2026."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/joueurs/<equipe>")
def get_joueurs(equipe):
    try:
        df_j = pd.read_csv(f"{DATA_PATH}/joueurs_pl_championship.csv")
        joueurs = df_j[df_j["equipe"]==equipe][
            ["joueur","poste_std"]
        ].to_dict(orient="records")
        return jsonify({"joueurs": joueurs})
    except Exception as e:
        return jsonify({"joueurs": [], "error": str(e)})

@app.route("/api/predire_avec_compo", methods=["POST"])
def predire_avec_compo():
    try:
        data     = request.json
        eq_dom   = data.get("equipe_dom")
        eq_ext   = data.get("equipe_ext")
        division = data.get("division","Premier_League")
        tit_dom  = data.get("titulaires_dom", [])
        tit_ext  = data.get("titulaires_ext", [])
        cote_dom = data.get("cote_dom")
        cote_nul = data.get("cote_nul")
        cote_ext = data.get("cote_ext")

        # Calcul features de base
        X, feats = get_features(eq_dom, eq_ext, division)

        # Calcul poids compositions si disponibles
        poids_compo = None
        if tit_dom and tit_ext:
            try:
                df_j = pd.read_csv(
                    f"{DATA_PATH}/joueurs_pl_championship.csv"
                )
                valeurs = dict(zip(df_j["joueur"], df_j["valeur_M"]))                     if "valeur_M" in df_j.columns else {}
                postes  = dict(zip(df_j["joueur"], df_j["poste_std"]))

                poids_poste = {
                    "GK":      {"att":0.0, "def":1.5},
                    "DEF":     {"att":0.2, "def":1.2},
                    "MID_DEF": {"att":0.4, "def":0.8},
                    "MID":     {"att":0.7, "def":0.7},
                    "MID_ATT": {"att":1.0, "def":0.4},
                    "ATT":     {"att":1.5, "def":0.1},
                }

                def calc_poids(titulaires):
                    att_t=def_t=val_t=val_att=val_def=0
                    for j in titulaires:
                        nom    = j.get("joueur","")
                        poste  = postes.get(nom, j.get("poste","MID"))
                        valeur = valeurs.get(nom, 10)
                        p      = poids_poste.get(poste,{"att":0.7,"def":0.7})
                        att_t += p["att"]*valeur
                        def_t += p["def"]*valeur
                        val_t += valeur
                        if poste in ["ATT","MID_ATT"]: val_att += valeur
                        elif poste in ["DEF","GK"]:    val_def += valeur
                    if val_t == 0: return None
                    return {
                        "score_att":    round(att_t/val_t*10, 2),
                        "score_def":    round(def_t/val_t*10, 2),
                        "score_global": round((att_t+def_t)/val_t*10/2, 2),
                        "valeur_totale":round(val_t, 0),
                        "valeur_att":   round(val_att, 0),
                        "valeur_def":   round(val_def, 0),
                    }

                pd_dom = calc_poids(tit_dom)
                pd_ext = calc_poids(tit_ext)

                if pd_dom and pd_ext:
                    poids_compo = {"dom": pd_dom, "ext": pd_ext}
                    diff_glob   = pd_dom["score_global"]-pd_ext["score_global"]

                    # Ajuster xG avec compositions
                    fact_d = pd_dom["score_att"]/5.0
                    fact_e = pd_ext["score_att"]/5.0
                    if "xg_dom" in FEATURES:
                        X[0][FEATURES.index("xg_dom")] =                             feats["xg_dom"]*(fact_d*0.4+0.6)
                    if "xg_ext" in FEATURES:
                        X[0][FEATURES.index("xg_ext")] =                             feats["xg_ext"]*(fact_e*0.4+0.6)

                    # Ajuster probabilités résultat
                    ajust = max(-0.05, min(0.05, diff_glob*0.02))
                    feats["ajust_compo"] = ajust

            except Exception as e:
                print(f"Erreur compo: {e}")

        # Prédictions
        proba_res  = m_res.predict_proba(X)[0]
        buts_dom_p = max(0, float(m_bdom.predict(X)[0]))
        buts_ext_p = max(0, float(m_bext.predict(X)[0]))
        nb_buts    = max(0, float(m_buts.predict(X)[0]))
        prob_o25   = float(m_o25.predict_proba(X)[0][1])
        prob_o15   = float(m_o15.predict_proba(X)[0][1])
        prob_btts  = float(m_btts.predict_proba(X)[0][1])
        corners_p  = max(0, float(m_corners.predict(X)[0]))
        prob_oc    = float(m_oc.predict_proba(X)[0][1])
        cartons_p  = max(0, float(m_cartons.predict(X)[0]))
        fautes_p   = max(0, float(m_fautes.predict(X)[0]))
        proba_s    = m_score.predict_proba(X)[0]
        top5_idx   = np.argsort(proba_s)[::-1][:5]
        top5       = [(m_le.classes_[i],
                       float(proba_s[i])) for i in top5_idx]

        # Ajustement compositions
        if poids_compo and "ajust_compo" in feats:
            ajust = feats["ajust_compo"]
            proba_res[0] = np.clip(proba_res[0]+ajust, 0.05, 0.90)
            proba_res[2] = np.clip(proba_res[2]-ajust, 0.05, 0.90)
            proba_res    = proba_res / proba_res.sum()

        # Fusion cotes
        source = "Modèle IA"
        if tit_dom: source += " + Compositions"
        if cote_dom and cote_nul and cote_ext:
            try:
                p_d=1/float(cote_dom); p_n=1/float(cote_nul)
                p_e=1/float(cote_ext)
                tot=p_d+p_n+p_e; p_d/=tot; p_n/=tot; p_e/=tot
                proba_res[0]=proba_res[0]*0.55+p_d*0.45
                proba_res[1]=proba_res[1]*0.55+p_n*0.45
                proba_res[2]=proba_res[2]*0.55+p_e*0.45
                proba_res/=proba_res.sum()
                source += " + Cotes"
            except: pass

        result = {
            "equipe_dom": eq_dom,
            "equipe_ext": eq_ext,
            "division":   division,
            "source":     source,
            "date":       datetime.now().strftime("%d/%m/%Y %H:%M"),
            "compositions": poids_compo,
            "resultat": {
                "prediction": ["Victoire dom","Match nul",
                               "Victoire ext"][int(np.argmax(proba_res))],
                "prob_dom": round(float(proba_res[0])*100, 1),
                "prob_nul": round(float(proba_res[1])*100, 1),
                "prob_ext": round(float(proba_res[2])*100, 1),
            },
            "buts": {
                "score_dom": int(round(buts_dom_p)),
                "score_ext": int(round(buts_ext_p)),
                "total":     round(nb_buts, 1),
                "over_15":   round(prob_o15*100, 1),
                "over_25":   round(prob_o25*100, 1),
                "btts":      round(prob_btts*100, 1),
            },
            "scores_exacts": [
                {"score":s, "proba":round(p*100,1)} for s,p in top5
            ],
            "corners": {
                "total":   round(corners_p, 1),
                "over_11": round(prob_oc*100, 1),
            },
            "cartons": {
                "jaunes": round(cartons_p, 1),
                "fautes": round(fautes_p, 1),
            },
            "contexte": {
                "elo_dom":      round(feats["elo_dom"]),
                "elo_ext":      round(feats["elo_ext"]),
                "xg_dom":       round(feats["xg_dom"], 2),
                "xg_ext":       round(feats["xg_ext"], 2),
                "forme_dom":    int(feats["forme_pts_dom"]),
                "forme_ext":    int(feats["forme_pts_ext"]),
                "rang_dom":     int(feats["rang_dom"]),
                "rang_ext":     int(feats["rang_ext"]),
                "momentum_dom": round(feats["momentum_dom"], 2),
                "momentum_ext": round(feats["momentum_ext"], 2),
                "fatigue_dom":  int(feats["fatigue_dom"]),
                "fatigue_ext":  int(feats["fatigue_ext"]),
            }
        }

        # Sauvegarder historique
        try:
            with open(HISTO_FILE,"r") as f_h: histo=json.load(f_h)
            histo.insert(0, result); histo=histo[:100]
            with open(HISTO_FILE,"w") as f_h:
                json.dump(histo, f_h, ensure_ascii=False)
        except: pass

        return jsonify(result)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e),
                        "detail": traceback.format_exc()}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
