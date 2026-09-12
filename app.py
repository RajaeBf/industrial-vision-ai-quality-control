"""
Plateforme de test qualite - PatchCore.

Lancement :
    pip install -r requirements.txt
    streamlit run app.py

Le checkpoint attendu est est dans le dossier drive :
02_doc du projet/checkpoint/<categorie>/<categorie>.ckpt
"""
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import gdown

from patchcore_inference import load_model, analyze_product

st.set_page_config(page_title="Controle qualite - PatchCore", layout="wide")
st.title("🔍 Controle qualite automatique — PatchCore")

# Checkpoints charges automatiquement au demarrage de la plateforme.
# Format : {nom_categorie: URL de telechargement direct}
CKPT_URLS = {
    "cable": "1xqSKlTAGl5DscPSt47M4hBMyUOwn7Fq3",
    "bouteille": "1lejiSBVlJ1jz-8ivXGvEElDc1EmEeKsh",
}
# Seuil de decision PAR DEFAUT par categorie (modifiable dans la sidebar)
DEFAULT_THRESHOLDS = {
    "cable": 42.00,
    "bouteille": 30.00,
}
CKPT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "patchcore_ckpts")
os.makedirs(CKPT_CACHE_DIR, exist_ok=True)


@st.cache_resource(show_spinner="Téléchargement du checkpoint (une seule fois par catégorie)...")
def _download_ckpt(category: str, file_id: str) -> str:
    """Télécharge le checkpoint via gdown, contourne automatiquement la page de confirmation des gros fichiers."""
    dest = os.path.join(CKPT_CACHE_DIR, f"{category}.ckpt")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    
    gdown.download(id=file_id, output=dest, quiet=False)
    return dest

with st.sidebar:
    st.header("Configuration")
    ckpt_source_path = None
    if CKPT_URLS:
        category = st.selectbox("Categorie de produit", sorted(CKPT_URLS.keys()))
        try:
            ckpt_source_path = _download_ckpt(category, CKPT_URLS[category])
            st.success(f"✅ Checkpoint '{category}' charge automatiquement")
        except Exception as e:
            st.error(f"Echec du telechargement automatique du checkpoint : {e}")

    with st.expander("Utiliser un autre checkpoint (upload manuel)", expanded=ckpt_source_path is None):
        ckpt_file = st.file_uploader("Checkpoint entraine (.ckpt)", type=["ckpt"])
        if ckpt_file is not None:
            with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
                f.write(ckpt_file.read())
                ckpt_source_path = f.name

    # Seuil modifiable, pre-rempli avec la valeur par defaut de la categorie
    if CKPT_URLS:
        default_threshold = DEFAULT_THRESHOLDS.get(category, 0.5)
    else:
        default_threshold = 0.5

    threshold = st.number_input(
        "Seuil de decision (score d'anomalie)",
        min_value=0.0,
        max_value=1000.0,
        value=float(default_threshold),
        step=0.1,
        key=f"threshold_{category if CKPT_URLS else 'default'}",   # reset auto quand on change de categorie
        help=(
            "Le score brut Patchcore n'est PAS borne entre 0 et 1. Utilise "
            "l'onglet 'Calibrer le seuil' pour obtenir une valeur fiable, "
            "puis reporte-la ici."
        ),
    )
    st.caption(f"Seuil par defaut pour cette categorie : **{default_threshold:.2f}** (modifiable ci-dessus)")
    st.divider()

tab_images, tab_calib = st.tabs(["🖼️ Images individuelles", "🎯 Calibrer le seuil"])


def _run_analysis(model, device, items, threshold):
    """items : liste de (nom, image_bgr)."""
    n_ok, n_defaut = 0, 0
    scores = []
    for i, (name, crop) in enumerate(items):
        result = analyze_product(model, device, crop, threshold=threshold)
        scores.append(result["score"])
        n_ok += result["status"] == "OK"
        n_defaut += result["status"] == "DEFAUT"

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=name)
        with col2:
            badge = "✅ OK" if result["status"] == "OK" else "❌ DEFAUT"
            st.metric("Statut", badge)
            st.write(f"Score d'anomalie : `{result['score']:.6f}`")
            if result["defect_bbox"]:
                x, y, w, h = result["defect_bbox"]
                st.write(f"Position du defaut (x, y, w, h) : `({x}, {y}, {w}, {h})`")
        with col3:
            st.image(
                cv2.cvtColor(result["heatmap_overlay"], cv2.COLOR_BGR2RGB),
                caption="Carte de chaleur + zone du defaut",
            )
        if i == 0:
            with st.expander("🔧 Debug (premier produit) — type et contenu brut de la sortie modele"):
                st.write(f"Type : `{result['debug_raw_output_type']}`")
                st.code(result["debug_raw_output_repr"])
        st.divider()

    st.subheader("Resume")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total analyse", len(items))
    c2.metric("Bons", n_ok)
    c3.metric("Defectueux", n_defaut)
    c4.metric("Plage de scores", f"{min(scores):.3f} — {max(scores):.3f}")
    if max(scores) - min(scores) < 1e-6:
        st.error(
            "Tous les scores sont identiques : le checkpoint a probablement un "
            "probleme de calibration plutot qu'un probleme de pipeline."
        )


with tab_images:
    st.caption(
        "Chaque image uploadee est traitee comme UN SEUL produit. "
        "Uploade une ou plusieurs images, puis lance l'analyse."
    )
    image_files = st.file_uploader(
        "Images de produits (une image = un produit)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="images",
    )
    run_images = st.button("Lancer l'analyse (images)", type="primary")

    if run_images:
        if ckpt_source_path is None or not image_files:
            st.error("Merci d'avoir un checkpoint charge (auto ou upload) ET au moins une image.")
            st.stop()

        with st.spinner("Chargement du modele..."):
            model, device = load_model(ckpt_source_path)

        items = []
        for img_file in image_files:
            file_bytes = np.frombuffer(img_file.read(), np.uint8)
            bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if bgr is None:
                st.warning(f"Impossible de lire {img_file.name}, ignoree.")
                continue
            items.append((img_file.name, bgr))

        if not items:
            st.warning("Aucune image valide.")
            st.stop()

        st.success(f"{len(items)} image(s) chargee(s). Analyse en cours...")
        _run_analysis(model, device, items, threshold)
    else:
        st.info("Uploade un checkpoint (sidebar) et une ou plusieurs images, puis clique sur 'Lancer l'analyse (images)'.")

with tab_calib:
    st.caption(
        "Uploade une serie d'images que tu sais etre BONNES (issues de ton set de "
        "test/validation, statut good). L'appli calcule leurs scores bruts et te "
        "suggere un seuil place juste au-dessus de la pire d'entre elles, avec une marge. "
        "Plus tu fournis d'images good variees (differents angles/reflets), plus le seuil "
        "sera fiable."
    )
    good_files = st.file_uploader(
        "Images 'good' connues (plusieurs)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="calib_good",
    )
    margin_pct = 4 # Marge fixe de 4%
    run_calib = st.button("Calculer le seuil suggere", type="primary")

    if run_calib:
        if ckpt_source_path is None or not good_files:
            st.error("Merci d'avoir un checkpoint charge (auto ou upload) ET plusieurs images 'good'.")
            st.stop()

        with st.spinner("Chargement du modele..."):
            model, device = load_model(ckpt_source_path)

        good_scores = []
        rows = []
        for img_file in good_files:
            file_bytes = np.frombuffer(img_file.read(), np.uint8)
            bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            result = analyze_product(model, device, bgr, threshold=1e18)  # jamais DEFAUT ici, on veut juste le score
            good_scores.append(result["score"])
            rows.append((img_file.name, result["score"]))

        if not good_scores:
            st.warning("Aucune image valide.")
            st.stop()

        good_scores = np.array(good_scores)
        worst_good = float(good_scores.max())
        suggested_threshold = worst_good * (1 + margin_pct / 100)

        st.subheader("Resultats")
        st.write(
            f"Scores sur {len(good_scores)} image(s) good : "
            f"min = `{good_scores.min():.3f}`, moyenne = `{good_scores.mean():.3f}`, "
            f"max (pire cas) = `{worst_good:.3f}`"
        )
        st.success(f"➡️ Seuil suggere : `{suggested_threshold:.3f}` (pire score good + {margin_pct}% de marge)")
        st.caption(
            "Reporte cette valeur dans le champ 'Seuil de decision' de la sidebar avant "
            "de lancer tes analyses dans l'onglet Images individuelles. Si tes images good "
            "sont peu variees (toujours le meme angle/eclairage), ce seuil restera fragile — "
            "fournis-en davantage si possible, avec des reflets/eclairages differents."
        )
        with st.expander("Detail par image"):
            for name, score in sorted(rows, key=lambda r: -r[1]):
                st.write(f"`{score:.3f}` — {name}")