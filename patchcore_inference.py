"""
Charge un checkpoint Patchcore entraine (celui produit par ton notebook,
ANOMALIB_ROOT/Patchcore/<categorie>/last.ckpt) et l'utilise pour analyser
une image de produit isole.

NOTE IMPORTANTE : le format exact de la sortie du modele peut varier
legerement selon ta version d'anomalib (2.x). Le code ci-dessous essaie
plusieurs noms d'attributs usuels (pred_score/anomaly_map ou pred_score
via dict). Si ta version renvoie un format different, adapte juste la
fonction `_unpack_output`.
"""
import cv2
import numpy as np
import torch
from torchvision import transforms

from anomalib.models import Patchcore

IMG_SIZE = (256, 256)  # taille par defaut utilisee a l'entrainement dans ton notebook

_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    # PyTorch 2.6 a change la valeur par defaut de `weights_only` a True dans torch.load,
    # ce qui bloque le chargement des checkpoints anomalib (objets custom comme PrecisionType).
    # On force temporairement weights_only=False (checkpoint de confiance, entraine par nos soins).
    _original_torch_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_load
    try:
        model = Patchcore.load_from_checkpoint(ckpt_path, map_location=device)
    finally:
        torch.load = _original_torch_load

    model.eval()
    model.to(device)
    return model, device


def _unpack_output(output):
    """Renvoie (score, anomaly_map) quel que soit le format retourne."""
    if hasattr(output, "pred_score"):
        score = float(output.pred_score)
        amap = output.anomaly_map
    elif isinstance(output, dict):
        score = float(output.get("pred_score", output.get("pred_scores")))
        amap = output.get("anomaly_map", output.get("anomaly_maps"))
    else:
        raise ValueError(f"Format de sortie non reconnu : {type(output)}")

    if isinstance(amap, torch.Tensor):
        amap = amap.squeeze().detach().cpu().numpy()
    return score, amap


@torch.no_grad()
def analyze_product(model, device, bgr_image, threshold=0.5, defect_pct=97.0, use_raw_score=True):
    """
    Renvoie un dict avec :
      - status : "OK" ou "DEFAUT"
      - score  : score d'anomalie (plus haut = plus anormal)
      - heatmap_overlay : image BGR avec la heatmap superposee
      - defect_bbox : (x, y, w, h) dans le repere de `bgr_image`, ou None

    `use_raw_score=True` (par defaut) : on appelle `model.model(tensor)`, le
    sous-module PyTorch pur de Patchcore, en CONTOURNANT la couche
    Lightning `model(tensor)` qui applique une normalisation min-max
    interne. Si le checkpoint n'a pas de stats de normalisation valides
    (min/max jamais calibres pendant l'entrainement, ex: `trainer.test()`
    jamais lance), cette normalisation ecrase tous les scores a 1.0 -
    exactement le symptome observe. Le score brut n'est PAS borne dans
    [0, 1] : ajuste le seuil en consequence (regarde les valeurs affichees
    pour calibrer).
    """
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    tensor = _TRANSFORM(rgb).unsqueeze(0).to(device)

    if use_raw_score and hasattr(model, "model"):
        output = model.model(tensor)
    else:
        output = model(tensor)
    score, amap = _unpack_output(output)

    status = "DEFAUT" if score >= threshold else "OK"

    # remet la heatmap a la taille originale de l'image
    amap_resized = cv2.resize(amap, (bgr_image.shape[1], bgr_image.shape[0]))
    amap_norm = cv2.normalize(amap_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(amap_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr_image, 0.6, heatmap_color, 0.4, 0)

    defect_bbox = None
    if status == "DEFAUT":
        thresh_val = np.percentile(amap_norm, defect_pct)
        _, mask = cv2.threshold(amap_norm, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            defect_bbox = cv2.boundingRect(biggest)
            x, y, w, h = defect_bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return {
        "status": status,
        "score": score,
        "heatmap_overlay": overlay,
        "defect_bbox": defect_bbox,
        "debug_raw_output_type": type(output).__name__,
        "debug_raw_output_repr": repr(output)[:2000],
    }