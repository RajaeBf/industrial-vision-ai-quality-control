"""
Isole chaque produit d'une video de tapis roulant.

Principe :
- Soustraction de fond (MOG2) pour detecter les objets en mouvement.
- Un "tracker" tres simple associe les contours d'une frame a l'autre
  (par distance de centroide) pour suivre le MEME produit pendant
  qu'il traverse le champ de la camera.
- Quand un produit n'est plus vu pendant `max_missed` frames, on
  considere qu'il est sorti du champ : on garde son MEILLEUR crop
  (celui ou son aire est la plus grande, donc le plus net/centre)
  et on l'ajoute a la liste des produits a analyser.

C'est volontairement simple (pas de deep tracker) : suffisant pour
un tapis roulant avec un produit a la fois dans le champ, ou quelques
produits espaces. Si les produits se chevauchent souvent, il faudra
remplacer la detection de mouvement par un vrai detecteur d'objets
(YOLO) + un tracker (ByteTrack, DeepSORT...).
"""
import cv2
import numpy as np


class Track:
    def __init__(self, track_id, centroid, bbox, frame_img):
        self.id = track_id
        self.centroid = centroid
        self.bbox = bbox
        self.best_area = bbox[2] * bbox[3]
        self.best_crop = frame_img
        self.missed = 0


def _crop_with_margin(frame, bbox, margin=0.15):
    x, y, w, h = bbox
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(frame.shape[1], x + w + mx), min(frame.shape[0], y + h + my)
    return frame[y0:y1, x0:x1].copy()


class ProductTracker:
    """Tracker par soustraction de fond, utilisable frame par frame.

    Reutilisable pour un fichier video (boucle sur cv2.VideoCapture(path))
    OU pour un flux camera en direct (boucle sur cv2.VideoCapture(0) /
    une URL RTSP), puisque `update(frame)` traite une seule image a la fois
    et renvoie les produits qui viennent de "sortir du champ".
    """

    def __init__(self, min_area=3000, max_missed=10, max_match_dist=80, warmup_frames=30):
        self.min_area = min_area
        self.max_missed = max_missed
        self.max_match_dist = max_match_dist
        self.warmup_frames = warmup_frames
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False
        )
        self.tracks = {}
        self.next_id = 0
        self.frame_idx = 0

    def update(self, frame):
        """Traite une frame. Renvoie la liste des crops (np.array BGR) des
        produits qui viennent de se terminer sur cette frame (souvent vide)."""
        self.frame_idx += 1
        fg_mask = self.bg_subtractor.apply(frame)

        if self.frame_idx <= self.warmup_frames:
            return []

        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg_mask = cv2.dilate(fg_mask, np.ones((7, 7), np.uint8), iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w / 2, y + h / 2
            detections.append(((cx, cy), (x, y, w, h)))

        unmatched = list(detections)
        for tid, tr in list(self.tracks.items()):
            best, best_dist = None, self.max_match_dist
            for det in unmatched:
                dist = np.hypot(det[0][0] - tr.centroid[0], det[0][1] - tr.centroid[1])
                if dist < best_dist:
                    best, best_dist = det, dist
            if best is not None:
                centroid, bbox = best
                tr.centroid = centroid
                tr.missed = 0
                area = bbox[2] * bbox[3]
                if area > tr.best_area:
                    tr.best_area = area
                    tr.bbox = bbox
                    tr.best_crop = _crop_with_margin(frame, bbox)
                unmatched.remove(best)
            else:
                tr.missed += 1

        for centroid, bbox in unmatched:
            self.tracks[self.next_id] = Track(self.next_id, centroid, bbox, _crop_with_margin(frame, bbox))
            self.next_id += 1

        finished = []
        for tid in [tid for tid, tr in self.tracks.items() if tr.missed > self.max_missed]:
            finished.append(self.tracks.pop(tid).best_crop)
        return finished

    def flush(self):
        """A appeler en fin de flux : renvoie les tracks encore actifs
        (produits toujours visibles quand le flux s'est arrete)."""
        remaining = [tr.best_crop for tr in self.tracks.values()]
        self.tracks = {}
        return remaining


def extract_products(
    video_path,
    min_area=3000,
    max_missed=10,
    max_match_dist=80,
    frame_skip=1,
    warmup_frames=30,
):
    """Retourne une liste d'images (np.array BGR), une par produit detecte,
    pour un fichier video complet (mode hors-ligne). Utilise ProductTracker
    en interne."""
    cap = cv2.VideoCapture(video_path)
    tracker = ProductTracker(
        min_area=min_area, max_missed=max_missed,
        max_match_dist=max_match_dist, warmup_frames=warmup_frames,
    )
    finished_products = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue
        finished_products.extend(tracker.update(frame))

    finished_products.extend(tracker.flush())
    cap.release()
    return finished_products