import pandas as pd
import numpy as np
import joblib
import ast
import os
from datetime import timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from core.utils import get_path_similarity_stats

class SmartPatchEngine:
    def __init__(self):
        print("⚡ Initializing SmartPatch Engine...")
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        self.datasets = {}
        self.models = {}
        self.thresholds = {}
        self.loaded_projects = []

    def load_project(self, project_key, csv_path, model_path):
        """Loads data for a specific project into memory."""
        if not os.path.exists(csv_path) or not os.path.exists(model_path):
            print(f"⚠️ Missing files for {project_key}. Skipping.")
            return

        print(f"   -> Loading {project_key} dataset...")
        df = pd.read_csv(csv_path, parse_dates=["created_time"], dtype={"patch_id": str})
        df = df.dropna(subset=["patch_id", "created_time"]).reset_index(drop=True)
        
        # Pre-parse file lists to save time later
        df['files_parsed'] = df['files'].apply(self._safe_parse_list)
        
        self.datasets[project_key] = df
        
        print(f"   -> Loading {project_key} model...")
        loaded_obj = joblib.load(model_path)
        if isinstance(loaded_obj, dict) and "model" in loaded_obj:
            self.models[project_key] = loaded_obj["model"]
            self.thresholds[project_key] = loaded_obj.get("threshold", 0.5)
        else:
            self.models[project_key] = loaded_obj
            self.thresholds[project_key] = 0.5
        self.loaded_projects.append(project_key)

    def _safe_parse_list(self, x):
        if isinstance(x, list): return x
        try: return ast.literal_eval(x)
        except: return []

    def get_candidates(self, project, ref_time, exclude_id, days=14):
        """Filters the dataset for changes within the time window."""
        df = self.datasets.get(project)
        if df is None: return pd.DataFrame()

        start = ref_time - timedelta(days=days)
        end = ref_time + timedelta(days=days)
        
        mask = (df.created_time >= start) & (df.created_time <= end) & (df.patch_id != exclude_id)
        return df[mask].copy()

    def predict(self, project, patch_ref, top_k=5, window_days=14):
        # 1. Get Candidates
        candidates = self.get_candidates(project, patch_ref['created_time'], patch_ref['patch_id'], window_days)
        if candidates.empty:
            return []

        # 2. Extract Features (Vectorized where possible)
        emb_target = self.sbert.encode((patch_ref['title'] or "") + " " + (patch_ref['description'] or ""))
        
        features_list = []
        ids = []

        # Optimization: Batch encode candidate texts if dataset is huge, 
        # but for top-k window, iterrows is acceptable and easier to debug.
        for _, row in candidates.iterrows():
            # Text Similarity
            emb_candidate = self.sbert.encode((row['title'] or "") + " " + (row['description'] or ""))
            sim_cosine = cosine_similarity(emb_target.reshape(1, -1), emb_candidate.reshape(1, -1))[0][0]
            
            # File Similarity
            file_stats = get_path_similarity_stats(row['files_parsed'], patch_ref['files'])
            
            # Combine
            feats = {
                **file_stats,
                "sim_cosine": sim_cosine,
                "delta_time_hours": abs((row['created_time'] - patch_ref['created_time']).total_seconds() / 3600),
                "len_A": len(row['files_parsed']),
                "len_B": len(patch_ref['files'])
            }
            # Fill missing required columns with 0 if model expects them
            # (You might need to adjust this based on exact columns your pickle model expects)
            
            features_list.append(feats)
            ids.append(row['patch_id'])

        if not features_list:
            return []

        X = pd.DataFrame(features_list)
        
        # 3. Predict
        model = self.models[project]
        # Align columns to model's expected features
        # Silent fail safety: Ensure X has columns model expects, fill missing with 0
        if hasattr(model, "feature_names_in_"):
            missing_cols = set(model.feature_names_in_) - set(X.columns)
            for c in missing_cols: X[c] = 0
            X = X[model.feature_names_in_]

        probs = model.predict_proba(X)[:, 1]
        threshold = self.thresholds.get(project, 0.5)

        # 4. Format Results
        results = []
        for pid, score, date, title in zip(ids, probs, candidates['created_time'], candidates['title']):
            results.append({
                "patch_id": pid,
                "score": float(score),
                "is_related": bool(score >= threshold),
                "created_time": date.strftime("%Y-%m-%d"),
                "title": title
            })

        # Sort and Slice
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]