"""
SmartPatch RAG API — Flask application entry-point.

Routes
------
GET  /health          — Liveness check; lists loaded projects.
POST /predict_topk    — Find top-k similar patches for a given patch ID.
"""

from __future__ import annotations

import logging
import os
import traceback

from flask import Flask, jsonify, request
from flask_cors import CORS

import config
from core.gerrit import GerritClient
from core.improved_rag_engine import ImprovedRAGEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app)

    # --- Initialise RAG engine ---
    engine = ImprovedRAGEngine()

    for project in config.PROJECTS:
        csv_path = os.path.join(config.DATA_DIR, project, "all_candidates.csv")
        engine.load_project(project, csv_path)

    # Attach engine to app so routes can access it
    app.engine = engine  # type: ignore[attr-defined]

    # --- Register routes ---
    @app.route("/health", methods=["GET"])
    def health():
        """Liveness probe — returns loaded project list."""
        return jsonify({
            "status": "active",
            "loaded_projects": engine.loaded_projects,
        })

    @app.route("/predict_topk", methods=["POST"])
    def predict_topk():
        """Find top-k semantically similar patches for a given patch ID.

        Request body (JSON)
        -------------------
        project     : str  — Project key (e.g. ``"onap"``).
        patch_id    : str  — Target patch ID.
        time_window : int  — Time window in days (default: ``14``).
        top_k       : int  — Number of results (default: ``5``).
        strategy    : str  — Retrieval strategy: ``"multi_query"`` (default),
                             ``"hybrid"``, or ``"file_boost"``.

        Returns
        -------
        JSON array of similar patches ranked by score.
        """
        data = request.get_json(silent=True) or {}
        project: str = str(data.get("project", "")).strip().lower()
        patch_id: str = str(data.get("patch_id", "")).strip()
        window: int = int(data.get("time_window", config.DEFAULT_WINDOW_DAYS))
        top_k: int = int(data.get("top_k", config.DEFAULT_TOP_K))
        strategy: str = str(data.get("strategy", config.RETRIEVAL_STRATEGY))

        if not project or project not in engine.loaded_projects:
            return jsonify({
                "error": f"Project '{project}' is not loaded. "
                         f"Available: {engine.loaded_projects}"
            }), 400

        if not patch_id:
            return jsonify({"error": "Missing required field: patch_id"}), 400

        # --- Resolve patch reference ---
        df = engine.datasets[project]
        existing = df[df.patch_id == patch_id]

        if not existing.empty:
            row = existing.iloc[0]
            patch_ref = {
                "patch_id": row.patch_id,
                "title": row.title,
                "description": row.description,
                "created_time": row.created_time,
                "files": engine._safe_parse_list(row.files),
            }
        else:
            logger.info("Patch %s not in dataset — fetching from Gerrit API.", patch_id)
            patch_ref = GerritClient.get_patch_details(project, patch_id)
            if not patch_ref:
                return jsonify({
                    "error": f"Patch '{patch_id}' not found in dataset or Gerrit API."
                }), 404

        # --- Run prediction ---
        try:
            results = engine.predict(
                project, patch_ref, top_k=top_k, window_days=window, strategy=strategy
            )
            return jsonify(results)
        except Exception as exc:
            logger.error("Prediction failed for patch %s: %s", patch_id, exc)
            logger.debug(traceback.format_exc())
            return jsonify({"error": str(exc)}), 500

    return app


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

app = create_app()

if __name__ == "__main__":
    logger.info("🚀 SmartPatch RAG Server — http://%s:%d", config.HOST, config.PORT)
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
