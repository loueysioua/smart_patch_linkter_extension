"""
Gerrit REST API client for SmartPatch.

Fetches patch metadata (subject, files, commit message, creation time)
from a Gerrit instance given a project key and change number.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Gerrit base URLs keyed by project identifier
PROJECT_URLS: dict[str, str] = {
    "onap": "https://gerrit.onap.org/r",
    "qt": "https://codereview.qt-project.org",
    "android": "https://android-review.googlesource.com",
    "openstack": "https://review.opendev.org",
}

# Timeout for Gerrit REST calls (seconds)
_REQUEST_TIMEOUT: int = 10


class GerritClient:
    """Thin wrapper around the Gerrit REST API (read-only)."""

    @staticmethod
    def get_patch_details(project_key: str, patch_id: str) -> dict[str, Any] | None:
        """Fetch and normalise patch metadata from Gerrit.

        Args:
            project_key: One of the keys in :data:`PROJECT_URLS`
                         (e.g. ``"onap"``).
            patch_id: Gerrit change number or Change-Id string.

        Returns:
            Normalised patch dictionary, or ``None`` on error / unknown project.

        Raises:
            ValueError: If *project_key* is not in :data:`PROJECT_URLS`.
        """
        base_url = PROJECT_URLS.get(project_key)
        if not base_url:
            raise ValueError(
                f"Unknown project: '{project_key}'. "
                f"Supported projects: {list(PROJECT_URLS)}"
            )

        url = (
            f"{base_url}/changes/{patch_id}"
            "?o=ALL_REVISIONS&o=ALL_FILES&o=MESSAGES"
        )

        try:
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Gerrit API request failed for patch %s: %s", patch_id, exc)
            return None

        # Gerrit prepends ")]}'\n" to all JSON responses to prevent XSSI
        content = resp.text
        if content.startswith(")]}'"):
            content = content[4:]

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Gerrit response for patch %s: %s", patch_id, exc)
            return None

        return GerritClient._parse_gerrit_response(data, patch_id)

    @staticmethod
    def _parse_gerrit_response(data: dict[str, Any], patch_id: str) -> dict[str, Any]:
        """Normalise a raw Gerrit change JSON into a clean patch dictionary.

        Args:
            data: Parsed JSON dict from the Gerrit REST API.
            patch_id: Original patch identifier (used as fallback).

        Returns:
            Dictionary with keys: ``patch_id``, ``title``, ``description``,
            ``created_time``, ``files``.
        """
        subject: str = data.get("subject", "")
        created_str: str = data.get("created", "")
        created_time = pd.to_datetime(created_str) if created_str else pd.NaT

        revisions: dict[str, Any] = data.get("revisions", {})

        # Find the latest revision by _number
        last_rev: dict[str, Any] = {}
        if revisions:
            max_num = max(r.get("_number", 0) for r in revisions.values())
            for rev in revisions.values():
                if rev.get("_number") == max_num:
                    last_rev = rev
                    break

        commit_msg: str = last_rev.get("commit", {}).get("message", subject)

        # Collect changed files from revision 1 (canonical for patch comparisons)
        _EXCLUDED_PATHS = frozenset({"/COMMIT_MSG", "MERGE_LIST"})
        rev1 = next(
            (r for r in revisions.values() if r.get("_number") == 1), None
        )
        files: list[str] = []
        if rev1:
            files = [f for f in rev1.get("files", {}) if f not in _EXCLUDED_PATHS]

        return {
            "patch_id": str(patch_id),
            "title": subject,
            "description": commit_msg,
            "created_time": created_time,
            "files": files,
        }