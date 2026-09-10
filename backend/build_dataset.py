#!/usr/bin/env python3
"""
transform_gerrit_data.py

Walks a directory tree of the form:

    <root>/<project>/<change_dir>/metadata.json
    <root>/<project>/<change_dir>/patchsets.json
    <root>/<project>/<change_dir>/linkages.json   (ignored here)

and produces one CSV per project at:

    <root>/<project>/all_candidates.csv

with columns matching what SmartPatchEngine.load_project() / GerritClient
already expect elsewhere in your codebase:

    patch_id, change_id, number, title, description, created_time, files, status

and produces one CSV per project at:

    <root>/<project>/all_candidates.csv
    <root>/<project>/ground_truth.csv
    
with columns matching what SmartPatchEngine.load_project() / GerritClient
already expect elsewhere in your codebase:

    patch_id, title, description, created_time, files

Usage:
    python transform_gerrit_data.py /path/to/root
    python transform_gerrit_data.py /path/to/root --project openstack
    python transform_gerrit_data.py /path/to/root --out-name all_candidates.csv
"""

import argparse
import json
import sys
import csv
from pathlib import Path


def load_json(path: Path):
    """Load a JSON file, returning None (and printing a warning) on failure."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"  \u26a0\ufe0f  Malformed JSON in {path}: {e}")
        return None


def pick_patchset_for_files(patchsets):
    """
    patchsets.json can be either:
      - a single patchset object (dict)
      - a list of patchset objects (multiple revisions)

    Files come from patchset_number == 1 if available (mirrors the
    "standard practice for patch comparisons" convention already used in
    GerritClient._parse_gerrit_json), otherwise the highest-numbered
    patchset available (i.e. the latest/most-complete revision).
    """
    if patchsets is None:
        return None

    if isinstance(patchsets, dict):
        return patchsets

    if isinstance(patchsets, list) and patchsets:
        for ps in patchsets:
            if ps.get("patchset_number") == 1:
                return ps
        return max(patchsets, key=lambda p: p.get("patchset_number", 0))

    return None


def extract_ids(metadata: dict, dirname: str):
    """
    Prefer BOTH _number and change_id when available; fall back to
    whichever is present, and finally to the directory name (since
    directories are named with the change id per your description).

    patch_id is the combined identifier used as the primary key
    downstream (unique even if one of the two source fields is missing).
    """
    number = metadata.get("_number")
    change_id = metadata.get("change_id")

    if number is None and change_id is None:
        change_id = dirname

    if number is not None and change_id is not None:
        patch_id = str(number)
    elif number is not None:
        patch_id = str(number)
    else:
        patch_id = str(change_id)

    return patch_id, str(change_id) if change_id else change_id, number


def process_change_dir(change_dir: Path):
    metadata = load_json(change_dir / "metadata.json")
    patchsets = load_json(change_dir / "patchsets.json")
    comments_data = load_json(change_dir / "comments.json")

    if metadata is None:
        print(f"  \u26a0\ufe0f  Skipping {change_dir.name}: missing/invalid metadata.json")
        return None

    patch_id, change_id, number = extract_ids(metadata, change_dir.name)

    title = metadata.get("subject", "") or ""
    description = metadata.get("commit_message", "") or title

    created_time = (
        metadata.get("timestamps", {}).get("created")
        or metadata.get("created")
        or ""
    )

    files = []
    ps = pick_patchset_for_files(patchsets)
    if ps:
        files = ps.get("modified_file_paths", []) or []
        files = [f for f in files if f not in ("/COMMIT_MSG", "/MERGE_LIST")]
    else:
        print(f"  \u26a0\ufe0f  {change_dir.name}: no patchsets.json / no files found")

    # --- Author (from accounts.owner) ---
    owner = metadata.get("accounts", {}).get("owner") or {}
    author_name     = owner.get("name", "") or ""
    author_email    = owner.get("email", "") or ""
    author_username = owner.get("username", "") or ""

    # --- Change log (compact JSON list of entries) ---
    cl = metadata.get("change_log") or {}
    cl_entries = cl.get("entries") or []
    change_log = json.dumps(cl_entries, ensure_ascii=False)

    # --- Comments from comments.json ---
    change_comments  = []
    inline_comments  = []
    if comments_data:
        change_comments = comments_data.get("change_comments") or []
        inline_comments = comments_data.get("inline_comments") or []
    comments = json.dumps(change_comments + inline_comments, ensure_ascii=False)

    return {
        "patch_id":        patch_id,
        "change_id":       change_id,
        "title":           title,
        "description":     description,
        "created_time":    created_time,
        "files":           files,
        "author_name":     author_name,
        "author_email":    author_email,
        "author_username": author_username,
        "change_log":      change_log,
        "comments":        comments,
    }


def _resolve_target(link, known_ids, change_id_to_patch_id):
    """
    Returns (target_id, resolved_via, drop_reason).
    Exactly one of (target_id, drop_reason) is set.
    """
    had_identifier = False

    num = link.get("changeNumber")
    if num:  # nonzero and not None
        had_identifier = True
        candidate = str(num)
        if candidate in known_ids:
            return candidate, "number", None

    cid = link.get("changeId")
    if cid:
        had_identifier = True
        candidate = change_id_to_patch_id.get(cid)
        if candidate is not None:
            return candidate, "change_id", None

    return None, None, ("target_unknown" if had_identifier else "unresolvable")


def process_project(project_dir: Path, out_name: str, gt_out_name: str = "ground_truth.csv"):
    rows = []
    change_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
    print(f"\U0001F4C1 {project_dir.name}: {len(change_dirs)} change directories found")

    known_ids = set()
    change_id_to_patch_id = {}

    for change_dir in sorted(change_dirs):
        row = process_change_dir(change_dir)
        if row is not None:
            rows.append(row)
            known_ids.add(row["patch_id"])
            if row["change_id"]:
                change_id_to_patch_id[row["change_id"]] = row["patch_id"]

    if not rows:
        print(f"  \u26a0\ufe0f  No rows produced for project {project_dir.name}, skipping CSV.")
        return

    out_path = project_dir / out_name
    fieldnames = [
        "patch_id", "title",
        "description", "created_time", "files",
        "author_name", "author_email", "author_username",
        "change_log", "comments",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_out = dict(row)
            # Remove change_id before writing if you don't want it in all_candidates.csv
            row_out.pop("change_id", None)
            row_out["files"] = repr(row["files"])
            # change_log and comments are already JSON strings — write as-is
            writer.writerow(row_out)

    print(f"  \u2705 Wrote {len(rows)} rows -> {out_path}")

    # Pass 2: resolve linkages
    pairs = set()
    stats = {
        "changes_with_linkage_file": 0,
        "changes_with_no_links": 0,
        "raw_link_entries": 0,
        "links_dropped_target_unknown": 0,
        "links_dropped_self": 0,
        "links_dropped_unresolvable": 0,
        "links_resolved_via_number": 0,
        "links_resolved_via_change_id": 0,
        "links_kept": 0,
    }

    for change_dir in sorted(change_dirs):
        link_path = change_dir / "linkages.json"
        if not link_path.exists():
            continue
        
        metadata = load_json(change_dir / "metadata.json")
        if not metadata:
            continue
            
        source_id, _, _ = extract_ids(metadata, change_dir.name)
        if source_id not in known_ids:
            continue

        links = load_json(link_path)
        stats["changes_with_linkage_file"] += 1
        if not links:
            stats["changes_with_no_links"] += 1
            continue

        for link in links:
            stats["raw_link_entries"] += 1

            target_id, resolved_via, drop_reason = _resolve_target(
                link, known_ids, change_id_to_patch_id
            )

            if target_id is None:
                stats[f"links_dropped_{drop_reason}"] += 1
                continue

            if target_id == source_id:
                stats["links_dropped_self"] += 1
                continue

            pairs.add((source_id, target_id))
            stats["links_kept"] += 1
            stats[f"links_resolved_via_{resolved_via}"] += 1

    gt_out_path = project_dir / gt_out_name
    with open(gt_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_patch_id", "target_patch_id"])
        for p in sorted(pairs):
            writer.writerow(p)

    print(f"  \u2705 Wrote {len(pairs)} links -> {gt_out_path}")
    print(f"     -> changes with a linkages.json: {stats['changes_with_linkage_file']}")
    print(f"     -> changes with empty linkages.json (no links): {stats['changes_with_no_links']}")
    print(f"     -> raw link entries seen: {stats['raw_link_entries']}")
    print(f"     -> resolved via changeNumber: {stats['links_resolved_via_number']}")
    print(f"     -> resolved via changeId fallback: {stats['links_resolved_via_change_id']}")
    print(f"     -> dropped (self-reference): {stats['links_dropped_self']}")
    print(f"     -> dropped (unresolvable — no changeNumber and no matching changeId): {stats['links_dropped_unresolvable']}")
    print(f"     -> dropped (target not in this project's dataset): {stats['links_dropped_target_unknown']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="Root directory containing one subdirectory per project")
    parser.add_argument("--project", help="Only process this single project (subdirectory name)")
    parser.add_argument("--out-name", default="all_candidates.csv",
                         help="Output CSV filename written inside each project directory")
    parser.add_argument("--gt-out-name", default="ground_truth.csv",
                         help="Output ground truth CSV filename written inside each project directory")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"Root directory not found: {root}")
        sys.exit(1)

    if args.project:
        project_dirs = [root / args.project]
        if not project_dirs[0].is_dir():
            print(f"Project directory not found: {project_dirs[0]}")
            sys.exit(1)
    else:
        project_dirs = [d for d in root.iterdir() if d.is_dir()]

    for project_dir in sorted(project_dirs):
        process_project(project_dir, args.out_name, args.gt_out_name)


if __name__ == "__main__":
    main()