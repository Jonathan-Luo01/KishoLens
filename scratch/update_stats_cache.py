"""Resumable stats_cache.json rebuild across all 2,813 novels in DB with latest taxonomy definitions & classical epic guardrails."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlmodel import Session, select
from kisholens.models import Novel, get_engine
from kisholens.api.main import (
    get_novel_stats, _cached_novel_stats, _save_disk_cache, _load_disk_cache
)

_load_disk_cache()
engine = get_engine()

with Session(engine) as session:
    novel_ids = session.exec(select(Novel.id)).all()

total = len(novel_ids)

# Check how many are marked version 2
already_updated = sum(
    1 for nid in novel_ids 
    if nid in _cached_novel_stats and _cached_novel_stats[nid].get("_v") == 2
)
print(f"Resumable Stats Cache Rebuild: {already_updated}/{total} updated with latest taxonomy version (v2).", flush=True)

for i, nid in enumerate(novel_ids):
    # If already updated to v2, skip
    if nid in _cached_novel_stats and _cached_novel_stats[nid].get("_v") == 2:
        continue

    # Delete old cached entry to force get_novel_stats to recalculate
    if nid in _cached_novel_stats:
        del _cached_novel_stats[nid]

    try:
        res = get_novel_stats(nid)
        res["_v"] = 2  # Mark version 2
    except Exception as e:
        print(f"  [SKIP] Novel {nid}: {e}", flush=True)
    
    if (i + 1) % 50 == 0:
        _save_disk_cache()
        curr_updated = sum(1 for nid_check in novel_ids if nid_check in _cached_novel_stats and _cached_novel_stats[nid_check].get("_v") == 2)
        print(f"  Checkpoint: {i+1}/{total} novels processed ({curr_updated}/{total} updated v2)", flush=True)

_save_disk_cache()
final_updated = sum(1 for nid_check in novel_ids if nid_check in _cached_novel_stats and _cached_novel_stats[nid_check].get("_v") == 2)
print(f"Done! Successfully updated and saved {final_updated}/{total} novel stats in stats_cache.json!", flush=True)
