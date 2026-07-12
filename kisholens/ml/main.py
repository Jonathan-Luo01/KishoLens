import os
import sys
from sqlmodel import Session, select
import pandas as pd

from kisholens.models import Novel, Chapter, get_engine
from kisholens.ml.features import (
    extract_english_features,
    extract_japanese_features,
    extract_chinese_features,
)

# Ensure UTF-8 output encoding for console prints on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        # Fallback if reconfigure is not available
        pass

def main():
    engine = get_engine()
    features_list = []

    with Session(engine) as session:
        novels = session.exec(select(Novel)).all()
        novels_map = {n.id: n for n in novels}
        
        chapters = session.exec(select(Chapter)).all()
        print(f"Processing {len(chapters)} chapters...")
        
        for ch in chapters:
            novel = novels_map.get(ch.novel_id)
            if not novel:
                continue
                
            row = {
                "novel_title": novel.title,
                "author": novel.author,
                "source": novel.source,
                "chapter_num": ch.chapter_number,
                "chapter_title": ch.title
            }
            
            if ch.text_en:
                en_feat = extract_english_features(ch.text_en)
                row.update({f"en_{k}": v for k, v in en_feat.items()})
                
            if ch.text_ja:
                ja_feat = extract_japanese_features(ch.text_ja)
                row.update({f"ja_{k}": v for k, v in ja_feat.items()})

            if hasattr(ch, "text_zh") and ch.text_zh:
                zh_feat = extract_chinese_features(ch.text_zh)
                row.update({f"zh_{k}": v for k, v in zh_feat.items()})
                
            features_list.append(row)

    if not features_list:
        print("No features extracted. The database might be empty.")
        return

    df = pd.DataFrame(features_list)

    print("\nStyle Aggregates by Source Platform:")
    group_cols = ["source"]
    num_cols = [c for c in df.columns if c.startswith("en_") or c.startswith("ja_") or c.startswith("zh_")]
    
    if num_cols:
        agg_df = df.groupby(group_cols)[num_cols].mean()
        print(agg_df.to_string())
    else:
        print("No numeric style columns available to aggregate.")

    print("\n--------------------------------------------------")

    # Stylistic pacing comparisons (English texts)
    en_cols = [c for c in df.columns if c.startswith("en_")]
    if en_cols:
        print("\nEnglish Style Aggregates by Novel:")
        novel_en = df.groupby(["novel_title", "source"])[en_cols].mean().dropna(how='all')
        print(novel_en.to_string())
    else:
        print("\nNo English style columns available.")

    # Stylistic pacing comparisons (Japanese texts)
    ja_cols = [c for c in df.columns if c.startswith("ja_")]
    if ja_cols:
        print("\nJapanese Style Aggregates by Novel:")
        novel_ja = df.groupby(["novel_title", "source"])[ja_cols].mean().dropna(how='all')
        print(novel_ja.to_string())
    else:
        print("\nNo Japanese style columns available.")

    # Stylistic pacing comparisons (Chinese texts)
    zh_cols = [c for c in df.columns if c.startswith("zh_")]
    if zh_cols:
        print("\nChinese Style Aggregates by Novel:")
        novel_zh = df.groupby(["novel_title", "source"])[zh_cols].mean().dropna(how='all')
        print(novel_zh.to_string())
    else:
        print("\nNo Chinese style columns available.")

if __name__ == "__main__":
    main()
