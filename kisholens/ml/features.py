import re
from typing import List, Dict, Any

# Dynamic library availability flags
HAS_SPACY = False
HAS_SUDACHI = False
HAS_NLTK = False
HAS_HANLP = False

# Shared sentiment constants
EN_POS_WORDS = ["good", "great", "joy", "happy", "love", "hope", "bright", "beautiful", "triumph", "warm"]
EN_NEG_WORDS = ["bad", "dark", "grief", "hate", "fear", "pain", "cold", "loss", "death", "despair"]

JA_POS_WORDS = ["嬉しい", "楽しい", "美しい", "素晴らしい", "愛する", "成功", "幸せ", "感謝", "満足"]
JA_NEG_WORDS = ["悲しい", "苦しい", "怒る", "嫌い", "失敗", "痛い", "最悪", "残念", "孤独"]

ZH_POS_WORDS = ["高兴", "开心", "美丽", "棒", "爱", "成功", "幸福", "感谢", "满意", "喜欢"]
ZH_NEG_WORDS = ["悲伤", "痛苦", "生气", "讨厌", "失败", "疼", "差", "可惜", "孤独", "难过"]

def compute_narrative_feature_diversity(vals: List[float]) -> float:
    """Computes narrative feature diversity from a list of metrics using their variance."""
    if not vals:
        return 1.0
    mean_val = sum(vals) / len(vals)
    variance = sum((v - mean_val) ** 2 for v in vals) / len(vals)
    return float(1.0 / (1.0 + variance))

# Module-level globals for lazy loading
_nlp_en = None
_nlp_ja = None
_nlp_zh = None
_nlp_hanlp = None
_initialized = False

def _init_nlp_resources():
    """Dynamically load NLP resources when needed, avoiding unnecessary downloads or errors on import."""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    global HAS_SPACY, HAS_SUDACHI, HAS_NLTK, HAS_HANLP
    global _nlp_en, _nlp_ja, _nlp_zh, _nlp_hanlp, _initialized
    
    if _initialized:
        return
        
    # Check spacy
    try:
        import spacy
        HAS_SPACY = True
    except ImportError:
        HAS_SPACY = False

    # Check sudachipy
    try:
        import sudachipy
        HAS_SUDACHI = True
    except ImportError:
        HAS_SUDACHI = False

    # Check nltk
    try:
        import nltk
        HAS_NLTK = True
    except ImportError:
        HAS_NLTK = False

    # Check hanlp
    try:
        import os
        if os.environ.get("DISABLE_HANLP") == "1":
            raise ImportError("HanLP disabled via environment variable")
        import hanlp
        HAS_HANLP = True
    except ImportError:
        HAS_HANLP = False

    # Initialize NLTK if available
    if HAS_NLTK:
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
        except Exception as e:
            print(f"Could not download NLTK resources: {e}")

    # Helper to load/download spacy models
    def load_spacy_model(model_name: str):
        if not HAS_SPACY:
            return None
        import spacy
        try:
            return spacy.load(model_name)
        except Exception as e:
            print(f"Could not load spaCy model {model_name}: {e}")
            return None

    if HAS_SPACY:
        import spacy
        if spacy.util.is_package("en_core_web_sm"):
            _nlp_en = load_spacy_model("en_core_web_sm")
        if HAS_SUDACHI and spacy.util.is_package("ja_core_news_sm"):
            _nlp_ja = load_spacy_model("ja_core_news_sm")
        if spacy.util.is_package("zh_core_web_sm"):
            _nlp_zh = load_spacy_model("zh_core_web_sm")

    if HAS_HANLP:
        try:
            import hanlp
            _nlp_hanlp = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
        except Exception as e:
            print(f"Could not initialize HanLP model: {e}")

    _initialized = True


def compute_type_token_ratio(tokens: List[str]) -> float:
    """Computes Type-Token Ratio (vocabulary diversity)."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_dep_tree_depth(doc) -> float:
    """Calculates the average dependency parse tree depth for the sentences in a spaCy doc."""
    def get_depth(node):
        if not list(node.children):
            return 1
        return 1 + max(get_depth(child) for child in node.children)
    depths = [get_depth(s.root) for s in doc.sents]
    return sum(depths) / len(depths) if depths else 0.0


def extract_english_features(text: str) -> Dict[str, Any]:
    """
    Extracts stylistic features from English text (with spaCy, NLTK, or Regex fallback).
    """
    if not text:
        return {}
        
    _init_nlp_resources()
    
    # Calculate baseline metrics on the FULL text using fast regex/string operations
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    dialogue_lines = [l for l in lines if l.startswith('"') or l.startswith("'") or l.startswith('“') or l.startswith('”')]
    punc_count = len(re.findall(r'[.,\/#!$%\^&\*;:{}=\-_`~()?"\']', text))
    
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0.0
    dialogue_ratio = len(dialogue_lines) / len(lines) if lines else 0.0
    punc_density = punc_count / len(text) if len(text) > 0 else 0.0

    # New features: avg_sentences_per_paragraph and compound_sentiment
    para_sentence_counts = [len([s.strip() for s in re.split(r'[.!?]+', p) if s.strip()]) for p in lines]
    avg_sentences_per_paragraph = sum(para_sentence_counts) / len(lines) if lines else 0.0

    compound_sentiment = 0.0
    if HAS_NLTK:
        try:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            compound_sentiment = sia.polarity_scores(text)["compound"]
        except Exception:
            pass
    
    # Fallback/default metrics
    dep_tree_depth = 0.0
    adj_ratio = 0.0
    verb_ratio = 0.0
    pron_ratio = 0.0
    entity_density = 0.0
    ttr = compute_type_token_ratio(words)
    
    # Extract advanced features on a sample limit of 10,000 characters
    if HAS_SPACY and _nlp_en is not None:
        try:
            sample_text = text[:10000]
            doc = _nlp_en(sample_text)
            sample_words = [t for t in doc if not t.is_punct and not t.is_space]
            sample_word_count = len(sample_words)
            if sample_word_count > 0:
                lemmas = [t.lemma_.lower() for t in sample_words]
                ttr = compute_type_token_ratio(lemmas)
                
                dep_tree_depth = compute_dep_tree_depth(doc)
                
                adj_count = len([t for t in doc if t.pos_ == "ADJ"])
                verb_count = len([t for t in doc if t.pos_ in ("VERB", "AUX")])
                pron_count = len([t for t in doc if t.pos_ == "PRON"])
                
                adj_ratio = adj_count / sample_word_count
                verb_ratio = verb_count / sample_word_count
                pron_ratio = pron_count / sample_word_count
                
                entity_count = len(doc.ents)
                entity_density = (entity_count / sample_word_count) * 100
        except Exception as e:
            print(f"Error in English spaCy features extraction: {e}")
            
    elif HAS_NLTK:
        try:
            import nltk
            sample_text = text[:10000]
            sample_words = nltk.word_tokenize(sample_text.lower())
            sample_word_count = len(sample_words)
            if sample_word_count > 0:
                tagged = nltk.pos_tag(sample_words)
                adj_count = len([w for w, tag in tagged if tag in ('JJ', 'JJR', 'JJS')])
                verb_count = len([w for w, tag in tagged if tag.startswith('V') or tag == 'MD'])
                pron_count = len([w for w, tag in tagged if tag in ('PRP', 'PRP$', 'WP', 'WP$')])
                
                adj_ratio = adj_count / sample_word_count
                verb_ratio = verb_count / sample_word_count
                pron_ratio = pron_count / sample_word_count
        except Exception:
            pass
            
    # Extended Arxiv feature metrics (English)
    theme_words = ["love", "justice", "truth", "death", "fate", "honor", "humanity", "destiny", "wisdom", "morality", "grief", "joy", "peace", "war", "hope", "despair", "time", "memory", "soul", "mind", "life", "world", "history", "nature"]
    theme_pattern = r'\b(' + '|'.join(theme_words) + r')\b'
    theme_count = len(re.findall(theme_pattern, text.lower()))
    theme_explication_ratio = theme_count / max(1, word_count)

    linearity_words = ["remembered", "recalled", "flashback", "years ago", "decades ago", "months ago", "in the past", "formerly", "once", "suddenly", "memories", "yesterday", "tomorrow", "future", "past"]
    linearity_pattern = r'\b(' + '|'.join(linearity_words) + r')\b'
    linearity_count = len(re.findall(linearity_pattern, text.lower()))
    break_punc_count = len(re.findall(r'—|…|\.\.\.|\(|\)', text))
    linearity_subversion_score = (linearity_count + break_punc_count) / max(1, word_count)

    sensory_words = ["see", "hear", "smell", "taste", "feel", "touch", "look", "listen", "sound", "voice", "dark", "light", "red", "blue", "green", "black", "white", "cold", "hot", "warm", "sharp", "soft", "loud", "quiet", "eye", "hand", "face", "breath", "heart", "blood", "head", "body", "finger", "arm", "leg", "throat", "skin"]
    sensory_pattern = r'\b(' + '|'.join(sensory_words) + r')\b'
    sensory_count = len(re.findall(sensory_pattern, text.lower()))
    sensory_body_density = sensory_count / max(1, word_count)

    outside_words = ["sky", "wind", "rain", "sun", "moon", "star", "cloud", "street", "road", "building", "house", "city", "town", "tree", "forest", "mountain", "river", "sea", "ocean", "grass", "flower", "ground", "earth", "weather", "window", "door", "wall", "stone", "wood"]
    outside_pattern = r'\b(' + '|'.join(outside_words) + r')\b'
    outside_count = len(re.findall(outside_pattern, text.lower()))
    outside_world_engagement = outside_count / max(1, word_count)

    # Temporal shift score: density of flashback / time-jump markers per sentence
    temporal_shift_words = [
        "remembered", "recalled", "flashback", "years ago", "decades ago",
        "months ago", "in the past", "formerly", "once upon a time",
        "meanwhile", "at that time", "back then", "long ago",
        "that day", "that year", "that moment", "that night", "used to",
    ]
    temporal_shift_pattern = r'\b(' + '|'.join(re.escape(w) for w in temporal_shift_words) + r')\b'
    temporal_shift_count = len(re.findall(temporal_shift_pattern, text.lower()))
    temporal_shift_score = temporal_shift_count / max(1, sentence_count)

    # Narrative feature diversity (English)
    vals = [ttr or 0.0, dialogue_ratio or 0.0, min(1.0, (punc_density or 0.0) * 10), verb_ratio or 0.0, adj_ratio or 0.0]
    narrative_feature_diversity = compute_narrative_feature_diversity(vals)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": avg_sentence_len,
        "dialogue_ratio": dialogue_ratio,
        "ttr": ttr,
        "punc_density": punc_density,
        "dep_tree_depth": dep_tree_depth,
        "adj_ratio": adj_ratio,
        "verb_ratio": verb_ratio,
        "pron_ratio": pron_ratio,
        "entity_density": entity_density,
        "avg_sentences_per_paragraph": avg_sentences_per_paragraph,
        "compound_sentiment": compound_sentiment,
        "theme_explication_ratio": theme_explication_ratio,
        "linearity_subversion_score": linearity_subversion_score,
        "sensory_body_density": sensory_body_density,
        "outside_world_engagement": outside_world_engagement,
        "narrative_feature_diversity": narrative_feature_diversity,
        "temporal_shift_score": temporal_shift_score
    }


def extract_japanese_features(text: str) -> Dict[str, Any]:
    """
    Extracts stylistic features from Japanese text (with spaCy/Sudachi or Regex/NLTK fallback).
    """
    if not text:
        return {}
        
    _init_nlp_resources()
    
    # Calculate baseline metrics on the FULL text using fast string/regex operations
    chars = [c for c in text if not c.isspace()]
    sentences = [s.strip() for s in re.split(r'[。！？]+', text) if s.strip()]
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    dialogue_lines = [l for l in lines if l.startswith('「') or l.startswith('『')]
    punc_count = len(re.findall(r'[、。！？「」『』（）―…ー・]', text))
    
    char_count = len(chars)
    sentence_count = len(sentences)
    avg_sentence_len = char_count / sentence_count if sentence_count > 0 else 0.0
    dialogue_ratio = len(dialogue_lines) / len(lines) if lines else 0.0
    punc_density = punc_count / len(text) if len(text) > 0 else 0.0
    
    kanji_chars = re.findall(r'[\u4e00-\u9fff]', text)
    kanji_ratio = len(kanji_chars) / char_count if char_count > 0 else 0.0

    # New features: avg_sentences_per_paragraph and compound_sentiment
    para_sentence_counts = [len([s.strip() for s in re.split(r'[。！？]+', p) if s.strip()]) for p in lines]
    avg_sentences_per_paragraph = sum(para_sentence_counts) / len(lines) if lines else 0.0

    pos_count = sum(text.count(w) for w in JA_POS_WORDS)
    neg_count = sum(text.count(w) for w in JA_NEG_WORDS)
    compound_sentiment = (pos_count - neg_count) / (pos_count + neg_count + 1)
    
    # Defaults
    dep_tree_depth = 0.0
    particle_ratio = 0.0
    verb_ratio = 0.0
    ttr = compute_type_token_ratio(chars)
    
    # Advanced features on a sample limit of 10,000 characters
    if HAS_SPACY and _nlp_ja is not None:
        try:
            sample_text = text[:10000]
            doc = _nlp_ja(sample_text)
            sample_words = [t for t in doc if not t.is_punct and not t.is_space]
            sample_word_count = len(sample_words)
            if sample_word_count > 0:
                lemmas = [t.lemma_ for t in sample_words]
                ttr = compute_type_token_ratio(lemmas)
                dep_tree_depth = compute_dep_tree_depth(doc)
                
                particle_count = len([t for t in doc if t.pos_ == "ADP" or "助词" in t.tag_ or t.tag_.startswith("助詞")])
                verb_count = len([t for t in doc if t.pos_ in ("VERB", "AUX") or "动词" in t.tag_ or t.tag_.startswith("動詞")])
                
                particle_ratio = particle_count / sample_word_count
                verb_ratio = verb_count / sample_word_count
        except Exception as e:
            print(f"Error in Japanese spaCy features extraction: {e}")
            
    # Extended Arxiv feature metrics (Japanese)
    theme_words = ["愛", "正義", "真実", "死", "運命", "名誉", "人間", "宿命", "知恵", "道徳", "悲しみ", "喜び", "平和", "戦争", "希望", "絶望", "時間", "記憶", "魂", "心", "命", "世界", "歴史", "自然"]
    theme_count = sum(text.count(w) for w in theme_words)
    theme_explication_ratio = theme_count / max(1, char_count)

    linearity_words = ["思い出した", "回想", "昔", "過去", "以前", "かつて", "突然", "記憶", "昨日", "明日", "未来"]
    linearity_count = sum(text.count(w) for w in linearity_words)
    break_punc_count = len(re.findall(r'―|…|（|）|\(|\)', text))
    linearity_subversion_score = (linearity_count + break_punc_count) / max(1, char_count)

    sensory_words = ["見る", "聞く", "匂う", "味わう", "感じる", "触れる", "見る", "聴く", "音", "声", "暗い", "明るい", "赤い", "青い", "緑", "黒い", "白い", "冷たい", "熱い", "暖かい", "鋭い", "柔らかい", "うるさい", "静か", "目", "手", "顔", "息", "心臓", "血", "頭", "体", "指", "腕", "足", "喉", "肌"]
    sensory_count = sum(text.count(w) for w in sensory_words)
    sensory_body_density = sensory_count / max(1, char_count)

    outside_words = ["空", "風", "雨", "太陽", "月", "星", "雲", "通り", "道", "建物", "家", "都市", "町", "木", "森", "山", "川", "海", "芝生", "花", "地面", "地球", "天気", "窓", "ドア", "壁", "石", "木材"]
    outside_count = sum(text.count(w) for w in outside_words)
    outside_world_engagement = outside_count / max(1, char_count)

    # Temporal shift score (Japanese)
    jp_temporal_markers = ["昔", "かつて", "当時", "あの頃", "あのとき", "記憶", "思い出", "振り返", "突然", "その時", "それ以来", "以前"]
    jp_temporal_count = sum(text.count(m) for m in jp_temporal_markers)
    temporal_shift_score = jp_temporal_count / max(1, sentence_count)

    # Narrative feature diversity (Japanese)
    vals = [ttr or 0.0, dialogue_ratio or 0.0, min(1.0, (punc_density or 0.0) * 5), verb_ratio or 0.0, particle_ratio or 0.0]
    narrative_feature_diversity = compute_narrative_feature_diversity(vals)

    return {
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": avg_sentence_len,
        "dialogue_ratio": dialogue_ratio,
        "ttr": ttr,
        "punc_density": punc_density,
        "dep_tree_depth": dep_tree_depth,
        "particle_ratio": particle_ratio,
        "verb_ratio": verb_ratio,
        "kanji_ratio": kanji_ratio,
        "avg_sentences_per_paragraph": avg_sentences_per_paragraph,
        "compound_sentiment": compound_sentiment,
        "theme_explication_ratio": theme_explication_ratio,
        "linearity_subversion_score": linearity_subversion_score,
        "sensory_body_density": sensory_body_density,
        "outside_world_engagement": outside_world_engagement,
        "narrative_feature_diversity": narrative_feature_diversity,
        "temporal_shift_score": temporal_shift_score
    }


def extract_chinese_features(text: str) -> Dict[str, Any]:
    """
    Extracts stylistic features from Chinese text (utilizing spaCy, NLTK, HanLP, or Regex fallback).
    """
    if not text:
        return {}
        
    _init_nlp_resources()
    
    # Calculate baseline metrics on the FULL text using fast string/regex operations
    chars = [c for c in text if not c.isspace()]
    char_count = len(chars)
    sentences = [s.strip() for s in re.split(r'[。！？\n]+', text) if s.strip()]
    sentence_count = len(sentences)
    avg_sentence_len = char_count / sentence_count if sentence_count > 0 else 0.0
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    dialogue_lines = [l for l in lines if l.startswith('“') or l.startswith('「') or l.startswith('『')]
    dialogue_ratio = len(dialogue_lines) / len(lines) if lines else 0.0
    
    ttr = compute_type_token_ratio(chars)

    # New features: avg_sentences_per_paragraph and compound_sentiment
    para_sentence_counts = [len([s.strip() for s in re.split(r'[。！？]+', p) if s.strip()]) for p in lines]
    avg_sentences_per_paragraph = sum(para_sentence_counts) / len(lines) if lines else 0.0

    pos_count = sum(text.count(w) for w in ZH_POS_WORDS)
    neg_count = sum(text.count(w) for w in ZH_NEG_WORDS)
    compound_sentiment = (pos_count - neg_count) / (pos_count + neg_count + 1)
    punc_count = len(re.findall(r'[，、。！？；：""‘’（）《》【】『』「」——……]', text))
    punc_density = punc_count / len(text) if len(text) > 0 else 0.0
    
    dep_tree_depth = 0.0
    particle_ratio = 0.0
    verb_ratio = 0.0
    
    # Advanced features on a sample limit of 10,000 characters
    if HAS_HANLP and _nlp_hanlp is not None:
        try:
            sample_text = text[:10000]
            doc = _nlp_hanlp(sample_text)
            words = doc.get('tok') or doc.get('tok/fine') or doc.get('tok/coarse')
            if words:
                word_count = len(words)
                ttr = compute_type_token_ratio(words)
                
                pos_tags = doc.get('pos') or doc.get('pos/pku') or doc.get('pos/ctb') or doc.get('pos/863')
                if pos_tags:
                    particle_count = sum(1 for tag in pos_tags if tag in ('u', 'y', '助词') or tag.startswith(('u', 'y')))
                    verb_count = sum(1 for tag in pos_tags if tag in ('v', 'vd', 'vn', '动词') or tag.startswith('v'))
                    particle_ratio = particle_count / word_count if word_count > 0 else 0.0
                    verb_ratio = verb_count / word_count if word_count > 0 else 0.0
                    
                heads = doc.get('dep')
                if heads:
                    def get_node_depth(idx, memo):
                        if idx in memo:
                            return memo[idx]
                        parent = heads[idx][0]
                        if parent == 0:
                            memo[idx] = 1
                            return 1
                        val = 1 + get_node_depth(parent - 1, memo)
                        memo[idx] = val
                        return val
                    memo = {}
                    depths = [get_node_depth(i, memo) for i in range(len(heads))]
                    dep_tree_depth = max(depths) if depths else 0.0
        except Exception:
            pass
            
    elif HAS_SPACY and _nlp_zh is not None:
        try:
            sample_text = text[:10000]
            doc = _nlp_zh(sample_text)
            words = [t for t in doc if not t.is_punct and not t.is_space]
            word_count = len(words)
            if word_count > 0:
                lemmas = [t.lemma_ for t in words]
                ttr = compute_type_token_ratio(lemmas)
                dep_tree_depth = compute_dep_tree_depth(doc)
                
                particle_count = len([t for t in doc if t.pos_ in ("ADP", "PART")])
                verb_count = len([t for t in doc if t.pos_ in ("VERB", "AUX")])
                
                particle_ratio = particle_count / word_count
                verb_ratio = verb_count / word_count
        except Exception:
            pass
            
    # Extended Arxiv feature metrics (Chinese)
    theme_words = ["爱", "正义", "真实", "死", "命运", "名誉", "人类", "宿命", "智慧", "道德", "悲伤", "喜悦", "和平", "战争", "希望", "绝望", "时间", "记忆", "灵魂", "心", "生命", "世界", "历史", "自然"]
    theme_count = sum(text.count(w) for w in theme_words)
    theme_explication_ratio = theme_count / max(1, char_count)

    linearity_words = ["想起", "回忆", "以前", "过去", "曾经", "突然", "记忆", "昨天", "明天", "未来"]
    linearity_count = sum(text.count(w) for w in linearity_words)
    break_punc_count = len(re.findall(r'——|……|（|）|\(|\)', text))
    linearity_subversion_score = (linearity_count + break_punc_count) / max(1, char_count)

    sensory_words = ["看", "听", "闻", "尝", "感觉", "触摸", "瞧", "声音", "嗓音", "黑暗", "明亮", "红色", "蓝色", "绿色", "黑色", "白色", "冷", "热", "温暖", "锋利", "柔软", "吵闹", "安静", "眼睛", "手", "脸", "呼吸", "心脏", "血液", "头", "身体", "手指", "手臂", "腿", "喉咙", "皮肤"]
    sensory_count = sum(text.count(w) for w in sensory_words)
    sensory_body_density = sensory_count / max(1, char_count)

    outside_words = ["天空", "风", "雨", "太阳", "月亮", "星星", "云", "街道", "路", "建筑物", "房子", "城市", "城镇", "树", "森林", "山", "河流", "海", "草", "花", "地面", "地球", "天气", "窗户", "门", "墙", "石头", "木头"]
    outside_count = sum(text.count(w) for w in outside_words)
    outside_world_engagement = outside_count / max(1, char_count)

    # Temporal shift score (Chinese)
    zh_temporal_markers = ["记得", "曾经", "当时", "那时", "那一刻", "那一天", "那一年", "回忆", "往事", "突然", "从前", "以前", "过去", "那个时候"]
    zh_temporal_count = sum(text.count(m) for m in zh_temporal_markers)
    temporal_shift_score = zh_temporal_count / max(1, sentence_count)

    # Narrative feature diversity (Chinese)
    vals = [ttr or 0.0, dialogue_ratio or 0.0, min(1.0, (punc_density or 0.0) * 5), verb_ratio or 0.0, particle_ratio or 0.0]
    narrative_feature_diversity = compute_narrative_feature_diversity(vals)

    return {
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": avg_sentence_len,
        "dialogue_ratio": dialogue_ratio,
        "ttr": ttr,
        "punc_density": punc_density,
        "dep_tree_depth": dep_tree_depth,
        "particle_ratio": particle_ratio,
        "verb_ratio": verb_ratio,
        "avg_sentences_per_paragraph": avg_sentences_per_paragraph,
        "compound_sentiment": compound_sentiment,
        "theme_explication_ratio": theme_explication_ratio,
        "linearity_subversion_score": linearity_subversion_score,
        "sensory_body_density": sensory_body_density,
        "outside_world_engagement": outside_world_engagement,
        "narrative_feature_diversity": narrative_feature_diversity,
        "temporal_shift_score": temporal_shift_score
    }


import math

MIN_MAX_BOUNDS = {
    "ttr": (0.01, 0.60),
    "dialogue_ratio": (0.0, 0.8),
    "punc_density": (0.0, 0.25),
    "dep_tree_depth": (0.0, 8.0),
    "verb_ratio": (0.0, 0.4),
    "avg_sentences_per_paragraph": (1.0, 10.0),
    "compound_sentiment": (-1.0, 1.0),
    "theme_explication_ratio": (0.0, 0.05),
    "linearity_subversion_score": (0.0, 0.05),
    "sensory_body_density": (0.0, 0.1),
    "outside_world_engagement": (0.0, 0.1),
    "narrative_feature_diversity": (0.0, 1.0),
    "temporal_shift_score": (0.0, 0.10)
}

ARCHETYPES = {
    "Victorian Novel": {
        "ttr": 0.8,
        "dialogue_ratio": 0.4,
        "punc_density": 0.5,
        "dep_tree_depth": 0.85,
        "verb_ratio": 0.45,
        "avg_sentences_per_paragraph": 0.6,
        "compound_sentiment": 0.5,
        "theme_explication_ratio": 0.6,
        "linearity_subversion_score": 0.3,
        "sensory_body_density": 0.7,
        "outside_world_engagement": 0.7,
        "narrative_feature_diversity": 0.8,
        "temporal_shift_score": 0.65
    },
    "Philosophical Fiction": {
        "ttr": 0.85,
        "dialogue_ratio": 0.2,
        "punc_density": 0.4,
        "dep_tree_depth": 0.8,
        "verb_ratio": 0.5,
        "avg_sentences_per_paragraph": 0.7,
        "compound_sentiment": 0.4,
        "theme_explication_ratio": 0.95,
        "linearity_subversion_score": 0.4,
        "sensory_body_density": 0.3,
        "outside_world_engagement": 0.4,
        "narrative_feature_diversity": 0.8,
        "temporal_shift_score": 0.55
    },
    "LitRPG": {
        "ttr": 0.3,
        "dialogue_ratio": 0.6,
        "punc_density": 0.7,
        "dep_tree_depth": 0.3,
        "verb_ratio": 0.6,
        "avg_sentences_per_paragraph": 0.2,
        "compound_sentiment": 0.5,
        "theme_explication_ratio": 0.2,
        "linearity_subversion_score": 0.9,
        "sensory_body_density": 0.6,
        "outside_world_engagement": 0.3,
        "narrative_feature_diversity": 0.4,
        "temporal_shift_score": 0.15
    },
    "Isekai": {
        "ttr": 0.35,
        "dialogue_ratio": 0.65,
        "punc_density": 0.5,
        "dep_tree_depth": 0.35,
        "verb_ratio": 0.55,
        "avg_sentences_per_paragraph": 0.25,
        "compound_sentiment": 0.6,
        "theme_explication_ratio": 0.3,
        "linearity_subversion_score": 0.6,
        "sensory_body_density": 0.65,
        "outside_world_engagement": 0.4,
        "narrative_feature_diversity": 0.5,
        "temporal_shift_score": 0.20
    },
    "Xianxia Cultivation": {
        "ttr": 0.4,
        "dialogue_ratio": 0.45,
        "punc_density": 0.4,
        "dep_tree_depth": 0.4,
        "verb_ratio": 0.5,
        "avg_sentences_per_paragraph": 0.3,
        "compound_sentiment": 0.4,
        "theme_explication_ratio": 0.75,
        "linearity_subversion_score": 0.5,
        "sensory_body_density": 0.8,
        "outside_world_engagement": 0.75,
        "narrative_feature_diversity": 0.6,
        "temporal_shift_score": 0.30
    }
}

ARCHETYPE_TERRITORIES = {
    "Victorian Novel": "Classic Literature Territory",
    "Philosophical Fiction": "Classic Literature Territory",
    "LitRPG": "Web Novel Territory",
    "Isekai": "Web Novel Territory",
    "Xianxia Cultivation": "Web Novel Territory"
}

def match_archetype(features: dict) -> dict:
    prefix = ""
    for k in features.keys():
        if k.startswith("en_"):
            prefix = "en_"
            break
        elif k.startswith("ja_"):
            prefix = "ja_"
            break
        elif k.startswith("zh_"):
            prefix = "zh_"
            break
            
    agnostic = {}
    for k, v in features.items():
        if prefix and k.startswith(prefix):
            agnostic[k[len(prefix):]] = v
        elif not k.startswith("en_") and not k.startswith("ja_") and not k.startswith("zh_"):
            agnostic[k] = v
            
    # min-max normalization
    normalized = {}
    for key, bounds in MIN_MAX_BOUNDS.items():
        val = agnostic.get(key, None)
        if val is None:
            val = 0.0
        min_v, max_v = bounds
        norm_val = (val - min_v) / max(1e-9, max_v - min_v)
        norm_val = max(0.0, min(1.0, norm_val))
        normalized[key] = norm_val

    # Cosine similarity
    similarities = {}
    keys = list(MIN_MAX_BOUNDS.keys())
    input_norm = math.sqrt(sum(normalized[k] ** 2 for k in keys))
    
    best_trope = None
    best_sim = -1.0
    
    for trope, ref_vector in ARCHETYPES.items():
        dot_product = sum(normalized[k] * ref_vector[k] for k in keys)
        ref_norm = math.sqrt(sum(ref_vector[k] ** 2 for k in keys))
        
        if input_norm == 0.0 or ref_norm == 0.0:
            sim = 0.0
        else:
            sim = dot_product / (input_norm * ref_norm)
            
        similarities[trope] = sim
        if sim > best_sim:
            best_sim = sim
            best_trope = trope
            
    territory = ARCHETYPE_TERRITORIES.get(best_trope, "Unknown Territory")
    
    return {
        "territory": territory,
        "closest_trope": best_trope,
        "confidence": best_sim,
        "similarities": similarities
    }
