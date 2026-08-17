import re
import math
import threading
import numpy as np
from typing import List, Dict, Any

# Dynamic library availability flags
HAS_SPACY = False
HAS_SUDACHI = False
HAS_NLTK = False
HAS_CNTEXT = False
HAS_OSETI = False

# Shared sentiment constants
EN_POS_WORDS = ["good", "great", "joy", "happy", "love", "hope", "bright", "beautiful", "triumph", "warm", "kind", "wonderful", "excellent"]
EN_NEG_WORDS = ["bad", "dark", "grief", "hate", "fear", "pain", "cold", "loss", "death", "despair", "horrible", "terrible", "sad"]

JA_POS_WORDS = ["嬉しい", "楽しい", "美しい", "素晴らしい", "愛する", "成功", "幸せ", "感謝", "満足", "喜ぶ", "希望", "勝利"]
JA_NEG_WORDS = ["悲しい", "苦しい", "怒る", "嫌い", "失敗", "痛い", "最悪", "残念", "孤独", "恐ろしい", "絶望", "憎い"]

ZH_POS_WORDS = ["高兴", "开心", "美丽", "棒", "爱", "成功", "幸福", "感谢", "满意", "喜欢", "胜利", "希望", "精彩", "好", "美", "佳", "喜悦", "美好", "光明"]
ZH_NEG_WORDS = ["悲伤", "痛苦", "生气", "讨厌", "失败", "疼", "差", "可惜", "孤独", "难过", "绝望", "惨", "痛", "杀", "死", "哭", "恨", "恐惧", "阴暗", "灾难"]

# Module-level globals for lazy loading
_nlp_en = None
_nlp_ja = None
_nlp_zh = None
_oseti_analyzer = None
_sia_en = None
_initialized = False
_nlp_lock = threading.Lock()

def _init_nlp_resources():
    """
    Task 1: Tri-language NLP resource initialization for English (spaCy/VADER),
    Chinese (cntext/spaCy), and Japanese (sudachipy/oseti/spaCy).
    Guarded by _nlp_lock for multi-threaded safety.
    """
    global HAS_SPACY, HAS_SUDACHI, HAS_NLTK, HAS_CNTEXT, HAS_OSETI
    global _nlp_en, _nlp_ja, _nlp_zh, _oseti_analyzer, _sia_en, _initialized
    
    if _initialized:
        return

    with _nlp_lock:
        if _initialized:
            return

        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

        # 1. spaCy
        try:
            import spacy
            HAS_SPACY = True
        except ImportError:
            HAS_SPACY = False

        # 2. NLTK (VADER sentiment)
        try:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            HAS_NLTK = True
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
                _sia_en = SentimentIntensityAnalyzer()
            except Exception:
                pass
        except Exception:
            HAS_NLTK = False

        # 3. SudachiPy (Japanese Tokenization Check)
        try:
            from sudachipy import Dictionary
            Dictionary().create()
            HAS_SUDACHI = True
        except Exception:
            HAS_SUDACHI = False

        # 4. Oseti (Japanese Sentiment Analysis)
        try:
            import oseti
            try:
                import ipadic
                _oseti_analyzer = oseti.Analyzer(mecab_args=ipadic.MECAB_ARGS)
            except Exception:
                _oseti_analyzer = oseti.Analyzer()
            HAS_OSETI = True
        except Exception:
            HAS_OSETI = False

        # 5. Chinese Sentiment Library
        try:
            import cntext
            HAS_CNTEXT = True
        except Exception:
            HAS_CNTEXT = False

        if HAS_SPACY:
            import spacy
            try:
                if spacy.util.is_package("en_core_web_sm"):
                    _nlp_en = spacy.load("en_core_web_sm")
            except Exception:
                pass
            try:
                if HAS_SUDACHI and spacy.util.is_package("ja_core_news_sm"):
                    _nlp_ja = spacy.load("ja_core_news_sm")
            except Exception:
                pass
            try:
                if spacy.util.is_package("zh_core_web_sm"):
                    _nlp_zh = spacy.load("zh_core_web_sm")
            except Exception:
                pass

        _initialized = True


def split_paragraphs(text: str) -> List[str]:
    """
    Splits text into paragraphs cleanly handling both ASCII hard-wrapped texts
    (Gutenberg) and single-line paragraph formats (web fiction / HTML scrapers).
    """
    if not text:
        return []
    is_hard_wrapped = bool(re.search(r'[a-zA-Z0-9,]\r?\n[a-z]', text))
    if is_hard_wrapped:
        return [p.strip() for p in re.split(r'(\r?\n\s*){2,}', text) if p.strip()]
    return [p.strip() for p in re.split(r'[\r\n]+', text) if p.strip()]


def detect_language(text: str) -> str:
    """Task 1: Language Routing mechanism based on text script."""
    if not text:
        return "en"
    if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text):
        return "ja"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


def compute_dep_tree_depth(doc) -> float:
    """Calculates average dependency tree depth."""
    def get_depth(node):
        if not list(node.children):
            return 1
        return 1 + max(get_depth(child) for child in node.children)
    depths = [get_depth(s.root) for s in doc.sents]
    return sum(depths) / len(depths) if depths else 0.0


def compute_sentence_sentiment_normalized(sentence: str, lang: str) -> float:
    """
    Task 3: Ensures sentiment functions for all three languages output to the exact same
    -1.0 to +1.0 scale per sentence before calculating intensity.
    """
    if not sentence or not sentence.strip():
        return 0.0
        
    s_clean = sentence.strip()
    _init_nlp_resources()
    
    if lang == "en":
        if _sia_en is not None:
            return float(_sia_en.polarity_scores(s_clean)["compound"])
        pos = len(re.findall(r'\b(' + '|'.join(EN_POS_WORDS) + r')\b', s_clean.lower()))
        neg = len(re.findall(r'\b(' + '|'.join(EN_NEG_WORDS) + r')\b', s_clean.lower()))
        return (pos - neg) / (pos + neg + 1)

    elif lang == "zh":
        if HAS_CNTEXT:
            try:
                import cntext as ct
                cn_dict = {
                    'pos': ['高兴', '开心', '美丽', '棒', '爱', '成功', '幸福', '感谢', '满意', '喜欢', '胜利', '希望', '精彩', '好', '美', '佳', '喜悦', '美好', '光明'],
                    'neg': ['悲伤', '痛苦', '生气', '讨厌', '失败', '疼', '差', '可惜', '孤独', '难过', '绝望', '惨', '痛', '杀', '死', '哭', '恨', '恐惧', '阴暗', '灾难']
                }
                res = ct.sentiment(s_clean, diction=cn_dict, lang='chinese')
                pos_count = res.get('pos_num', 0)
                neg_count = res.get('neg_num', 0)
                if pos_count + neg_count > 0:
                    return float((pos_count - neg_count) / (pos_count + neg_count))
            except Exception:
                pass
        pos = sum(s_clean.count(w) for w in ['高兴', '开心', '美丽', '棒', '爱', '成功', '幸福', '感谢', '满意', '喜欢', '胜利', '希望', '精彩', '好', '美', '佳', '喜悦'])
        neg = sum(s_clean.count(w) for w in ['悲伤', '痛苦', '生气', '讨厌', '失败', '疼', '差', '可惜', '孤独', '难过', '绝望', '惨', '痛', '杀', '死', '哭', '恨', '恐惧'])
        return (pos - neg) / max(1.0, pos + neg)

    elif lang == "ja":
        if HAS_OSETI and _oseti_analyzer is not None:
            try:
                scores = _oseti_analyzer.analyze(s_clean)
                if scores:
                    return float(sum(scores) / len(scores))
            except Exception:
                pass
        ja_pos = ['嬉しい', '楽しい', '美しい', '素晴らしい', '愛する', '成功', '幸せ', '感謝', '満足', '喜ぶ', '希望', '勝利', '笑', '喜', '好', '幸', '美', '勝', '善', '優', '愛', '快', '願', '助け', '安心', '可愛い', '最高', '光', '輝く', '仲間', '笑顔', '微笑', '平和', '素直']
        ja_neg = ['悲しい', '苦しい', '怒る', '嫌い', '失敗', '痛い', '最悪', '残念', '孤独', '恐ろしい', '絶望', '憎い', '怒', '恐', '痛', '死', '殺', '嫌', '害', '泣', '苦', '壊', '悪', '怖', '逃', '悲', '魔物', '敵', '倒す', '襲う', '闇', '絶望', '怪我', '暗い', '問題', '困る', '泣く', '悲傷', '涙', '残酷']
        pos = sum(s_clean.count(w) for w in ja_pos)
        neg = sum(s_clean.count(w) for w in ja_neg)
        if pos + neg > 0:
            return (pos - neg) / (pos + neg)
        return 0.0

    return 0.0


def compute_emotional_tone(sentences: List[str], lang: str) -> float:
    """
    Task 3: Calculates Emotional Tone using absolute mean intensity across sentences.
    Prevents positive and negative sentences from cancelling out to zero.
    """
    if not sentences:
        return 0.0
        
    sentence_scores = [compute_sentence_sentiment_normalized(s, lang) for s in sentences if s.strip()]
    if not sentence_scores:
        return 0.0
        
    abs_scores = [abs(score) for score in sentence_scores]
    mean_abs_intensity = sum(abs_scores) / len(abs_scores)
    return float(mean_abs_intensity)


# --- ARCHETYPE RADAR METRIC CALCULATORS ---

def compute_sliding_window_ttr(tokens: List[str], window_size: int = 500, step: int = 100) -> float:
    """Task 4: Type-Token Ratio calculated on a 500-word sliding window and averaged."""
    if not tokens:
        return 0.0
    if len(tokens) <= window_size:
        return len(set(tokens)) / len(tokens)
    
    ttrs = []
    for i in range(0, len(tokens) - window_size + 1, step):
        window = tokens[i : i + window_size]
        ttrs.append(len(set(window)) / len(window))
    
    return sum(ttrs) / len(ttrs) if ttrs else (len(set(tokens)) / len(tokens))


def compute_visceral_emotion(text: str, doc=None, lang: str = "en") -> float:
    """
    Task 2: Visceral Emotion = proportion of total emotional expressions rendered as body sensations
    vs direct emotional expressions.
    """
    if not text:
        return 0.0
        
    text_lower = text.lower()
    
    if lang == "en":
        body_phrases = [
            "tightening chest", "cold sweat", "white knuckles", "pale face", "racing heart",
            "lump in throat", "goosebumps", "trembling hands", "clenched jaw", "heavy breath",
            "shiver down", "blood ran cold", "stomach churned", "heart pounding", "flushed face",
            "blushing", "trembling", "choked up", "gasping", "pulse raced", "stiffened",
            "chest tightened", "gooseflesh", "shivering", "clenched fist", "palpitations", "dizzy"
        ]
        body_words = ["chest", "sweat", "knuckles", "throat", "heartbeat", "pulse", "stomach", "spine", "breath", "jaw", "shiver", "tears", "blush", "gasp", "choke"]
        
        direct_emotions = [
            "afraid", "sad", "happy", "angry", "joyful", "terrified", "furious", "grieved",
            "delighted", "anxious", "depressed", "excited", "disgusted", "scared", "sorrowful",
            "cheerful", "fearful", "enraged", "elated", "sorrow", "happiness", "grief", "anger",
            "fear", "joy", "sadness", "hate", "disgust", "horror", "panic", "melancholy", "contentment", "despair"
        ]
        
        body_count = sum(len(re.findall(r'\b' + re.escape(p) + r'\b', text_lower)) for p in body_phrases)
        body_count += sum(len(re.findall(r'\b' + re.escape(w) + r'\b', text_lower)) for w in body_words)
        direct_count = sum(len(re.findall(r'\b' + re.escape(w) + r'\b', text_lower)) for w in direct_emotions)
    elif lang == "ja":
        body_words = ["心臓", "冷や汗", "鳥肌", "手が震え", "息が荒", "青ざめ", "胸が締め", "脈", "冷汗", "震え", "汗", "身震い", "喉", "血が引く"]
        direct_emotions = ["悲しい", "嬉しい", "怒る", "怖い", "寂しい", "楽しい", "喜ぶ", "不安", "恐ろしい", "憎い", "幸せ", "絶望"]
        body_count = sum(text.count(w) for w in body_words)
        direct_count = sum(text.count(w) for w in direct_emotions)
    else: # zh
        body_words = ["心跳", "冷汗", "鸡皮疙瘩", "颤抖", "脸色发白", "面色苍白", "握紧拳头", "呼吸急促", "额头冷汗", "咬紧牙关", "哽咽", "脉搏", "心惊肉跳", "发抖"]
        direct_emotions = ["难过", "高兴", "害怕", "愤怒", "伤心", "快乐", "恐惧", "伤感", "讨厌", "绝望", "焦虑", "开心", "激动"]
        body_count = sum(text.count(w) for w in body_words)
        direct_count = sum(text.count(w) for w in direct_emotions)

    total_emotional = body_count + direct_count
    if total_emotional == 0:
        return 0.35
    return float(body_count / total_emotional)


def compute_dialogue_density_quotations(text: str, lang: str = "en") -> float:
    """Task 3: Proportion of total words inside quotations over total words in the entire text."""
    if not text:
        return 0.0
        
    pattern = r'["“「『]([^"”」』]+)["”」』]|' + r"'([^']+)'"
    matches = re.findall(pattern, text)
    quoted_parts = [m[0] or m[1] for m in matches if (m[0] or m[1])]
    quoted_text = " ".join(quoted_parts)
    
    if lang == "en":
        total_words = len(re.findall(r'\b\w+\b', text))
        quoted_words = len(re.findall(r'\b\w+\b', quoted_text))
        return float(quoted_words / max(1, total_words))
    else:
        total_chars = len([c for c in text if not c.isspace()])
        quoted_chars = len([c for c in quoted_text if not c.isspace()])
        return float(quoted_chars / max(1, total_chars))


def compute_temporal_complexity(text: str, doc=None, sentences=None, lang: str = "en") -> float:
    """
    Task 1: Unified Temporal Complexity.
    Track variance in verb tenses across paragraphs and frequency of temporal adverbial phrases.
    """
    if not text:
        return 0.0
        
    text_lower = text.lower()
    
    if lang == "en":
        temporal_markers = [
            "before", "after", "earlier", "later", "years ago", "decades ago", "months ago",
            "formerly", "once upon a time", "meanwhile", "then", "soon", "when", "since",
            "until", "yesterday", "tomorrow", "already", "previously", "beforehand",
            "reminisced", "retrospect", "used to", "back then", "recalled", "flashback",
            "in the past", "former", "latter", "history", "long ago"
        ]
        pattern = r'\b(' + '|'.join(re.escape(w) for w in temporal_markers) + r')\b'
        marker_count = len(re.findall(pattern, text_lower))
    elif lang == "ja":
        temporal_markers = ["昔", "かつて", "当時", "あの頃", "あのとき", "記憶", "思い出", "振り返", "突然", "その時", "それ以来", "以前", "昨日", "明日"]
        marker_count = sum(text.count(m) for m in temporal_markers)
    else: # zh
        temporal_markers = [
            "记得", "曾经", "当时", "那时", "那一刻", "那一天", "那一年", "回忆", "往事",
            "突然", "从前", "以前", "过去", "那个时候", "昨天", "明天", "当年",
            "昔", "尝", "乃", "后", "先", "方", "适", "忽", "既", "遂", "临", "未几",
            "俄而", "异日", "昔者", "初", "末", "比", "寻", "向", "向者", "往日"
        ]
        marker_count = sum(text.count(m) for m in temporal_markers)
        
    sent_count = len(sentences) if sentences else len(re.split(r'[.!?。！？]+', text))
    marker_freq = marker_count / max(1, sent_count)
    
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        paragraphs = [text]
        
    past_ratios = []
    for p in paragraphs:
        p_lower = p.lower()
        if lang == "en":
            past_verbs = len(re.findall(r'\b(\w+ed|was|were|had|did|said|went|came|took|thought|saw|felt|knew|looked|made|told|gave|found|became)\b', p_lower))
            present_verbs = len(re.findall(r'\b(\w+s|\w+ing|is|are|am|have|has|do|does|says|goes|comes|takes|thinks|sees|feels|knows|looks|makes|tells|gives|finds|becomes)\b', p_lower))
            total_v = past_verbs + present_verbs
            past_ratios.append(past_verbs / max(1, total_v))
        else:
            past_count = len(re.findall(r'た|形|了|过|已|既|尝|毕|讫|矣|遂|乃', p))
            total_chars = max(1, len(p))
            past_ratios.append(past_count / total_chars)
            
    if len(past_ratios) > 1:
        mean_p = sum(past_ratios) / len(past_ratios)
        variance = sum((r - mean_p) ** 2 for r in past_ratios) / len(past_ratios)
    else:
        variance = 0.0
        
    temporal_complexity = (variance * 2.5) + marker_freq
    return float(temporal_complexity)


def compute_world_grounding(text: str, doc=None, lang: str = "en") -> float:
    """Task 5: World Grounding = (total adjectives + total nouns) / total verbs"""
    if not text:
        return 1.0
        
    adj_count = 0
    noun_count = 0
    verb_count = 0
    
    if doc is not None:
        try:
            adj_count = len([t for t in doc if t.pos_ in ("ADJ", "JJ", "JJR", "JJS")])
            noun_count = len([t for t in doc if t.pos_ in ("NOUN", "PROPN", "NN", "NNS", "NNP", "NNPS")])
            verb_count = len([t for t in doc if t.pos_ in ("VERB", "AUX", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ")])
        except Exception:
            pass
            
    if verb_count == 0:
        if lang == "en":
            words = re.findall(r'\b\w+\b', text.lower())
            adj_count = len([w for w in words if w.endswith("ful") or w.endswith("ous") or w.endswith("ive") or w.endswith("able") or w.endswith("al") or w.endswith("ic")])
            verb_count = len([w for w in words if w.endswith("ed") or w.endswith("ing") or w in ("is", "was", "were", "are", "be", "been", "have", "had", "has", "do", "did")])
            noun_count = max(1, len(words) - adj_count - verb_count)
        else:
            chars = len([c for c in text if not c.isspace()])
            noun_count = int(chars * 0.45)
            adj_count = int(chars * 0.15)
            verb_count = max(1, int(chars * 0.25))

    return float((adj_count + noun_count) / max(1, verb_count))


def compute_thematic_explicitness(text: str, lang: str = "en") -> float:
    """
    Computes Thematic Explicitness using didactic markers, epiphanic reflection structures,
    and thematic core concept frequencies, normalized per 1,000 units.
    """
    if not text:
        return 0.0

    text_lower = text.lower()
    didactic_count = 0
    epiphanic_count = 0
    theme_count = 0

    if lang == "en":
        didactic_markers = [
            "the true meaning of", "the lesson here was", "with the power of",
            "the moral of", "in the end", "what matters most", "the nature of",
            "real meaning of", "purpose of life", "true value of", "essence of",
            "meaning of", "lesson", "purpose", "moral", "destiny", "fate"
        ]
        didactic_count = sum(len(re.findall(r'\b' + re.escape(p) + r'\b', text_lower)) for p in didactic_markers)

        epiphanic_verbs = [
            "realized", "realize", "learned", "learn", "understood", "understand",
            "discovered", "discover", "comprehended", "recognized", "recognize",
            "decided", "decide", "believed", "believe", "remembered", "remember", "knew"
        ]
        theme_keywords = [
            "love", "life", "death", "truth", "friendship", "power", "destiny", "fate",
            "courage", "sacrifice", "family", "hope", "justice", "humanity", "peace",
            "war", "freedom", "honor", "trust", "soul", "heart", "world", "strength"
        ]
        epiphanic_count = sum(len(re.findall(r'\b' + v + r'\b', text_lower)) for v in epiphanic_verbs)
        theme_count = sum(len(re.findall(r'\b' + k + r'\b', text_lower)) for k in theme_keywords)

        words = re.findall(r'\b\w+\b', text_lower)
        total_units = max(1, len(words))

    elif lang == "zh":
        didactic_markers = [
            "道理是", "真正的意义", "生命的真谛", "教训是", "人生的意义",
            "这告诉我们", "真正的力量", "终究", "生命的意义", "真实含义", "核心道德",
            "天下大势", "分久必合", "合久必分", "天意", "兴亡", "大势", "成败", "治乱",
            "此乃", "天命", "盛衰", "道德", "仁义", "忠义", "英雄", "社稷", "天道"
        ]
        didactic_count = sum(text.count(p) for p in didactic_markers)

        epiphanic_verbs = [
            "悟出", "领悟", "明白", "懂得", "体会到", "意识到", "学到", "看清", "明白", "决定",
            "悟", "感", "叹曰", "知", "识", "明", "见", "决", "省", "觉"
        ]
        theme_keywords = [
            "爱", "生命", "死亡", "真理", "友情", "命运", "希望", "正义", "和平", "勇气", "世界", "力量",
            "义", "忠", "国", "天下", "君", "臣", "生", "死", "成", "败", "志", "道", "大义", "安危"
        ]

        epiphanic_count = sum(text.count(v) for v in epiphanic_verbs)
        theme_count = sum(text.count(k) for k in theme_keywords)

        total_units = max(1, len([c for c in text if not c.isspace()]))

    else:  # ja
        didactic_markers = [
            "本当の意味", "教訓", "人生の意義", "大切なのは", "真の価値",
            "命の意味", "結局のところ", "心の底から", "生きる意味", "友情の力"
        ]
        didactic_count = sum(text.count(p) for p in didactic_markers)

        epiphanic_verbs = ["悟った", "気づいた", "理解した", "学んだ", "見出した", "知った", "決めた", "信じた"]
        theme_keywords = ["愛", "命", "死", "真実", "絆", "運命", "希望", "正義", "平和", "世界", "力"]

        epiphanic_count = sum(text.count(v) for v in epiphanic_verbs)
        theme_count = sum(text.count(k) for k in theme_keywords)

        total_units = max(1, len([c for c in text if not c.isspace()]))

    raw_score = ((didactic_count * 2.0) + (epiphanic_count * 1.0) + (theme_count * 0.15)) / (total_units / 1000.0)
    return float(round(raw_score, 2))


def compute_subplot_diversity(text: str, doc=None, lang: str = "en") -> float:
    """
    Computes Subplot Diversity by:
    1. Dividing input text into 4 equal sequential macro-windows.
    2. Generating MiniLM embedding vectors for each window.
    3. Calculating average adjacent cosine distance between consecutive windows.
    4. Extracting GPE, LOC, and FAC entities via spaCy across windows.
    5. Applying a +0.05 boost per distinct novel location introduced across window transitions.
    """
    if not text or len(text.strip()) < 100:
        return 0.0

    import numpy as np

    # 1. Divide input text into 4 equal sequential macro-windows
    n = len(text)
    w_size = n // 4
    if w_size == 0:
        return 0.0

    w0 = text[0 : w_size]
    w1 = text[w_size : 2 * w_size]
    w2 = text[2 * w_size : 3 * w_size]
    w3 = text[3 * w_size :]
    windows = [w0, w1, w2, w3]

    # 2. Generate embedding vectors for each macro-window using MiniLM
    try:
        from kisholens.ml.embeddings import get_transformer_model
        model = get_transformer_model("all-MiniLM-L6-v2")
        slices = [w.strip() if w.strip() else "prose text" for w in windows]
        v_s = model.encode(slices, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

        norms = np.linalg.norm(v_s, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        v_s = v_s / norms

        # 3. Calculate cosine distance between consecutive windows & average adjacent distance
        d01 = float(1.0 - np.dot(v_s[0], v_s[1]))
        d12 = float(1.0 - np.dot(v_s[1], v_s[2]))
        d23 = float(1.0 - np.dot(v_s[2], v_s[3]))
        avg_adj_dist = (d01 + d12 + d23) / 3.0
    except Exception as e:
        print(f"Error computing MiniLM window embeddings: {e}")
        avg_adj_dist = 0.35

    # 4. Extract GPE, LOC, and FAC entities from each window using spaCy
    window_locations = []
    nlp_model = _nlp_en if lang == "en" else (_nlp_ja if lang == "ja" else _nlp_zh)

    for win in windows:
        locs = set()
        if HAS_SPACY and nlp_model is not None:
            try:
                c_doc = nlp_model(win[:5000])
                for ent in c_doc.ents:
                    if ent.label_ in ("GPE", "LOC", "FAC", "地名", "施設"):
                        locs.add(ent.text.strip().lower())
            except Exception:
                pass

        if not locs:
            matches = re.findall(r'\b(?:[A-Z][a-z]+\s+)?(?:Kingdom|Empire|Capital|City|Town|Village|Forest|Mountain|River|Sea|Castle|Palace|Tower|Domain|Territory)\b', win)
            for m in matches:
                locs.add(m.strip().lower())

        window_locations.append(locs)

    # 5. Track distinct novel locations introduced across window transitions (+0.05 boost per distinct shift)
    seen_locs = set(window_locations[0])
    distinct_location_shifts = 0
    for i in range(1, 4):
        new_locs = window_locations[i] - seen_locs
        if new_locs:
            distinct_location_shifts += len(new_locs)
            seen_locs.update(new_locs)

    location_boost = 0.05 * distinct_location_shifts
    final_score = avg_adj_dist + location_boost
    return float(round(max(0.0, min(1.0, final_score)), 4))


# --- FEATURE EXTRACTION ROUTER ---

def extract_english_features(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    _init_nlp_resources()
    
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    punc_count = len(re.findall(r'[.,\/#!$%\^&\*;:{}=\-_`~()?"\']', text))
    
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0.0
    punc_density = punc_count / len(text) if len(text) > 0 else 0.0

    paragraphs = split_paragraphs(text)
    para_sentence_counts = [len([s.strip() for s in re.split(r'[.!?]+', p) if s.strip()]) for p in paragraphs]
    avg_sentences_per_paragraph = sum(para_sentence_counts) / len(paragraphs) if paragraphs else 0.0

    ttr = compute_sliding_window_ttr(words, window_size=500, step=100)
    dialogue_ratio = compute_dialogue_density_quotations(text, lang="en")
    compound_sentiment = compute_emotional_tone(sentences, lang="en")
    thematic_explicitness = compute_thematic_explicitness(text, lang="en")
    temporal_complexity = compute_temporal_complexity(text, sentences=sentences, lang="en")
    visceral_emotion = compute_visceral_emotion(text, lang="en")
    world_grounding = compute_world_grounding(text, lang="en")
    subplot_diversity = compute_subplot_diversity(text, lang="en")
    
    dep_tree_depth = 0.0
    adj_ratio = 0.0
    verb_ratio = 0.0
    pron_ratio = 0.0
    entity_density = 0.0
    
    if HAS_SPACY and _nlp_en is not None:
        try:
            sample_text = text[:10000]
            doc = _nlp_en(sample_text)
            sample_words = [t for t in doc if not t.is_punct and not t.is_space]
            sample_word_count = len(sample_words)
            if sample_word_count > 0:
                dep_tree_depth = compute_dep_tree_depth(doc)
                adj_count = len([t for t in doc if t.pos_ == "ADJ"])
                verb_count = len([t for t in doc if t.pos_ in ("VERB", "AUX")])
                pron_count = len([t for t in doc if t.pos_ == "PRON"])
                
                adj_ratio = adj_count / sample_word_count
                verb_ratio = verb_count / sample_word_count
                pron_ratio = pron_count / sample_word_count
                
                entity_count = len(doc.ents)
                entity_density = (entity_count / sample_word_count) * 100
                world_grounding = compute_world_grounding(sample_text, doc=doc, lang="en")
                temporal_complexity = compute_temporal_complexity(sample_text, doc=doc, sentences=sentences, lang="en")
                subplot_diversity = compute_subplot_diversity(sample_text, doc=doc, lang="en")
        except Exception as e:
            print(f"Error in English spaCy features extraction: {e}")

    # Fallback when spaCy is not loaded or returned zero values
    if adj_ratio == 0.0 or verb_ratio == 0.0 or dep_tree_depth == 0.0:
        try:
            tagged = nltk.pos_tag(words[:5000]) if HAS_NLTK else []
            if tagged:
                total_t = max(1, len(tagged))
                if adj_ratio == 0.0:
                    adj_ratio = len([w for w, t in tagged if t.startswith("JJ")]) / total_t
                if verb_ratio == 0.0:
                    verb_ratio = len([w for w, t in tagged if t.startswith("VB")]) / total_t
                if pron_ratio == 0.0:
                    pron_ratio = len([w for w, t in tagged if t.startswith("PR")]) / total_t
        except Exception:
            pass

        if adj_ratio == 0.0 and word_count > 0:
            adj_cnt = len([w for w in words if w.endswith(("ful", "ous", "ive", "able", "al", "ic", "y", "ish", "less", "ent", "ant", "ary", "ory"))])
            adj_ratio = adj_cnt / word_count

        if verb_ratio == 0.0 and word_count > 0:
            verb_cnt = len([w for w in words if w.endswith(("ed", "ing", "es", "s")) or w in ("is", "was", "were", "are", "be", "been", "have", "had", "has", "do", "did", "go", "went", "come", "came", "make", "made", "know", "knew", "think", "thought", "see", "saw", "take", "took", "say", "said")])
            verb_ratio = verb_cnt / word_count

        if dep_tree_depth == 0.0:
            clause_markers = ["that", "which", "who", "whom", "whose", "because", "although", "though", "if", "when", "where", "while", "unless", "since", "until", "and", "but", "or"]
            depths = []
            for s in sentences:
                s_words = re.findall(r'\b\w+\b', s.lower())
                c_count = sum(1 for w in s_words if w in clause_markers)
                depth = 2.0 + (c_count * 0.75) + (math.sqrt(len(s_words)) * 0.35)
                depths.append(depth)
            dep_tree_depth = float(sum(depths) / len(depths)) if depths else 2.5

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
        "theme_explication_ratio": thematic_explicitness,
        "linearity_subversion_score": temporal_complexity,
        "sensory_body_density": visceral_emotion,
        "outside_world_engagement": world_grounding,
        "narrative_feature_diversity": subplot_diversity,
        "temporal_shift_score": temporal_complexity
    }


def extract_japanese_features(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    _init_nlp_resources()
    
    chars = [c for c in text if not c.isspace()]
    sentences = [s.strip() for s in re.split(r'[。！？]+', text) if s.strip()]
    punc_count = len(re.findall(r'[、。！？「」『』（）―…ー・]', text))
    
    char_count = len(chars)
    sentence_count = len(sentences)
    avg_sentence_len = char_count / sentence_count if sentence_count > 0 else 0.0
    punc_density = punc_count / len(text) if len(text) > 0 else 0.0
    
    kanji_chars = re.findall(r'[\u4e00-\u9fff]', text)
    kanji_ratio = len(kanji_chars) / char_count if char_count > 0 else 0.0

    paragraphs = split_paragraphs(text)
    para_sentence_counts = [len([s.strip() for s in re.split(r'[。！？]+', p) if s.strip()]) for p in paragraphs]
    avg_sentences_per_paragraph = sum(para_sentence_counts) / len(paragraphs) if paragraphs else 0.0

    ttr = compute_sliding_window_ttr(chars, window_size=500, step=100)
    dialogue_ratio = compute_dialogue_density_quotations(text, lang="ja")
    compound_sentiment = compute_emotional_tone(sentences, lang="ja")
    thematic_explicitness = compute_thematic_explicitness(text, lang="ja")
    temporal_complexity = compute_temporal_complexity(text, sentences=sentences, lang="ja")
    visceral_emotion = compute_visceral_emotion(text, lang="ja")
    world_grounding = compute_world_grounding(text, lang="ja")
    subplot_diversity = compute_subplot_diversity(text, lang="ja")

    dep_tree_depth = 0.0
    particle_ratio = 0.0
    verb_ratio = 0.0
    
    if HAS_SPACY and _nlp_ja is not None:
        try:
            sample_text = text[:10000]
            doc = _nlp_ja(sample_text)
            sample_words = [t for t in doc if not t.is_punct and not t.is_space]
            sample_word_count = len(sample_words)
            if sample_word_count > 0:
                dep_tree_depth = compute_dep_tree_depth(doc)
                particle_count = len([t for t in doc if t.pos_ == "ADP" or "助词" in t.tag_ or t.tag_.startswith("助詞")])
                verb_count = len([t for t in doc if t.pos_ in ("VERB", "AUX") or "动词" in t.tag_ or t.tag_.startswith("動詞")])
                particle_ratio = particle_count / sample_word_count
                verb_ratio = verb_count / sample_word_count
                world_grounding = compute_world_grounding(sample_text, doc=doc, lang="ja")
                subplot_diversity = compute_subplot_diversity(sample_text, doc=doc, lang="ja")
        except Exception as e:
            print(f"Error in Japanese spaCy features extraction: {e}")

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
        "theme_explication_ratio": thematic_explicitness,
        "linearity_subversion_score": temporal_complexity,
        "sensory_body_density": visceral_emotion,
        "outside_world_engagement": world_grounding,
        "narrative_feature_diversity": subplot_diversity,
        "temporal_shift_score": temporal_complexity
    }


def extract_chinese_features(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    _init_nlp_resources()
    
    chars = [c for c in text if not c.isspace()]
    char_count = len(chars)
    sentences = [s.strip() for s in re.split(r'[。！？\n]+', text) if s.strip()]
    sentence_count = len(sentences)
    avg_sentence_len = char_count / sentence_count if sentence_count > 0 else 0.0
    
    paragraphs = split_paragraphs(text)
    para_sentence_counts = [len([s.strip() for s in re.split(r'[。！？]+', p) if s.strip()]) for p in paragraphs]
    avg_sentences_per_paragraph = sum(para_sentence_counts) / len(paragraphs) if paragraphs else 0.0

    punc_count = len(re.findall(r'[，、。！？；：""‘’（）《》【】『』「」——……]', text))
    punc_density = punc_count / len(text) if len(text) > 0 else 0.0

    ttr = compute_sliding_window_ttr(chars, window_size=500, step=100)
    dialogue_ratio = compute_dialogue_density_quotations(text, lang="zh")
    compound_sentiment = compute_emotional_tone(sentences, lang="zh")
    thematic_explicitness = compute_thematic_explicitness(text, lang="zh")
    temporal_complexity = compute_temporal_complexity(text, sentences=sentences, lang="zh")
    visceral_emotion = compute_visceral_emotion(text, lang="zh")
    world_grounding = compute_world_grounding(text, lang="zh")
    subplot_diversity = compute_subplot_diversity(text, lang="zh")
    
    dep_tree_depth = 0.0
    particle_ratio = 0.0
    verb_ratio = 0.0

    if HAS_SPACY and _nlp_zh is not None:
        try:
            sample_text = text[:10000]
            doc = _nlp_zh(sample_text)
            words = [t for t in doc if not t.is_punct and not t.is_space]
            word_count = len(words)
            if word_count > 0:
                dep_tree_depth = compute_dep_tree_depth(doc)
                particle_count = len([t for t in doc if t.pos_ in ("ADP", "PART")])
                verb_count = len([t for t in doc if t.pos_ in ("VERB", "AUX")])
                particle_ratio = particle_count / word_count
                verb_ratio = verb_count / word_count
                world_grounding = compute_world_grounding(sample_text, doc=doc, lang="zh")
                subplot_diversity = compute_subplot_diversity(sample_text, doc=doc, lang="zh")
        except Exception:
            pass

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
        "theme_explication_ratio": thematic_explicitness,
        "linearity_subversion_score": temporal_complexity,
        "sensory_body_density": visceral_emotion,
        "outside_world_engagement": world_grounding,
        "narrative_feature_diversity": subplot_diversity,
        "temporal_shift_score": temporal_complexity
    }


def extract_features(text: str, lang: str = "auto") -> Dict[str, Any]:
    """
    Unified entry point for multi-lingual feature extraction across English, Japanese, and Chinese.
    Autodetects language if lang is 'auto' or unspecified, and dispatches to appropriate extractor.
    """
    _init_nlp_resources()
    target_lang = lang.lower() if lang and lang != "auto" else detect_language(text)
    if target_lang == "ja":
        return extract_japanese_features(text)
    elif target_lang == "zh":
        return extract_chinese_features(text)
    else:
        return extract_english_features(text)



# --- TASK 2: DATA-DRIVEN PERCENTILE NORMALIZATION STRATEGY ---

HISTORICAL_CORPUS_DISTRIBUTIONS = {
    "ttr": [0.20, 0.25, 0.30, 0.35, 0.38, 0.42, 0.45, 0.48, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    "dialogue_ratio": [0.02, 0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.62, 0.68, 0.75, 0.80],
    "punc_density": [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.23, 0.25],
    "dep_tree_depth": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
    "verb_ratio": [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36],
    "avg_sentences_per_paragraph": [1.2, 1.8, 2.5, 3.2, 4.0, 5.0, 6.2, 7.5, 9.0, 11.0],
    "compound_sentiment": [-0.80, -0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95],
    "theme_explication_ratio": [0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.2, 3.0, 4.0, 5.2, 6.5, 8.0, 10.0, 12.5, 15.0],
    "linearity_subversion_score": [0.001, 0.01, 0.03, 0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.24, 0.28, 0.33, 0.38, 0.44, 0.50, 0.58, 0.68, 0.80, 0.95, 1.15, 1.40],
    "sensory_body_density": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.72, 0.78, 0.85, 0.90],
    "outside_world_engagement": [0.5, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.6, 4.2, 5.0, 6.0, 7.2, 8.5, 10.0, 12.0],
    "narrative_feature_diversity": [0.0, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 1.0],
    "word_count": [500, 1500, 3000, 5000, 8000, 12000, 18000, 25000, 35000, 50000, 75000, 100000],
    "sentence_count": [20, 50, 100, 200, 400, 700, 1000, 1500, 2200, 3000, 4500, 6000],
    "avg_sentence_len": [5.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 23.0, 26.0, 30.0, 35.0],
    "adj_ratio": [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.23, 0.26],
    "pron_ratio": [0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.23, 0.26, 0.30, 0.34, 0.38],
    "temporal_shift_score": [0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.26, 0.33, 0.40, 0.50, 0.62, 0.75],
    "entity_density": [0.01, 0.03, 0.05, 0.08, 0.11, 0.14, 0.18, 0.22, 0.27, 0.33, 0.40, 0.50],
    "char_count": [1500, 4000, 8000, 15000, 25000, 40000, 60000, 90000, 130000, 180000],
    "kanji_ratio": [0.10, 0.15, 0.20, 0.25, 0.28, 0.32, 0.35, 0.38, 0.42, 0.46, 0.50],
    "particle_ratio": [0.10, 0.15, 0.18, 0.22, 0.25, 0.28, 0.31, 0.34, 0.37, 0.40]
}

def normalize_feature_percentile(key: str, raw_val: float) -> float:
    """
    Uses smooth continuous linear interpolation over historical corpus distributions
    to score a raw metric value into a precise percentile between 0.01 and 0.99.
    Strips language prefixes (en_, ja_, zh_) to match distribution keys cleanly.
    """
    if raw_val is None or raw_val != raw_val:
        return 0.5
        
    base_key = key.split("_", 1)[1] if (key.startswith("en_") or key.startswith("ja_") or key.startswith("zh_")) else key
    dist = HISTORICAL_CORPUS_DISTRIBUTIONS.get(base_key)
    if not dist:
        return 0.5
        
    pct = float(np.interp(raw_val, dist, np.linspace(0.05, 0.95, len(dist))))
    return round(max(0.01, min(0.99, pct)), 4)


# ARCHETYPE REFERENCE VECTORS
ARCHETYPES = {
    # Classic Literature Territory
    "Philosophy": {
        "ttr": 0.85,
        "dialogue_ratio": 0.15,
        "punc_density": 0.60,
        "dep_tree_depth": 0.85,
        "verb_ratio": 0.35,
        "avg_sentences_per_paragraph": 0.70,
        "compound_sentiment": 0.30,
        "theme_explication_ratio": 0.90,
        "linearity_subversion_score": 0.60,
        "sensory_body_density": 0.20,
        "outside_world_engagement": 0.40,
        "narrative_feature_diversity": 0.80
    },
    "Poetry": {
        "ttr": 0.90,
        "dialogue_ratio": 0.10,
        "punc_density": 0.75,
        "dep_tree_depth": 0.90,
        "verb_ratio": 0.30,
        "avg_sentences_per_paragraph": 0.80,
        "compound_sentiment": 0.50,
        "theme_explication_ratio": 0.70,
        "linearity_subversion_score": 0.70,
        "sensory_body_density": 0.85,
        "outside_world_engagement": 0.50,
        "narrative_feature_diversity": 0.90
    },
    "Tragedy": {
        "ttr": 0.75,
        "dialogue_ratio": 0.40,
        "punc_density": 0.55,
        "dep_tree_depth": 0.75,
        "verb_ratio": 0.45,
        "avg_sentences_per_paragraph": 0.60,
        "compound_sentiment": 0.15,
        "theme_explication_ratio": 0.70,
        "linearity_subversion_score": 0.60,
        "sensory_body_density": 0.65,
        "outside_world_engagement": 0.50,
        "narrative_feature_diversity": 0.70
    },
    "Supernatural": {
        "ttr": 0.80,
        "dialogue_ratio": 0.25,
        "punc_density": 0.50,
        "dep_tree_depth": 0.80,
        "verb_ratio": 0.45,
        "avg_sentences_per_paragraph": 0.65,
        "compound_sentiment": 0.35,
        "theme_explication_ratio": 0.60,
        "linearity_subversion_score": 0.65,
        "sensory_body_density": 0.85,
        "outside_world_engagement": 0.60,
        "narrative_feature_diversity": 0.75
    },

    # Traditional Fiction Territory
    "Historical": {
        "ttr": 0.75,
        "dialogue_ratio": 0.30,
        "punc_density": 0.45,
        "dep_tree_depth": 0.75,
        "verb_ratio": 0.45,
        "avg_sentences_per_paragraph": 0.60,
        "compound_sentiment": 0.45,
        "theme_explication_ratio": 0.65,
        "linearity_subversion_score": 0.40,
        "sensory_body_density": 0.75,
        "outside_world_engagement": 0.90,
        "narrative_feature_diversity": 0.75
    },
    "Fantasy": {
        "ttr": 0.70,
        "dialogue_ratio": 0.40,
        "punc_density": 0.45,
        "dep_tree_depth": 0.65,
        "verb_ratio": 0.50,
        "avg_sentences_per_paragraph": 0.50,
        "compound_sentiment": 0.50,
        "theme_explication_ratio": 0.55,
        "linearity_subversion_score": 0.45,
        "sensory_body_density": 0.70,
        "outside_world_engagement": 0.80,
        "narrative_feature_diversity": 0.75
    },
    "Sci-Fi": {
        "ttr": 0.75,
        "dialogue_ratio": 0.35,
        "punc_density": 0.40,
        "dep_tree_depth": 0.70,
        "verb_ratio": 0.45,
        "avg_sentences_per_paragraph": 0.50,
        "compound_sentiment": 0.45,
        "theme_explication_ratio": 0.60,
        "linearity_subversion_score": 0.50,
        "sensory_body_density": 0.40,
        "outside_world_engagement": 0.85,
        "narrative_feature_diversity": 0.75
    },
    "Mystery": {
        "ttr": 0.65,
        "dialogue_ratio": 0.55,
        "punc_density": 0.45,
        "dep_tree_depth": 0.55,
        "verb_ratio": 0.55,
        "avg_sentences_per_paragraph": 0.40,
        "compound_sentiment": 0.40,
        "theme_explication_ratio": 0.45,
        "linearity_subversion_score": 0.75,
        "sensory_body_density": 0.60,
        "outside_world_engagement": 0.60,
        "narrative_feature_diversity": 0.65
    },
    "Horror": {
        "ttr": 0.75,
        "dialogue_ratio": 0.30,
        "punc_density": 0.50,
        "dep_tree_depth": 0.70,
        "verb_ratio": 0.50,
        "avg_sentences_per_paragraph": 0.55,
        "compound_sentiment": 0.20,
        "theme_explication_ratio": 0.50,
        "linearity_subversion_score": 0.60,
        "sensory_body_density": 0.90,
        "outside_world_engagement": 0.50,
        "narrative_feature_diversity": 0.65
    },
    "Romance": {
        "ttr": 0.55,
        "dialogue_ratio": 0.70,
        "punc_density": 0.45,
        "dep_tree_depth": 0.45,
        "verb_ratio": 0.50,
        "avg_sentences_per_paragraph": 0.35,
        "compound_sentiment": 0.70,
        "theme_explication_ratio": 0.35,
        "linearity_subversion_score": 0.40,
        "sensory_body_density": 0.70,
        "outside_world_engagement": 0.40,
        "narrative_feature_diversity": 0.50
    },
    "Action / Adventure": {
        "ttr": 0.60,
        "dialogue_ratio": 0.45,
        "punc_density": 0.40,
        "dep_tree_depth": 0.45,
        "verb_ratio": 0.65,
        "avg_sentences_per_paragraph": 0.35,
        "compound_sentiment": 0.50,
        "theme_explication_ratio": 0.30,
        "linearity_subversion_score": 0.45,
        "sensory_body_density": 0.75,
        "outside_world_engagement": 0.75,
        "narrative_feature_diversity": 0.60
    },
    "Comedy": {
        "ttr": 0.60,
        "dialogue_ratio": 0.65,
        "punc_density": 0.50,
        "dep_tree_depth": 0.40,
        "verb_ratio": 0.55,
        "avg_sentences_per_paragraph": 0.30,
        "compound_sentiment": 0.80,
        "theme_explication_ratio": 0.30,
        "linearity_subversion_score": 0.50,
        "sensory_body_density": 0.50,
        "outside_world_engagement": 0.45,
        "narrative_feature_diversity": 0.55
    },
    "Drama": {
        "ttr": 0.65,
        "dialogue_ratio": 0.60,
        "punc_density": 0.45,
        "dep_tree_depth": 0.55,
        "verb_ratio": 0.50,
        "avg_sentences_per_paragraph": 0.40,
        "compound_sentiment": 0.45,
        "theme_explication_ratio": 0.60,
        "linearity_subversion_score": 0.50,
        "sensory_body_density": 0.60,
        "outside_world_engagement": 0.50,
        "narrative_feature_diversity": 0.65
    },

    # Web Novel Territory
    "Slice of Life": {
        "ttr": 0.50,
        "dialogue_ratio": 0.60,
        "punc_density": 0.40,
        "dep_tree_depth": 0.40,
        "verb_ratio": 0.50,
        "avg_sentences_per_paragraph": 0.30,
        "compound_sentiment": 0.70,
        "theme_explication_ratio": 0.25,
        "linearity_subversion_score": 0.35,
        "sensory_body_density": 0.50,
        "outside_world_engagement": 0.40,
        "narrative_feature_diversity": 0.45
    },
    "Cultivation": {
        "ttr": 0.50,
        "dialogue_ratio": 0.45,
        "punc_density": 0.40,
        "dep_tree_depth": 0.45,
        "verb_ratio": 0.55,
        "avg_sentences_per_paragraph": 0.30,
        "compound_sentiment": 0.45,
        "theme_explication_ratio": 0.50,
        "linearity_subversion_score": 0.50,
        "sensory_body_density": 0.80,
        "outside_world_engagement": 0.70,
        "narrative_feature_diversity": 0.55
    },
    "Isekai": {
        "ttr": 0.45,
        "dialogue_ratio": 0.65,
        "punc_density": 0.50,
        "dep_tree_depth": 0.35,
        "verb_ratio": 0.55,
        "avg_sentences_per_paragraph": 0.25,
        "compound_sentiment": 0.60,
        "theme_explication_ratio": 0.30,
        "linearity_subversion_score": 0.50,
        "sensory_body_density": 0.65,
        "outside_world_engagement": 0.45,
        "narrative_feature_diversity": 0.45
    },
    "Progression Fantasy": {
        "ttr": 0.48,
        "dialogue_ratio": 0.55,
        "punc_density": 0.45,
        "dep_tree_depth": 0.40,
        "verb_ratio": 0.60,
        "avg_sentences_per_paragraph": 0.25,
        "compound_sentiment": 0.50,
        "theme_explication_ratio": 0.40,
        "linearity_subversion_score": 0.45,
        "sensory_body_density": 0.70,
        "outside_world_engagement": 0.65,
        "narrative_feature_diversity": 0.50
    }
}

ARCHETYPE_TERRITORIES = {
    # Classic Literature Territory
    "Philosophy":         "Classic Literature Territory",
    "Poetry":             "Classic Literature Territory",
    "Tragedy":            "Classic Literature Territory",
    "Supernatural":       "Classic Literature Territory",
    "Historical":         "Classic Literature Territory",
    "Mystery":            "Classic Literature Territory",
    "Horror":             "Classic Literature Territory",
    "Romance":            "Classic Literature Territory",
    "Action / Adventure": "Classic Literature Territory",
    "Comedy":             "Classic Literature Territory",
    "Drama":              "Classic Literature Territory",
    "Fantasy":            "Classic Literature Territory",
    "Sci-Fi":             "Classic Literature Territory",

    # Web Novel Territory
    "Slice of Life":        "Web Novel Territory",
    "Cultivation":          "Web Novel Territory",
    "Isekai":               "Web Novel Territory",
    "Progression Fantasy": "Web Novel Territory",
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
            
    # Task 2: Data-driven percentileofscore normalization
    normalized = {}
    keys = list(next(iter(ARCHETYPES.values())).keys())
    for k in keys:
        raw_val = agnostic.get(k, 0.0)
        normalized[k] = normalize_feature_percentile(k, raw_val)

    # Cosine similarity matching
    similarities = {}
    input_norm = math.sqrt(sum(normalized.get(k, 0.0) ** 2 for k in keys))
    
    best_trope = None
    best_sim = -1.0
    
    for trope, ref_vector in ARCHETYPES.items():
        dot_product = sum(normalized.get(k, 0.0) * ref_vector[k] for k in keys)
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
