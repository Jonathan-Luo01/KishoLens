import os
import re
from typing import Optional, List, Dict, Any

# Dynamic library availability flags
HAS_SPACY = False
HAS_SUDACHI = False
HAS_NLTK = False
HAS_HANLP = False

# Module-level globals for lazy loading
_nlp_en = None
_nlp_ja = None
_nlp_zh = None
_nlp_hanlp = None
_initialized = False

def _init_nlp_resources():
    """Dynamically load NLP resources when needed, avoiding unnecessary downloads or errors on import."""
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
        except Exception as e:
            print(f"Could not download NLTK resources: {e}")

    # Helper to load/download spacy models
    def load_spacy_model(model_name: str):
        if not HAS_SPACY:
            return None
        import spacy
        try:
            return spacy.load(model_name)
        except Exception:
            try:
                print(f"Downloading spaCy model {model_name}...")
                spacy.cli.download(model_name)
                return spacy.load(model_name)
            except Exception as e:
                print(f"Could not load spaCy model {model_name}: {e}")
                return None

    if HAS_SPACY:
        _nlp_en = load_spacy_model("en_core_web_sm")
        if HAS_SUDACHI:
            _nlp_ja = load_spacy_model("ja_core_news_sm")
        _nlp_zh = load_spacy_model("zh_core_web_sm")

    if HAS_HANLP:
        try:
            import hanlp
            _nlp_hanlp = hanlp.load(hanlp.pretrained.mtl.CLOSE_BAG_OF_WORDS_ONLY_ZH)
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
    
    if not HAS_SPACY or _nlp_en is None:
        adj_ratio, verb_ratio, pron_ratio = 0.0, 0.0, 0.0
        if HAS_NLTK:
            import nltk
            try:
                words = nltk.word_tokenize(text.lower())
                sentences = nltk.sent_tokenize(text)
                tagged = nltk.pos_tag(words)
                adj_count = len([w for w, tag in tagged if tag in ('JJ', 'JJR', 'JJS')])
                verb_count = len([w for w, tag in tagged if tag.startswith('V') or tag == 'MD'])
                pron_count = len([w for w, tag in tagged if tag in ('PRP', 'PRP$', 'WP', 'WP$')])
                adj_ratio = adj_count / len(words) if words else 0.0
                verb_ratio = verb_count / len(words) if words else 0.0
                pron_ratio = pron_count / len(words) if words else 0.0
            except Exception:
                words = re.findall(r'\b\w+\b', text.lower())
                sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        else:
            words = re.findall(r'\b\w+\b', text.lower())
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        dialogue_lines = [l for l in lines if l.startswith('"') or l.startswith("'") or l.startswith('“') or l.startswith('”')]
        punc_count = len(re.findall(r'[.,\/#!$%\^&\*;:{}=\-_`~()?"\']', text))
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_len": len(words) / len(sentences) if sentences else 0.0,
            "dialogue_ratio": len(dialogue_lines) / len(lines) if lines else 0.0,
            "ttr": compute_type_token_ratio(words),
            "punc_density": punc_count / len(text) if text else 0.0,
            "dep_tree_depth": 0.0,
            "adj_ratio": adj_ratio,
            "verb_ratio": verb_ratio,
            "pron_ratio": pron_ratio,
            "entity_density": 0.0
        }
        
    doc = _nlp_en(text)
    words = [t for t in doc if not t.is_punct and not t.is_space]
    word_count = len(words)
    sentence_count = len(list(doc.sents))
    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0.0
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    dialogue_lines = [l for l in lines if l.startswith('"') or l.startswith("'") or l.startswith('“') or l.startswith('”')]
    dialogue_ratio = len(dialogue_lines) / len(lines) if lines else 0.0
    
    lemmas = [t.lemma_.lower() for t in words]
    ttr = compute_type_token_ratio(lemmas)
    
    punc_count = len([t for t in doc if t.is_punct])
    punc_density = punc_count / len(text) if len(text) > 0 else 0.0
    avg_dep_depth = compute_dep_tree_depth(doc)
    
    adj_count = len([t for t in doc if t.pos_ == "ADJ"])
    verb_count = len([t for t in doc if t.pos_ in ("VERB", "AUX")])
    pron_count = len([t for t in doc if t.pos_ == "PRON"])
    
    adj_ratio = adj_count / word_count if word_count > 0 else 0.0
    verb_ratio = verb_count / word_count if word_count > 0 else 0.0
    pron_ratio = pron_count / word_count if word_count > 0 else 0.0
    
    entity_count = len(doc.ents)
    entity_density = (entity_count / word_count) * 100 if word_count > 0 else 0.0
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": avg_sentence_len,
        "dialogue_ratio": dialogue_ratio,
        "ttr": ttr,
        "punc_density": punc_density,
        "dep_tree_depth": avg_dep_depth,
        "adj_ratio": adj_ratio,
        "verb_ratio": verb_ratio,
        "pron_ratio": pron_ratio,
        "entity_density": entity_density
    }


def extract_japanese_features(text: str) -> Dict[str, Any]:
    """
    Extracts stylistic features from Japanese text (with spaCy/Sudachi or Regex/NLTK fallback).
    """
    if not text:
        return {}
        
    _init_nlp_resources()
    
    if not HAS_SPACY or _nlp_ja is None:
        chars = [c for c in text if not c.isspace()]
        sentences = [s.strip() for s in re.split(r'[。！？]+', text) if s.strip()]
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        dialogue_lines = [l for l in lines if l.startswith('「') or l.startswith('『')]
        punc_count = len(re.findall(r'[、。！？「」『』（）―…ー・]', text))
        
        kanji_chars = re.findall(r'[\u4e00-\u9fff]', text)
        kanji_ratio = len(kanji_chars) / len(chars) if chars else 0.0
        
        return {
            "char_count": len(chars),
            "sentence_count": len(sentences),
            "avg_sentence_len": len(chars) / len(sentences) if sentences else 0.0,
            "dialogue_ratio": len(dialogue_lines) / len(lines) if lines else 0.0,
            "ttr": compute_type_token_ratio(chars),
            "punc_density": punc_count / len(text) if text else 0.0,
            "dep_tree_depth": 0.0,
            "particle_ratio": 0.0,
            "verb_ratio": 0.0,
            "kanji_ratio": kanji_ratio
        }
        
    doc = _nlp_ja(text)
    words = [t for t in doc if not t.is_punct and not t.is_space]
    word_count = len(words)
    sentence_count = len(list(doc.sents))
    
    chars = [c for c in text if not c.isspace() and c not in ('、', '。', '！', '？', '「', '」', '『', '』')]
    char_count = len(chars)
    avg_sentence_len = char_count / sentence_count if sentence_count > 0 else 0.0
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    dialogue_lines = [l for l in lines if l.startswith('「') or l.startswith('『')]
    dialogue_ratio = len(dialogue_lines) / len(lines) if lines else 0.0
    
    lemmas = [t.lemma_ for t in words]
    ttr = compute_type_token_ratio(lemmas)
    
    punc_count = len([t for t in doc if t.is_punct])
    punc_density = punc_count / len(text) if len(text) > 0 else 0.0
    avg_dep_depth = compute_dep_tree_depth(doc)
    
    particle_count = len([t for t in doc if t.pos_ in ("ADP", "PART")])
    verb_count = len([t for t in doc if t.pos_ in ("VERB", "AUX")])
    
    particle_ratio = particle_count / word_count if word_count > 0 else 0.0
    verb_ratio = verb_count / word_count if word_count > 0 else 0.0
    
    kanji_chars = re.findall(r'[\u4e00-\u9fff]', text)
    kanji_ratio = len(kanji_chars) / char_count if char_count > 0 else 0.0
    
    return {
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": avg_sentence_len,
        "dialogue_ratio": dialogue_ratio,
        "ttr": ttr,
        "punc_density": punc_density,
        "dep_tree_depth": avg_dep_depth,
        "particle_ratio": particle_ratio,
        "verb_ratio": verb_ratio,
        "kanji_ratio": kanji_ratio
    }


def extract_chinese_features(text: str) -> Dict[str, Any]:
    """
    Extracts stylistic features from Chinese text (utilizing spaCy, NLTK, HanLP, or Regex fallback).
    """
    if not text:
        return {}
        
    _init_nlp_resources()
    
    chars = [c for c in text if not c.isspace()]
    char_count = len(chars)
    sentences = [s.strip() for s in re.split(r'[。！？\n]+', text) if s.strip()]
    sentence_count = len(sentences)
    avg_sentence_len = char_count / sentence_count if sentence_count > 0 else 0.0
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    dialogue_lines = [l for l in lines if l.startswith('“') or l.startswith('「') or l.startswith('『')]
    dialogue_ratio = len(dialogue_lines) / len(lines) if lines else 0.0
    
    ttr = compute_type_token_ratio(chars)
    punc_count = len(re.findall(r'[，、。！？；：""‘’（）《》【】『』「」——……]', text))
    punc_density = punc_count / len(text) if len(text) > 0 else 0.0
    
    dep_tree_depth = 0.0
    particle_ratio = 0.0
    verb_ratio = 0.0
    
    if HAS_HANLP and _nlp_hanlp is not None:
        try:
            doc = _nlp_hanlp(text)
            if 'tok' in doc:
                words = doc['tok']
                word_count = len(words)
                ttr = compute_type_token_ratio(words)
                if 'pos' in doc:
                    pos_tags = doc['pos']
                    particle_count = sum(1 for tag in pos_tags if tag in ('u', 'y', '助词'))
                    verb_count = sum(1 for tag in pos_tags if tag in ('v', 'vd', '动词'))
                    particle_ratio = particle_count / word_count if word_count > 0 else 0.0
                    verb_ratio = verb_count / word_count if word_count > 0 else 0.0
                if 'dep' in doc:
                    heads = doc['dep']
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
            return {
                "char_count": char_count,
                "sentence_count": sentence_count,
                "avg_sentence_len": avg_sentence_len,
                "dialogue_ratio": dialogue_ratio,
                "ttr": ttr,
                "punc_density": punc_density,
                "dep_tree_depth": dep_tree_depth,
                "particle_ratio": particle_ratio,
                "verb_ratio": verb_ratio
            }
        except Exception:
            pass
            
    if HAS_SPACY and _nlp_zh is not None:
        try:
            doc = _nlp_zh(text)
            words = [t for t in doc if not t.is_punct and not t.is_space]
            word_count = len(words)
            sentence_count = len(list(doc.sents))
            
            cleaned_chars = [c for c in text if not c.isspace() and c not in ('，', '、', '。', '！', '？', '；', '：')]
            char_count = len(cleaned_chars)
            avg_sentence_len = char_count / sentence_count if sentence_count > 0 else 0.0
            
            lemmas = [t.lemma_ for t in words]
            ttr = compute_type_token_ratio(lemmas)
            
            punc_count = len([t for t in doc if t.is_punct])
            punc_density = punc_count / len(text) if len(text) > 0 else 0.0
            avg_dep_depth = compute_dep_tree_depth(doc)
            
            particle_count = len([t for t in doc if t.pos_ in ("ADP", "PART")])
            verb_count = len([t for t in doc if t.pos_ in ("VERB", "AUX")])
            
            particle_ratio = particle_count / word_count if word_count > 0 else 0.0
            verb_ratio = verb_count / word_count if word_count > 0 else 0.0
            
            return {
                "char_count": char_count,
                "sentence_count": sentence_count,
                "avg_sentence_len": avg_sentence_len,
                "dialogue_ratio": dialogue_ratio,
                "ttr": ttr,
                "punc_density": punc_density,
                "dep_tree_depth": avg_dep_depth,
                "particle_ratio": particle_ratio,
                "verb_ratio": verb_ratio
            }
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
        "verb_ratio": verb_ratio
    }
