import re
from typing import List, Dict, Any

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
        "entity_density": entity_density
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
        "kanji_ratio": kanji_ratio
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
