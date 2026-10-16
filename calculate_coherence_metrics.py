"""Calculate the discourse coherence metrics reported in the study.

Run: python calculate_coherence_metrics.py
See README.md and CODEBOOK.md for data format and definitions."""
from __future__ import annotations
from pathlib import Path
# User settings
# Folder containing group/task folders with Reference.txt and participant .txt files.
# Replace this with your own input folder; relative paths are resolved beside this script.
INPUT_DIR = "demo_data"
# Excel output path. Replace this with your desired output file.
# The results folder is created when the script runs.
OUTPUT_FILE = "results/coherence_metrics_results.xlsx"
import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
import warnings
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
from typing import Callable, Iterable, Optional
import chardet
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import wordnet as wn
nlp = None
HAS_VECTORS = False
scorer = None
tokenizer = None
MAX_LEN = None
CONTENT_POS = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'}

def ensure_nltk_resources(download: bool=True) -> None:
    """Locate the original language resources, including zipped NLTK corpora."""
    for pkg in ['punkt', 'wordnet', 'omw-1.4']:
        base = f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}'
        found = False
        for candidate in [base, base + '.zip']:
            try:
                nltk.data.find(candidate)
                found = True
                break
            except LookupError:
                pass
        if not found:
            if not download or not nltk.download(pkg, quiet=False):
                raise RuntimeError(f'Missing NLTK resource: {pkg}. Check your internet connection.')
        if pkg == 'wordnet':
            wn.ensure_loaded()

def load_spacy_en(model_name: str='auto', allow_blank_fallback: bool=False):
    """
    Load an English spaCy model.

    The manuscript workflow requires en_core_web_md annotations and vectors.
    Blank fallback is disabled for the normal workflow.
    """
    candidates = ['en_core_web_md', 'en_core_web_sm'] if model_name == 'auto' else [model_name]
    for name in candidates:
        try:
            loaded = spacy.load(name)
            print(f'[info] spaCy model loaded: {name}')
            return loaded
        except Exception as exc:
            print(f'[warn] Could not load spaCy model {name}: {exc}')
    if allow_blank_fallback:
        loaded = spacy.blank('en')
        if 'sentencizer' not in loaded.pipe_names:
            loaded.add_pipe('sentencizer')
        print("[warn] Falling back to spacy.blank('en'); metrics depending on POS/NER/lemma may degrade.")
        return loaded
    raise RuntimeError('Required English model not found. Install the dependencies with:\n  pip install -r requirements.txt')

def init_discoscore(device: str='cpu', model_name: str='bert-base-uncased') -> None:
    """Initialize DiscoScore scorer and tokenizer."""
    global scorer, tokenizer, MAX_LEN
    try:
        from disco_score import DiscoScorer
    except ImportError:
        from discoscore import DiscoScorer
    scorer    = DiscoScorer(device='cpu', model_name='bert-base-uncased')
    tokenizer = scorer.tokenizer
    MAX_LEN   = tokenizer.model_max_length
    print(f'[info] DiscoScore initialized: model={model_name}, device={device}, max_len={MAX_LEN}')

def read_text(path: Path | str) -> str:
    """Read a text file using chardet-detected encoding, with a safe fallback."""
    path = Path(path)
    raw = path.read_bytes()
    enc = chardet.detect(raw).get('encoding') or 'utf-8'
    try:
        return raw.decode(enc)
    except Exception:
        return raw.decode(enc, errors='ignore')

def hf_truncate(text: str) -> str:
    """Truncate text with the DiscoScore/HuggingFace tokenizer."""
    if tokenizer is None or MAX_LEN is None:
        raise RuntimeError('DiscoScore tokenizer is not initialized. Call init_discoscore() first.')
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN)
    return tokenizer.decode(enc['input_ids'], skip_special_tokens=True)

def spacy_doc(text: str):
    """Return a spaCy Doc, or a sentence list only if sentence starts are unavailable."""
    if nlp is None:
        raise RuntimeError('spaCy model is not initialized. Call load_spacy_en() first.')
    doc = nlp(text)
    if not doc.has_annotation('SENT_START'):
        return [s.strip() for s in text.replace('\r', ' ').split('\n') if s.strip()]
    return doc

def safe_metric(metric_name: str, func: Callable[[], float], file_context: str='') -> float:
    """Run one scalar metric safely while logging errors instead of silently swallowing them."""
    try:
        return func()
    except Exception as exc:
        prefix = f' in {file_context}' if file_context else ''
        warnings.warn(f'{metric_name} failed{prefix}: {exc}')
        return float('nan')

def safe_dict_metric(metric_name: str, func: Callable[[], dict], file_context: str='') -> dict:
    """Run one dictionary-returning metric safely and return an empty dict if it fails."""
    try:
        return func()
    except Exception as exc:
        prefix = f' in {file_context}' if file_context else ''
        warnings.warn(f'{metric_name} failed{prefix}: {exc}')
        return {}

@lru_cache(maxsize=None)
def wn_lemmas(word: str) -> set[str]:
    syns = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemma_names():
            syns.add(lemma.replace('_', ' ').lower())
    return syns

@lru_cache(maxsize=None)
def wn_hyper_hypo(word: str) -> set[str]:
    rel = set()
    for syn in wn.synsets(word):
        for h in syn.hypernyms() + syn.hyponyms():
            for lemma in h.lemma_names():
                rel.add(lemma.replace('_', ' ').lower())
    return rel

def bigram_recurrence_count(lemmas: list[str]) -> int:
    bigrams = list(zip(lemmas, lemmas[1:]))
    freq = Counter(bigrams)
    repeated = {bg: c for (bg, c) in freq.items() if c >= 2}
    token_count = sum((c * 2 for c in repeated.values()))
    return token_count

def repetition_cohesion_metrics(text: str) -> dict[str, float]:
    doc = spacy_doc(text)
    if isinstance(doc, list):
        tokens = [w for w in ' '.join(doc).split() if w.isalpha()]
        content_lemmas = [w.lower() for w in tokens]
        denom_content = len(content_lemmas) if content_lemmas else 1
        freq = Counter(content_lemmas)
        all_types = len(freq)
        repeated_types = {lemma for (lemma, count) in freq.items() if count >= 2}
        rc_types = len(repeated_types)
        rc_types_norm = rc_types / all_types if all_types else 0.0
        lc_bigram_recurrence = bigram_recurrence_count(content_lemmas)
        lc_synonym = 0
        lc_hyper_hyponym = 0
        denom = denom_content
        return {'RC_types_norm': rc_types_norm, 'LC_synonym_norm': lc_synonym / denom, 'LC_hyper_hyponym_norm': lc_hyper_hyponym / denom, 'LC_bigram_recurrence_norm': lc_bigram_recurrence / denom}
    toks = [t for t in doc if not (t.is_space or t.is_punct)]
    content = [t for t in toks if t.pos_ in CONTENT_POS]
    denom_content = len(content) if content else 1
    content_lemmas = [t.lemma_.lower() for t in content]
    freq = Counter(content_lemmas)
    all_types = len(freq)
    repeated_types = {lemma for (lemma, count) in freq.items() if count >= 2}
    rc_types = len(repeated_types)
    rc_types_norm = rc_types / all_types if all_types else 0.0
    lemma_set = set(freq.keys())
    syn_flag = {lemma: False for lemma in lemma_set}
    for (lemma1, lemma2) in combinations(lemma_set, 2):
        if syn_flag[lemma1] and syn_flag[lemma2]:
            continue
        syns1 = wn_lemmas(lemma1)
        if lemma2 in syns1:
            syn_flag[lemma1] = True
            syn_flag[lemma2] = True
            continue
        syns2 = wn_lemmas(lemma2)
        if lemma1 in syns2:
            syn_flag[lemma1] = True
            syn_flag[lemma2] = True
    lc_synonym = sum((freq[lemma] for (lemma, flag) in syn_flag.items() if flag))
    ss_flag = {lemma: False for lemma in lemma_set}
    for (lemma1, lemma2) in combinations(lemma_set, 2):
        if ss_flag[lemma1] and ss_flag[lemma2]:
            continue
        rel1 = wn_hyper_hypo(lemma1)
        if lemma2 in rel1:
            ss_flag[lemma1] = True
            ss_flag[lemma2] = True
            continue
        rel2 = wn_hyper_hypo(lemma2)
        if lemma1 in rel2:
            ss_flag[lemma1] = True
            ss_flag[lemma2] = True
    lc_hyper_hyponym = sum((freq[lemma] for (lemma, flag) in ss_flag.items() if flag))
    lc_bigram_recurrence = bigram_recurrence_count(content_lemmas)
    denom = denom_content
    return {'RC_types_norm': rc_types_norm, 'LC_synonym_norm': lc_synonym / denom, 'LC_hyper_hyponym_norm': lc_hyper_hyponym / denom, 'LC_bigram_recurrence_norm': lc_bigram_recurrence / denom}

def entity_continuity_metrics(text: str, use_distance_decay: bool=True, max_gap: Optional[int]=None) -> dict[str, float]:
    """Calculate EG_PW and EG_PAcc using shared entities and accumulated role weights."""
    doc = spacy_doc(text)
    sents = list(doc.sents) if not isinstance(doc, list) else doc
    n = len(sents)
    if n <= 1:
        return {'EG_PW': 0.0, 'EG_PAcc': 0.0}
    subj_deps = {'nsubj', 'nsubjpass', 'csubj', 'csubjpass'}
    obj_deps = {'obj', 'dobj', 'iobj', 'pobj', 'attr', 'dative', 'oprd'}
    skip_ner = {'CARDINAL', 'ORDINAL', 'DATE', 'TIME', 'MONEY', 'QUANTITY', 'PERCENT', 'LANGUAGE'}

    def sent_entity_role_weights(sent):
        weights = defaultdict(int)
        used_spans = []
        try:
            chunks = list(sent.noun_chunks)
        except Exception:
            chunks = []
        for chunk in chunks:
            toks = [t for t in chunk if getattr(t, 'is_alpha', False)]
            if not toks:
                continue
            key = chunk.root.lemma_.lower()
            role = 'S' if any((t.dep_ in subj_deps for t in chunk)) else 'O' if any((t.dep_ in obj_deps for t in chunk)) else 'X'
            weight = 3 if role == 'S' else 2 if role == 'O' else 1
            weights[key] += weight
            used_spans.append((chunk.start, chunk.end))
        if not isinstance(sent, list) and nlp is not None and ('ner' in nlp.pipe_names):
            for ent in sent.ents:
                if ent.label_ in skip_ner:
                    continue
                if any((ent.start >= start and ent.end <= end for (start, end) in used_spans)):
                    continue
                lemmas = [t.lemma_.lower() for t in ent if getattr(t, 'is_alpha', False)]
                if not lemmas:
                    continue
                key = ' '.join(lemmas)
                role = 'S' if any((getattr(t, 'dep_', None) in subj_deps for t in ent)) else 'O' if any((getattr(t, 'dep_', None) in obj_deps for t in ent)) else 'X'
                weight = 3 if role == 'S' else 2 if role == 'O' else 1
                weights[key] += weight
        if not chunks:
            for t in sent if not isinstance(sent, str) else []:
                if getattr(t, 'is_alpha', False) and t.pos_ in {'NOUN', 'PROPN'}:
                    key = t.lemma_.lower()
                    role = 'S' if t.dep_ in subj_deps else 'O' if t.dep_ in obj_deps else 'X'
                    weight = 3 if role == 'S' else 2 if role == 'O' else 1
                    weights[key] += weight
        return weights
    if isinstance(doc, list):
        sent_weights = []
        for sent in sents:
            toks = [w.lower() for w in sent.split() if w.isalpha()]
            counts = Counter(toks)
            sent_weights.append({key: int(value > 0) for (key, value) in counts.items()})
    else:
        sent_weights = [sent_entity_role_weights(sent) for sent in sents]
    out_pw = np.zeros(n, dtype=float)
    out_pacc = np.zeros(n, dtype=float)
    for i in range(n):
        entities_i = sent_weights[i].keys()
        for j in range(i + 1, n):
            if max_gap is not None and j - i > max_gap:
                break
            entities_j = sent_weights[j].keys()
            inter = set(entities_i).intersection(entities_j)
            if not inter:
                continue
            gap = j - i
            decay = 1.0 / gap if use_distance_decay and gap > 0 else 1.0
            w_pw = float(len(inter))
            w_pacc = 0.0
            for entity in inter:
                w_pacc += float(sent_weights[i][entity]) * float(sent_weights[j][entity])
            w_pw *= decay
            w_pacc *= decay
            out_pw[i] += w_pw
            out_pacc[i] += w_pacc
    score_pw = float(out_pw.sum() / n)
    score_pacc = float(out_pacc.sum() / n)
    return {'EG_PW': score_pw, 'EG_PAcc': score_pacc}

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def sent_content_tokens_with_vecs(sent) -> list:
    toks = []
    for t in sent:
        if getattr(t, 'is_alpha', False) and t.pos_ in CONTENT_POS:
            vec = getattr(t, 'vector', None)
            if vec is not None and np.linalg.norm(vec) > 0:
                toks.append(t)
    return toks

def pair_similarity_embeddings(si_tokens: list, sj_tokens: list) -> float:
    """Bidirectional average of maximum token-token cosine similarities."""
    if not si_tokens or not sj_tokens:
        return 0.0
    a_matrix = np.array([t.vector for t in si_tokens], dtype=float)
    b_matrix = np.array([t.vector for t in sj_tokens], dtype=float)
    sims_ij = []
    for a_vec in a_matrix:
        cos_vals = np.dot(b_matrix, a_vec) / (np.linalg.norm(b_matrix, axis=1) * np.linalg.norm(a_vec) + 1e-12)
        sims_ij.append(float(np.max(cos_vals)))
    sims_ji = []
    for b_vec in b_matrix:
        cos_vals = np.dot(a_matrix, b_vec) / (np.linalg.norm(a_matrix, axis=1) * np.linalg.norm(b_vec) + 1e-12)
        sims_ji.append(float(np.max(cos_vals)))
    return 0.5 * (float(np.mean(sims_ij)) + float(np.mean(sims_ji)))

def lexical_similarity_metrics(text: str, window: int=3, sim_threshold: Optional[float]=None) -> dict[str, float]:
    """Calculate LG_avg_degree and LG_avg_weight from sentence lexical similarities."""
    doc = spacy_doc(text)
    sents = doc if isinstance(doc, list) else list(doc.sents)
    n = len(sents)
    if n == 0:
        return {'LG_avg_degree': 0.0, 'LG_avg_weight': 0.0}
    graph = nx.Graph()
    for i in range(n):
        graph.add_node(f'S{i + 1}')
    weights = []
    if not isinstance(doc, list) and HAS_VECTORS:
        sent_tokens = [sent_content_tokens_with_vecs(sent) for sent in sents]
        for i in range(n):
            for j in range(i + 1, n):
                if j - i > window:
                    break
                weight = pair_similarity_embeddings(sent_tokens[i], sent_tokens[j])
                if sim_threshold is None or weight >= sim_threshold:
                    if weight > 0:
                        graph.add_edge(f'S{i + 1}', f'S{j + 1}', weight=weight)
                        weights.append(weight)
    else:
        sent_lemmas = []
        for sent in sents:
            if isinstance(doc, list):
                toks = [w.lower() for w in sent.split() if w.isalpha()]
            else:
                toks = [t.lemma_.lower() for t in sent if getattr(t, 'is_alpha', False) and t.pos_ in CONTENT_POS]
            sent_lemmas.append(toks)
        vocab = sorted(set(sum(sent_lemmas, [])))
        idx = {word: i for (i, word) in enumerate(vocab)}
        tf = np.zeros((n, len(vocab)), dtype=float)
        for (i, toks) in enumerate(sent_lemmas):
            counts = Counter(toks)
            for (word, count) in counts.items():
                tf[i, idx[word]] = float(count)
        df = np.count_nonzero(tf > 0, axis=0)
        idf = np.log((1 + n) / (1 + df)) + 1.0
        tfidf = tf * idf
        for i in range(n):
            for j in range(i + 1, n):
                if j - i > window:
                    break
                weight = cosine(tfidf[i], tfidf[j])
                if sim_threshold is None or weight >= sim_threshold:
                    if weight > 0:
                        graph.add_edge(f'S{i + 1}', f'S{j + 1}', weight=weight)
                        weights.append(weight)
    degs = dict(graph.degree())
    avg_degree = sum(degs.values()) / len(degs) if degs else 0.0
    avg_weight = float(np.mean(weights)) if weights else 0.0
    return {'LG_avg_degree': avg_degree, 'LG_avg_weight': avg_weight}

def lemma_chain_metrics(text: str) -> dict[str, float]:
    doc = spacy_doc(text)
    if isinstance(doc, list):
        toks = [w for w in ' '.join(doc).split() if w.isalpha()]
        lemmas = [w.lower() for w in toks]
        freq = Counter(lemmas)
        chains = {lemma: count for (lemma, count) in freq.items() if count >= 2}
        lengths = list(chains.values())
        return {'LCh_avg_len': sum(lengths) / len(lengths) if lengths else 0.0, 'LCh_avg_span': 0.0}
    sents = list(doc.sents)
    lemma_sent_map = defaultdict(list)
    for (sent_idx, sent) in enumerate(sents):
        for t in sent:
            if getattr(t, 'is_alpha', False):
                lemma_sent_map[t.lemma_.lower()].append(sent_idx)
    chains = {lemma: idxs for (lemma, idxs) in lemma_sent_map.items() if len(idxs) >= 2}
    lengths = [len(values) for values in chains.values()]
    spans = [len(set(values)) for values in chains.values()]
    return {'LCh_avg_len': sum(lengths) / len(lengths) if lengths else 0.0, 'LCh_avg_span': sum(spans) / len(spans) if spans else 0.0}
MANUSCRIPT_METRICS = ['RC_types_norm', 'LC_synonym_norm', 'LC_hyper_hyponym_norm', 'LC_bigram_recurrence_norm', 'LG_avg_degree', 'LG_avg_weight', 'LCh_avg_len', 'LCh_avg_span', 'EG_PW', 'EG_PAcc', 'DS_FOCUS', 'DS_SENT']

def resolve_user_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(__file__).resolve().parent / path).resolve()

def select_manuscript_metrics(frame: pd.DataFrame, group_name: Optional[str]=None, task_name: Optional[str]=None) -> pd.DataFrame:
    """Select the manuscript metrics and participant identifiers without changing numeric values."""
    result = pd.DataFrame(index=frame.index)
    result['Group'] = group_name if group_name is not None else frame['group_name']
    result['Task'] = task_name if task_name is not None else frame['task_name']
    result['Participant'] = frame['file_name'].map(lambda name: Path(str(name)).stem)
    for metric in MANUSCRIPT_METRICS:
        column = metric
        if column not in frame:
            raise RuntimeError(f'Missing calculated metric: {metric}')
        result[metric] = frame[column]
        if not np.isfinite(pd.to_numeric(result[metric], errors='coerce')).all():
            raise RuntimeError(f'{metric} has failed or unavailable calculations; inspect the preceding warnings.')
    return result[['Group', 'Task', 'Participant'] + MANUSCRIPT_METRICS]

def write_metric_table(frame: pd.DataFrame, path: Path) -> None:
    """Write the manuscript metric table with readable column labels."""
    from openpyxl.styles import Font, PatternFill
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        frame.to_excel(writer, sheet_name='metrics', index=False)
        sheet = writer.book['metrics']
        sheet.freeze_panes = 'D2'
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(name='Arial', size=10, bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='24445C')
        sheet.row_dimensions[1].height = 24
        for column in sheet.columns:
            width = max((len(str(cell.value if cell.value is not None else '')) for cell in list(column)[:400])) + 2
            sheet.column_dimensions[column[0].column_letter].width = min(max(width, 16), 40)

def ensure_udpipe_model() -> None:
    """Use the same English EWT model as DiscoScore; reject HTML downloads.

    This configures a model-data directory only. No dependency functions or
    metric functions are replaced. Existing valid installations are reused.
    """
    from spacy_udpipe import utils as udpipe_utils
    from ufal.udpipe import Model
    filename = 'english-ewt-ud-2.5-191206.udpipe'
    expected_sha256 = '784bd0fa85e3d831fd02a55290d0acfd05c953159dc38cc33d52e1b28add9957'

    def valid(path: Path) -> bool:
        return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256 and (Model.load(str(path)) is not None)
    installed = Path(udpipe_utils.MODELS_DIR) / filename
    if valid(installed):
        return
    cache = Path.home() / '.cache' / 'coherence_metrics' / 'udpipe'
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / filename
    if not valid(target):
        url = 'https://raw.githubusercontent.com/jwijffels/udpipe.models.ud.2.5/master/inst/udpipe-ud-2.5-191206/' + filename
        print('[download] English EWT UDPipe model (approximately 16 MB)')
        with tempfile.NamedTemporaryFile(dir=cache, suffix='.download', delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary_path.open('wb') as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
            if not valid(temporary_path):
                raise RuntimeError('UDPipe download failed its model/hash validation; check the network connection.')
            temporary_path.replace(target)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    udpipe_utils.MODELS_DIR = str(cache)

def is_reference_path(path: Path, base: Path) -> bool:
    """Return True if path is inside a reference directory under base."""
    try:
        rel_parts = path.relative_to(base).parts
    except ValueError:
        return False
    return any((part.lower().startswith('reference') for part in rel_parts))

def find_reference_file(subfolder: Path) -> Optional[Path]:
    """Find the task's single direct Reference.txt, case-insensitively."""
    matches = [p for p in subfolder.iterdir() if p.is_file() and p.name.lower() == 'reference.txt']
    if len(matches) > 1:
        raise ValueError(f'Multiple reference files in {subfolder}')
    return matches[0] if matches else None

def process_one_group(input_dir: Path, output_dir: Path, group_name: str, overwrite: bool=False, internal_text: str='truncated', sort_items: bool=False, exclude_teacher_track: bool=False) -> list[pd.DataFrame]:
    """Recompute task records and export only manuscript variables."""
    if internal_text != 'truncated':
        raise ValueError('The manuscript protocol requires truncated text.')
    global scorer
    all_sub_dfs: list[pd.DataFrame] = []
    if not input_dir.is_dir():
        print(f'[skip] Input directory not found for {group_name}: {input_dir}')
        return all_sub_dfs
    output_dir.mkdir(parents=True, exist_ok=True)
    subfolders = [name for name in os.listdir(input_dir) if (input_dir / name).is_dir()]
    if sort_items:
        subfolders = sorted(subfolders)
    for sub in subfolders:
        subpath = input_dir / sub
        excel_path = output_dir / f'{sub}.xlsx'
        ref_file = find_reference_file(subpath)
        if not ref_file:
            raise FileNotFoundError(f'Expected one Reference.txt in {subpath}')
        ref_raw = read_text(ref_file)
        ref_disco = hf_truncate(ref_raw)
        rows = []
        for (root, dirs, files) in os.walk(subpath):
            root_path = Path(root)
            dirs[:] = [dirname for dirname in dirs if not dirname.lower().startswith('reference')]
            if is_reference_path(root_path, subpath):
                continue
            if sort_items:
                files = sorted(files)
            for filename in files:
                if not filename.lower().endswith('.txt'):
                    continue
                if exclude_teacher_track and 'teacher' in filename.lower() and ('track' in filename.lower()):
                    continue
                file_path = root_path / filename
                try:
                    if file_path.resolve() == ref_file.resolve():
                        continue
                except OSError:
                    pass
                hyp_raw = read_text(file_path)
                hyp_disco = hf_truncate(hyp_raw)
                hyp_internal = hyp_disco
                rec = {'file_name': filename}
                context = str(file_path)
                rec['DS_FOCUS'] = safe_metric('DS_FOCUS', lambda h=hyp_disco, r=ref_disco: scorer.DS_Focus_NN(h, [r]), context)
                rec['DS_SENT'] = safe_metric('DS_SENT', lambda h=hyp_disco, r=ref_disco: scorer.DS_SENT_NN(h, [r]), context)
                rc_lc = safe_dict_metric('RC_types_norm / LC_synonym_norm / LC_hyper_hyponym_norm / LC_bigram_recurrence_norm', lambda h=hyp_internal: repetition_cohesion_metrics(h), context)
                rec.update(rc_lc)
                eg_gs = safe_dict_metric('EG_PW / EG_PAcc transcript-internal', lambda h=hyp_internal: entity_continuity_metrics(h, use_distance_decay=True, max_gap=None), context)
                rec.update(eg_gs)
                lg_ms = safe_dict_metric('LG_avg_degree / LG_avg_weight transcript-internal', lambda h=hyp_internal: lexical_similarity_metrics(h, window=3, sim_threshold=None), context)
                rec.update(lg_ms)
                lch = safe_dict_metric('LCh_avg_len / LCh_avg_span transcript-internal', lambda h=hyp_internal: lemma_chain_metrics(h), context)
                rec.update(lch)
                rows.append(rec)
        if rows:
            df = pd.DataFrame(rows).dropna(how='all')
            write_metric_table(select_manuscript_metrics(df, group_name=group_name, task_name=sub), excel_path)
            print(f'[output] {excel_path}')
            df2 = df.copy()
            df2.insert(0, 'task_name', sub)
            df2.insert(0, 'group_name', group_name)
            all_sub_dfs.append(df2)
        else:
            print(f'[info] No .txt learner transcripts found under: {subpath}')
    return all_sub_dfs

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Calculate the manuscript coherence metrics using the original preprocessing.')
    parser.add_argument('--input-dir', default=INPUT_DIR, help='Optional input-folder override.')
    parser.add_argument('--output-file', default=OUTPUT_FILE, help='Optional combined Excel-output override.')
    return parser

def main(argv: Optional[list[str]]=None) -> int:
    args = build_arg_parser().parse_args(argv)
    base_dir = resolve_user_path(args.input_dir)
    output_file = resolve_user_path(args.output_file)
    if not base_dir.is_dir():
        raise FileNotFoundError(f'Input folder not found: {base_dir}')
    if output_file.suffix.lower() != '.xlsx':
        raise ValueError('OUTPUT_FILE must end with .xlsx')
    groups = [p for p in sorted(base_dir.iterdir()) if p.is_dir() and p.resolve() != output_file.parent.resolve()]
    if not groups:
        raise ValueError('No group folders found in INPUT_DIR')
    for group in groups:
        tasks = [p for p in group.iterdir() if p.is_dir()]
        if not tasks:
            raise ValueError(f'No task folders found: {group}')
        for task in tasks:
            reference = find_reference_file(task)
            if reference is None:
                raise FileNotFoundError(f'Expected Reference.txt: {task}')
            learners = [p for p in task.rglob('*') if p.is_file() and p.suffix.lower() == '.txt' and (p.resolve() != reference.resolve()) and (not is_reference_path(p.parent, task))]
            if not learners:
                raise ValueError(f'No participant transcripts found: {task}')
    ensure_nltk_resources()
    ensure_udpipe_model()
    global nlp, HAS_VECTORS
    nlp = load_spacy_en(model_name='en_core_web_md', allow_blank_fallback=False)
    HAS_VECTORS = bool(getattr(nlp.vocab, 'vectors_length', 0))
    if not HAS_VECTORS:
        raise RuntimeError('The manuscript protocol requires en_core_web_md word vectors.')
    init_discoscore(device='cpu', model_name='bert-base-uncased')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for group in groups:
        frames.extend(process_one_group(group, output_file.parent / f'{group.name}_metrics', group.name, overwrite=True, sort_items=True, exclude_teacher_track=False))
    if not frames:
        raise RuntimeError('No participant metrics were calculated')
    total = select_manuscript_metrics(pd.concat(frames, ignore_index=True))
    if total.duplicated(['Group', 'Task', 'Participant']).any():
        raise ValueError('Duplicate Group/Task/Participant identifiers; use unique participant filenames within each task.')
    write_metric_table(total, output_file)
    print(f'[output] {output_file}')
    print(f'Completed: {len(total)} participant transcripts; {len(MANUSCRIPT_METRICS)} manuscript metrics each.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
