# Metric Codebook
This file describes the variables produced by
`calculate_coherence_metrics.py`. Full definitions and methodological details
are provided in the associated manuscript. The accompanying code is
authoritative for implementation details.

## Identifiers
| Variable | Description |
| --- | --- |
| `Group` | Name of the input group folder. |
| `Task` | Name of the input task folder. |
| `Participant` | Participant transcript filename without `.txt`. |

Each row represents one participant transcript within a group and task.

## Discourse-coherence measures
| Metric | Description |
| --- | --- |
| `RC_types_norm` | Repetition-type ratio. |
| `LC_synonym_norm` | Normalized lexical cohesion based on synonym relations. |
| `LC_hyper_hyponym_norm` | Normalized lexical cohesion based on hypernym and hyponym relations. |
| `LC_bigram_recurrence_norm` | Normalized recurrence of content-lemma bigrams. |
| `LG_avg_degree` | Average degree of the sentence lexical graph. |
| `LG_avg_weight` | Average edge weight of the sentence lexical graph. |
| `LCh_avg_len` | Average length of repeated-lemma lexical chains. |
| `LCh_avg_span` | Average sentence coverage of repeated-lemma lexical chains. |
| `EG_PW` | Distance-weighted entity overlap across sentences. |
| `EG_PAcc` | Distance- and role-weighted entity overlap across sentences. |
| `DS_FOCUS` | Reference-based focus measure; interpreted as a distance-style measure. |
| `DS_SENT` | Reference-based sentence-representation measure; interpreted as a similarity-style measure. |

Only `DS_FOCUS` and `DS_SENT` use the task reference transcript. The remaining measures are calculated from the participant transcript.

Valid zeros are retained. Missing or non-finite values are not replaced with zero. No cross-metric aggregation, standardization, or score inversion is performed.

For complete definitions and methodological details, see: 
Huang, Y., Fan, L., Li, D., & Cheung, A. (2026). Machine-assisted preparation in source-constrained EFL speech: real-time discourse coherence. Language Learning & Technology, 30(1).
