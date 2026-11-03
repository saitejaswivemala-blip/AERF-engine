# AERF Engine – Real Model Experiment Report



## Experimental Status



The AERF Engine pipeline was executed using real NLP models and real

experiment-corpus claims. The experiment infrastructure and Phase 1–3

pipeline were successfully executed.



## Real Models



- SciSpaCy scientific NER

- Sentence-transformers embeddings

- `valhalla/distilbart-mnli-12-3` NLI model



## Curated Evaluation



Four manually verified cross-document claim pairs were evaluated.



| Pair | Human Label | Prediction |

|---|---|---|

| GT-01 | supports | neutral |

| GT-02 | supports | neutral |

| GT-03 | supports | neutral |

| GT-05 | neutral | contradicts |



Real NLI inference executed successfully for all four pairs.



The diagnostic accuracy was 0.000 and macro-F1 was 0.000. Because the

sample contains only four curated pairs and is non-random, these values

must NOT be interpreted as a benchmark of real-world AERF accuracy.



## Entity-Alignment Ablation



The full configuration applies an entity-overlap gate before NLI.

Removing this gate allowed all four similarity-qualified curated pairs

to reach the real NLI model.



This demonstrates that entity alignment is an active decision stage in

the implemented pipeline rather than merely a reporting feature.



## Limitations



The current four-pair evaluation is insufficient for statistically

meaningful precision, recall, or F1 measurement. The human ground truth

also contains only a small curated subset.



The observed NLI disagreements should be reported transparently and

should not be hidden or converted into favorable results by changing

thresholds after observing the outcomes.



## Reproducibility



The curated mapping and evaluation output are preserved in the

experiments directory. Human labels were not overwritten.



The experiment should be extended with a substantially larger,

independently annotated claim-pair dataset before making quantitative

accuracy claims.



## Conclusion



The current implementation demonstrates successful execution of the

real-model NLP pipeline, candidate matching, entity alignment, NLI

inference, and controlled entity-alignment ablation.



The experiment establishes implementation functionality and provides a

diagnostic evaluation, but does not yet establish statistically valid

accuracy or superiority over baseline methods.

