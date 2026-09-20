# Manipulation Detection and Manipulation-Aware Refusal

NLP research project on detecting psychological manipulation in conversations and making LLMs more resistant to adversarial prompting.

## Problem

LLMs can be pressured into unsafe behavior through shaming, intimidation, guilt, urgency, flattery, feigned innocence, emotional dependence, or other manipulation tactics. This project studies whether an LLM can:

1. Detect whether a dialogue is manipulative.
2. Identify the tactic or tactics being used.
3. Infer the speaker's underlying intent.
4. Generate a safe, polite, and appropriately constrained response.

The intended safety flow is:

```text
Dialogue -> detection -> tactic classification -> response policy -> guarded response
```

## Dataset

The main data is `Dataset/mentalmanip_con.csv`, a collection of dialogue excerpts adapted from movie scripts. Each row contains the dialogue, a binary manipulation label, technique labels, and sometimes an exploited vulnerability.

The original dataset has 2,915 examples: 2,016 manipulative and 899 non-manipulative. `Dataset/mentalmanip_con_cleaned.csv` contains 2,647 rows. The data is imbalanced, with aggressive techniques more common than subtle ones.

Implemented technique labels include:

- Denial
- Evasion
- Feigning Innocence
- Rationalization
- Playing the Victim Role
- Playing the Servant Role
- Shaming or Belittlement
- Intimidation
- Brandishing Anger
- Accusation
- Persuasion or Seduction

`Baseline_results/manipulation_test_prompts.csv` contains 30 adversarial prompts combining manipulation with unsafe requests such as malware, phishing, credential theft, privacy violations, and academic fraud.

## Approaches

### Zero-shot baseline

`Baseline_results/prompting_baselines.py` uses a local NLI model, by default `facebook/bart-large-mnli`, in two stages:

1. Detect manipulative versus non-manipulative communication.
2. Classify up to three manipulation tactics using multi-label scoring.

The script reports binary and multi-label metrics, confusion matrices, and per-technique results. Stored outputs are in `Baseline_results/`.

### Intent-aware prompting

The code in `Intent_Aware_Prompting/code/` first summarizes the speaker's underlying intent and then supplies that intent, together with the original dialogue, to the manipulation classifier.

The local pipeline uses FLAN-T5 for intent extraction and BART for classification. Gemini-based scripts separately extract Person1 and Person2 intents and request structured JSON tactic predictions.

This approach is more interpretable than direct classification, but requires extra model calls and therefore has higher latency.

### Sentiment and pragmatic features

`Sentiment_Pragmatic/Code/sentiment_pragmatic.py` combines:

- DistilBERT semantic embeddings.
- RoBERTa sentiment embeddings.
- Counts of words such as `you` and `must`.
- Question-mark and exclamation-mark counts.

Logistic regression performs binary detection, while XGBoost classifies techniques among manipulative examples. Oversampling is used for minority classes.

### Contrastive or mirror prompting

`Contrastive_Prompting/contrastive_prompting.py` asks BART to rewrite a dialogue into a healthier version. The original and rewritten text are compared using semantic shift, cosine similarity, and syntactic change. These features are used for binary detection and technique classification.

## Response Defense

`Intent_Aware_Prompting/code/iap_interactive_pipeline.py` compares an unguarded assistant with a guarded assistant. The guarded assistant analyzes recent conversation history, detects manipulation, selects tactic-specific policies, and generates a constrained response.

Examples of policies include:

- Do not respond defensively to shaming.
- Do not let victim claims bypass safety rules.
- Do not show fear or submission under intimidation.
- Do not mirror anger.
- Maintain professional distance during seduction or inappropriate framing.
- Do not validate unethical rationalizations.

`Response Analysis/response_analyses.py` contains an earlier prototype with response templates, a simple keyword safety filter, fallback responses, and basic safety/fluency/deflection metrics.

## Findings

The experiments show that:

- Binary manipulation detection is easier than exact tactic classification.
- Models can achieve useful recall but often produce false positives.
- Aggressive tactics are easier to detect than subtle emotional tactics.
- Intent extraction improves interpretability but increases latency and does not remove class imbalance problems.
- Sentiment and pragmatic features provide useful signals but do not solve minority-class performance.
- Contrastive rewriting is a promising way to expose manipulative language, but also over-detects manipulation.

## Results

The main reported results are summarized below. The zero-shot values were recalculated from `Baseline_results/results_zeroshot.csv`; the other values are reported in `Reports/End_term_presentation.pdf`.

| Approach | Evaluation | Accuracy | Precision | Recall | F1 | Main observation |
|---|---:|---:|---:|---:|---:|---|
| Zero-shot NLI | Binary, 433 examples | 59.8% | 64.7% | 80.3% | 71.7% | Good recall, but many false positives |
| Sentiment + pragmatic features | Binary | Not reported | 81% manipulative | 72% manipulative | 76% manipulative | Weaker on non-manipulative dialogue: F1 56% |
| Sentiment + pragmatic features | Technique classification | 34% | - | - | - | Poor performance on minority techniques |
| Contrastive mirror prompting | Binary | 62% | 81% manipulative | Not reported | Not reported | Non-manipulative precision was only 43% |

The intent-aware experiments report that IAP performed better than the compared zero-shot and multi-shot prompting configurations in the authors' API evaluation. Its approximate runtime was 72.1 minutes, compared with 26.7 minutes for zero-shot and 36.4 minutes for multi-shot prompting. The additional cost comes from extracting intent before classification.

The stored multi-label outputs show that tactic-set prediction remains difficult. In the 433-row zero-shot artifact, exact tactic-set accuracy was approximately 0.5%, with micro Jaccard of approximately 12.9%. In a separate 108-row local IAP artifact, exact-match accuracy was approximately 6.5%, with micro Jaccard of approximately 14.7%. These results show that identifying *whether* manipulation exists is substantially easier than identifying every tactic exactly.

## Structure

```text
Dataset/                   Datasets
Baseline_results/          Baselines and stored evaluations
Contrastive_Prompting/     Mirror-reframing experiments
Intent_Aware_Prompting/   Intent extraction and guarded pipelines
Sentiment_Pragmatic/       Embedding and pragmatic-feature model
Response Analysis/         Response-generation prototype
Reports/                   Project presentations
```

## Limitations

This is an experimental course-project prototype, not a production safety system. Results come from different dataset versions, models, and partially completed runs. Technique labels are imbalanced, some classifiers reduce multi-label annotations to one primary label, and response safety is evaluated only with limited keyword-based checks. Further work should include human response evaluation, realistic multi-turn red-teaming, calibration analysis, and stronger minority-class handling.
