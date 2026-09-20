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

The repository contains results from several experiments, but they are not complete or directly comparable because they use different models, dataset versions, and evaluation sizes. The available findings are:

- The stored zero-shot NLI evaluation contains 433 examples and achieved approximately 59.8% accuracy, 64.7% precision, 80.3% recall, and 71.7% F1 for binary manipulation detection.
- The zero-shot model had relatively high recall but also produced many false positives, especially on emotionally intense or ambiguous dialogue.
- The sentiment and pragmatic feature experiment reported 76% F1 for the manipulative class and 56% F1 for the non-manipulative class.
- The same sentiment-augmented approach reported approximately 34% accuracy for technique classification and struggled with under-represented techniques.
- The contrastive mirror-prompting experiment reported approximately 62% binary accuracy and 81% precision for the manipulative class. Its precision for non-manipulative dialogue was approximately 43%, indicating over-detection.
- The project presentations report that intent-aware prompting performed better than the compared zero-shot and multi-shot prompting configurations in the authors' API evaluation.
- Intent-aware prompting was slower because it requires additional intent-extraction calls: approximately 72.1 minutes versus 26.7 minutes for zero-shot and 36.4 minutes for multi-shot prompting.
- Tactic-level prediction was much weaker than binary detection. In the stored 433-row zero-shot artifact, exact tactic-set accuracy was approximately 0.5% and micro Jaccard was approximately 12.9%.
- A separate 108-row local IAP artifact achieved approximately 6.5% exact-match accuracy and 14.7% micro Jaccard.
- Some experiments are incomplete: for example, `Baseline_results/results_cot.csv` contains only one evaluated row, and several result files are checkpoints or partial runs rather than full benchmark reports.

Overall, the results suggest that detecting whether manipulation is present is feasible, but identifying every technique reliably and evaluating the quality of the guarded response remain open problems.

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
