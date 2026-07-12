# Track C: Model Prediction Explanation Pipeline

Load the best-performing model from Part 3 using `joblib.load('best_model.pkl')`. For three hand-crafted feature-vector inputs, call `.predict()` and `.predict_proba()`. For each input, pass the feature values, predicted class, and predicted probability to the LLM API as a structured user prompt and request a JSON explanation with at least 5 required scalar fields (e.g., `{"prediction_label": "string", "confidence_level": "low|medium|high", "top_reason": "string", "second_reason": "string", "next_step": "string"}`). Validate each JSON response against a schema and apply the PII guardrail before each LLM call.

## System Prompt

```text
You are an AI assistant that explains credit risk model predictions.
You will receive:

- Customer feature values.
- The predicted class (Good Credit or Bad Credit).
- The predicted probability.

Your task is to explain the prediction and return ONLY a valid JSON object.

The JSON must contain exactly these fields:
{
  "prediction_label": "Good Credit | Bad Credit",
  "confidence_level": "High | Medium | Low",
  "top_reason": "string",
  "second_reason": "string",
  "next_step": "string"
}

Rules:
- Return only valid JSON.
- Do not include markdown or any extra text.
- Base the explanation only on the provided feature values and prediction.
- Keep each explanation concise and professional.
- If the predicted probability is greater than or equal to 0.80, set "confidence_level" to "High".
- If the predicted probability is between 0.60 and 0.79, set "confidence_level" to "Medium".
- Otherwise, set "confidence_level" to "Low".
```

## User Prompt Template

```text
Feature Values:
{json.dumps(important_features, indent=2)}

Predicted Class:
{label}

Predicted Probability:
{probability:.4f}

Explain this prediction and return ONLY a valid JSON object with the following fields:

{
"prediction_label": "...",
"confidence_level": "...",
"top_reason": "...",
"second_reason": "...",
"next_step": "..."
}
```

## Temperature

The LLM was run with **temperature = 0** to produce deterministic and consistent outputs. A low temperature minimizes randomness, ensuring the same input generates the same structured JSON response. This is appropriate for structured prediction explanations where reproducibility and valid JSON output are more important than creativity.

## Results

| Feature Input | Predicted Class | Probability | Explanation JSON | Validation Status |
|---------------|-----------------|------------:|------------------|-------------------|
| checking_status=2, duration=24, credit_amount=4500, savings_status=2, employment=4, age=35, existing_credits=1, num_dependents=1 | Good Credit | 0.5942 | `{"prediction_label":"Good Credit","confidence_level":"Medium","top_reason":"Stable employment and moderate credit amount suggest reliable repayment capacity.","second_reason":"Positive checking and savings status indicate good financial habits.","next_step":"Proceed with a standard credit offer while monitoring repayment behavior."}` | Valid |
| checking_status=0, duration=48, credit_amount=9000, savings_status=2, employment=1, age=23, existing_credits=1, num_dependents=1 | Good Credit | 0.5551 | `{"prediction_label":"Good Credit","confidence_level":"Low","top_reason":"Moderate credit amount and short loan duration are viewed favorably by the model.","second_reason":"Employment status and savings indicate financial stability, while few existing credits and dependents reduce risk.","next_step":"Proceed with approval but monitor repayment history closely."}` | Valid |
| checking_status=3, duration=6, credit_amount=1500, savings_status=2, employment=4, age=45, existing_credits=1, num_dependents=1 | Good Credit | 0.7601 | `{"prediction_label":"Good Credit","confidence_level":"Medium","top_reason":"Short loan duration (6 months) and moderate credit amount indicate low risk.","second_reason":"Stable employment and few dependents support repayment capacity.","next_step":"Approve the loan and offer a competitive interest rate."}` | Valid |
