Model Prediction Explanation Pipeline: Load the best-performing model from Part 3 using joblib.load('best_model.pkl'). For three hand-crafted feature-vector inputs, call .predict() and .predict_proba(). For each input, pass the feature values, predicted class, and predicted probability to the LLM API as a structured user prompt and request a JSON explanation with at least 5 required scalar fields (e.g., {"prediction_label": "string", "confidence_level": "low|medium|high", "top_reason": "string", "second_reason": "string", "next_step": "string"}). Validate each JSON response against a schema and apply the PII guardrail before each LLM call.

system_prompt ='''
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
'''

user_prompt = '''
Feature Values:
{feature_values}

Predicted Class:
{predicted_class}

Predicted Probability:
{predicted_probability}

Explain this prediction and return only the required JSON object.
'''

Temperature: The LLM was run with temperature = 0 to produce deterministic and consistent outputs. A low temperature minimizes randomness, ensuring the same input generates the same structured JSON response. This is appropriate for structured prediction explanations where reproducibility and valid JSON output are more important than creativity.

