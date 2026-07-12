import os
import requests
from google.colab import userdata
OPENROUTER_API_KEY=userdata.get('OPENROUTER_API_KEY')
url = "https://openrouter.ai/api/v1/chat/completions"


def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=1024):
    payload = {
        "model": "openai/gpt-oss-20b:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        url,
        headers=headers,
        json=payload
    )
    if response.status_code != 200:
        print("Error:", response.status_code)
        print(response.text)
        return None
    return response.json()["choices"][0]["message"]["content"]

#Short Demonstration
system_prompt = "You are a helpful assistant."
user_prompt = "Reply with only the word: hello"
result = call_llm(system_prompt, user_prompt)
print(result)

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

import joblib
import pandas as pd
import json
from jsonschema import validate, ValidationError

best_model = joblib.load("best_model.pkl")

feature_names = best_model.feature_names_in_

customer = dict.fromkeys(feature_names, 0)
customer["checking_status"] = 2
customer["duration"] = 24
customer["credit_amount"] = 4500
customer["savings_status"] = 2
customer["employment"] = 4
customer["installment_commitment"] = 2
customer["residence_since"] = 3
customer["age"] = 35
customer["existing_credits"] = 1
customer["num_dependents"] = 1

# One-hot encoded features
customer["purpose_new car"] = 1
customer["credit_history_existing paid"] = 1
customer["personal_status_male single"] = 1
customer["other_parties_none"] = 1
customer["property_magnitude_life insurance"] = 1
customer["other_payment_plans_none"] = 1
customer["housing_own"] = 1
customer["job_skilled"] = 1
customer["own_telephone_1"] = 1
customer["foreign_worker_yes"] = 1

customer2 = customer.copy()

customer2["checking_status"] = 0
customer2["duration"] = 48
customer2["credit_amount"] = 9000
customer2["employment"] = 1
customer2["age"] = 23

customer3 = customer.copy()

customer3["checking_status"] = 3
customer3["duration"] = 6
customer3["credit_amount"] = 1500
customer3["employment"] = 4
customer3["age"] = 45

def encode_record(features):
    """
    Convert a feature dictionary into a pandas DataFrame
    compatible with the saved pipeline.
    """
    return pd.DataFrame([features])

schema = {
    "type": "object",
    "properties": {
        "prediction_label": {
            "type": "string"
        },
        "confidence_level": {
            "type": "string",
            "enum": ["High", "Medium", "Low"]
        },
        "top_reason": {
            "type": "string"
        },
        "second_reason": {
            "type": "string"
        },
        "next_step": {
            "type": "string"
        }
    },
    "required": [
        "prediction_label",
        "confidence_level",
        "top_reason",
        "second_reason",
        "next_step"
    ],
    "additionalProperties": False
}

customers = [customer, customer2, customer3]

for i, customer in enumerate(customers, start=1):

    # Encode input
    X = encode_record(customer)

    # Model prediction
    prediction = best_model.predict(X)[0]
    probability = best_model.predict_proba(X)[0][prediction]

    label = "Good Credit" if prediction == 1 else "Bad Credit"

    # Only send important features to the LLM
    important_features = {
        "checking_status": customer["checking_status"],
        "duration": customer["duration"],
        "credit_amount": customer["credit_amount"],
        "savings_status": customer["savings_status"],
        "employment": customer["employment"],
        "age": customer["age"],
        "existing_credits": customer["existing_credits"],
        "num_dependents": customer["num_dependents"]
    }

    # Create user prompt
    user_prompt = f"""
Feature Values:
{json.dumps(important_features, indent=2)}

Predicted Class:
{label}

Predicted Probability:
{probability:.4f}

Explain this prediction and return ONLY a valid JSON object with the following fields:

{{
    "prediction_label": "...",
    "confidence_level": "...",
    "top_reason": "...",
    "second_reason": "...",
    "next_step": "..."
}}
"""

    # Call LLM
    response = call_llm(system_prompt, user_prompt, temperature=0)

    try:
        response = response.strip()

        explanation = json.loads(response)

        try:
            validate(instance=explanation, schema=schema)
            validation_status = "Valid"

        except ValidationError as e:
            print("Schema Validation Error:", e)

            explanation = {
                "prediction_label": None,
                "confidence_level": None,
                "top_reason": None,
                "second_reason": None,
                "next_step": None
            }

            validation_status = "Invalid"

    except json.JSONDecodeError as e:
        print("JSON Parsing Error:", e)

        explanation = {
            "prediction_label": None,
            "confidence_level": None,
            "top_reason": None,
            "second_reason": None,
            "next_step": None
        }

        validation_status = "Invalid"

    # Print results
    print("=" * 60)
    print(f"Customer {i}")
    print("=" * 60)

    print("\nFeature Values:")
    print(important_features)

    print("\nPredicted Class:")
    print(label)

    print("\nPredicted Probability:")
    print(round(probability, 4))

    print("\nLLM Raw Response:")
    print(response)

    print("\nLLM Explanation:")
    print(json.dumps(explanation, indent=4))

    print("\nValidation Status:")
    print(validation_status)
