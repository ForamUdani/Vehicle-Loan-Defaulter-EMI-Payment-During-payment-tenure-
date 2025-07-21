#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load the trained model
#model = joblib.load('automl_model.pkl')

import cloudpickle

with open("automl_pipeline.pkl", "rb") as f:
    model = cloudpickle.load(f)


app = FastAPI(debug=True)

# Define the request data format
class LoanData(BaseModel):
    asset_manufacturer_id: float
    area_code: float
    credit_score: float
    new_loan_accounts_in_last_6_months: float
    overdue_accounts_in_last_6_months: float
    avg_account_age: float
    credit_history_length: float
    no_of_inquiries_in_last_month: float
    no_of_loan_accounts: float
    active_loan_accounts: float
    overdue_accounts: float
    existing_loan_balance: float
    total_disbursed_amount: float
    current_installment: float
    employment_type: str
    aadhaar_available:  bool
    pan_available: bool
    voter_id_available: bool 
    driving_licence_available: bool 
    passport_available: bool
    # Add any other features required for prediction

@app.post("/predict")
def predict(data: LoanData):
    # Extract data from the request
    try: 
        employment_map = {"salaried": 1, "self-employed": 0}
        emp_val = employment_map.get(data.employment_type.lower())
        if emp_val is None:
            raise HTTPException(status_code=400, detail="Invalid employment_type. Use 'salaried' or 'self-employed'.")
    
        input_dict = {
            "asset_manufacturer_id": data.asset_manufacturer_id,
            "area_code": data.area_code,
            "credit_score": data.credit_score,
            "new_loan_accounts_in_last_6_months": data.new_loan_accounts_in_last_6_months,
            "overdue_accounts_in_last_6_months": data.overdue_accounts_in_last_6_months,
            "avg_account_age": data.avg_account_age,
            "credit_history_length": data.credit_history_length,
            "no_of_inquiries_in_last_month": data.no_of_inquiries_in_last_month,
            "no_of_loan_accounts": data.no_of_loan_accounts,
            "active_loan_accounts": data.active_loan_accounts,
            "overdue_accounts": data.overdue_accounts,
            "existing_loan_balance": data.existing_loan_balance,
            "total_disbursed_amount": data.total_disbursed_amount,
            "current_installment": data.current_installment,
            "employment_type": data.employment_type,  # pass the string here
            "aadhaar_available": data.aadhaar_available,
            "pan_available": data.pan_available,
            "voter_id_available": data.voter_id_available,
            "driving_licence_available": data.driving_licence_available,
            "passport_available": data.passport_available
        }

        input_df = pd.DataFrame([input_dict])

        # Make a prediction using the model
        prediction = model.predict(input_df)
        return {"prediction": int(prediction[0])}  # Return the prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# In[16]:


import os
print("Current working directory:", os.getcwd())


# In[ ]:


#https://chatgpt.com/share/68247f1e-aa9c-800f-a947-e50f78068345

