# LLM Personality Measurement & Shaping System

This repository contains a simple implementation of the system described in the prompt: a FastAPI backend that can measure LLM personality using IPIP items, basic psychometric utilities, a personality shaping component, and a Streamlit dashboard.

## Run backend

Install dependencies (preferably in a virtualenv):

```pwsh
python -m pip install -r backend/requirements.txt
pip install uvicorn fastapi
```

Run the FastAPI app:

```pwsh
cd backend
uvicorn main:app --reload
```

## Run dashboard

```pwsh
cd dashboard
streamlit run app.py
```
