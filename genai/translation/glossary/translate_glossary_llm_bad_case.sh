#!/bin/bash

# 1. Translate with glossary
curl -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "x-goog-user-project: ali-icbu-gpu-project" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d @request_llm_bad_case.json \
    "https://translation.googleapis.com/v3/projects/ali-icbu-gpu-project/locations/us-central1:translateText"