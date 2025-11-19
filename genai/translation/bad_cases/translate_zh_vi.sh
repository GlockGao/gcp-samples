#!/bin/bash

# 1. Translate with glossary
curl -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "x-goog-user-project: veo-testing" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d @request_llm.json \
    "https://translation.googleapis.com/v3/projects/veo-testing/locations/us-central1:translateText"