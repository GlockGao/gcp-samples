#!/bin/bash

MODEL_ID=claude-opus-4@20250514
LOCATION=us-east5
PROJECT_ID=ali-icbu-gpu-project

curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/anthropic/models/${MODEL_ID}:streamRawPredict -d \
'{
  "anthropic_version": "vertex-2023-10-16",
  "messages": [{
    "role": "user",
    "content": "Hey Claude!"
  }],
  "max_tokens": 100,
  "stream": false
}'
