#!/bin/bash

# 1. Get operation status
curl -X GET \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    https://translation.googleapis.com/v3/projects/ali-icbu-gpu-project/locations/us-central1/operations/20250730-22271753939670-688a30d5-0000-21b4-829c-14223bc0029a

# 2. Get glossary
curl -X GET \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "x-goog-user-project: ali-icbu-gpu-project" \
    "https://translation.googleapis.com/v3/projects/ali-icbu-gpu-project/locations/us-central1/glossaries/aligame-glossary/glossaryEntries"