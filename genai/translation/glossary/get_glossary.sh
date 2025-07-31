#!/bin/bash

# 1. Get operation status
curl -X GET \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    https://translation.googleapis.com/v3/projects/ali-icbu-gpu-project/locations/us-central1/operations/20250730-19531753930381-688a32d1-0000-294a-8ed4-14223bad6756

# 2. Get glossary
curl -X GET \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "x-goog-user-project: ali-icbu-gpu-project" \
    "https://translation.googleapis.com/v3/projects/ali-icbu-gpu-project/locations/us-central1/glossaries/aligame-glossary/glossaryEntries"