#!/usr/bin/env bash
# Regenerate Pydantic models from OpenAPI specs in openapi/.
set -euo pipefail
cd "$(dirname "$0")/.."

engines=(
  youtube
  youtube_video
  youtube_transcripts
  youtube_comments
  youtube_channel
  youtube_channel_videos
)

mkdir -p src/yt_crawl/models

for engine in "${engines[@]}"; do
  uv run datamodel-codegen \
    --input "openapi/${engine}.yaml" \
    --input-file-type openapi \
    --output "src/yt_crawl/models/${engine}.py" \
    --output-model-type pydantic_v2.BaseModel \
    --target-python-version 3.13 \
    --use-standard-collections \
    --use-union-operator \
    --collapse-root-models \
    --field-constraints \
    --disable-timestamp
  echo "generated ${engine}"
done
