#!/bin/bash
# SpeechScore API — curl examples
# Replace sk-YOUR-KEY with your actual API key.

API="http://localhost:8000"
KEY="sk-YOUR-KEY"

# ─── Health Check (no auth) ───────────────────────────────────────
curl -s "$API/v1/health" | python3 -m json.tool

# ─── Sync Analysis (full) ────────────────────────────────────────
curl -X POST "$API/v1/analyze" \
  -H "Authorization: Bearer $KEY" \
  -F "audio=@speech.mp3"

# ─── Quick Transcription Only ────────────────────────────────────
curl -X POST "$API/v1/analyze?preset=quick" \
  -H "Authorization: Bearer $KEY" \
  -F "audio=@speech.mp3"

# ─── Select Specific Modules ─────────────────────────────────────
curl -X POST "$API/v1/analyze?modules=transcription,audio,sentiment" \
  -H "Authorization: Bearer $KEY" \
  -F "audio=@speech.mp3"

# ─── Clinical Preset (voice quality focus) ───────────────────────
curl -X POST "$API/v1/analyze?preset=clinical" \
  -H "Authorization: Bearer $KEY" \
  -F "audio=@patient_recording.wav"

# ─── Block Bad Audio (quality gate) ─────────────────────────────
curl -X POST "$API/v1/analyze?quality_gate=block" \
  -H "Authorization: Bearer $KEY" \
  -F "audio=@speech.mp3"

# ─── Specify Language (skip auto-detect) ─────────────────────────
curl -X POST "$API/v1/analyze?language=es" \
  -H "Authorization: Bearer $KEY" \
  -F "audio=@spanish_speech.mp3"

# ─── Async Submission ────────────────────────────────────────────
JOB=$(curl -s -X POST "$API/v1/analyze?mode=async" \
  -H "Authorization: Bearer $KEY" \
  -F "audio=@long_speech.mp3")

JOB_ID=$(echo "$JOB" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "Job submitted: $JOB_ID"

# ─── Poll Job Status ─────────────────────────────────────────────
curl -s "$API/v1/jobs/$JOB_ID" \
  -H "Authorization: Bearer $KEY" | python3 -m json.tool

# ─── List Jobs ────────────────────────────────────────────────────
curl -s "$API/v1/jobs" \
  -H "Authorization: Bearer $KEY" | python3 -m json.tool

# ─── List Completed Jobs ─────────────────────────────────────────
curl -s "$API/v1/jobs?status=complete" \
  -H "Authorization: Bearer $KEY" | python3 -m json.tool

# ─── Queue Status ────────────────────────────────────────────────
curl -s "$API/v1/queue" \
  -H "Authorization: Bearer $KEY" | python3 -m json.tool

# ─── Usage Stats ─────────────────────────────────────────────────
curl -s "$API/v1/usage" \
  -H "Authorization: Bearer $KEY" | python3 -m json.tool

# ─── Async with Webhook ──────────────────────────────────────────
curl -X POST "$API/v1/analyze?mode=async&webhook_url=https://example.com/webhook" \
  -H "Authorization: Bearer $KEY" \
  -F "audio=@speech.mp3"

# ─── Auth via Query Param (alternative) ──────────────────────────
curl -X POST "$API/v1/analyze?api_key=$KEY&preset=quick" \
  -F "audio=@speech.mp3"
