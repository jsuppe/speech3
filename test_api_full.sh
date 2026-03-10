#!/bin/bash
# Full API test suite for Supabase migration

API="http://localhost:8000/v1"
KEY="sk-5b7fd66025ead6b8731ef73b2c970f26"

echo "=========================================="
echo "SpeakFit API Test Suite - Supabase Migration"
echo "=========================================="

# 1. Health checks
echo -e "\n[1] Health Check..."
curl -s "$API/../health" | jq -c '.'

echo -e "\n[2] Cloud Status..."
curl -s "$API/cloud/status" | jq -c '.'

# 2. User endpoints
echo -e "\n[3] User Versions (admin)..."
curl -s "$API/auth/admin/users/versions?api_key=$KEY" | jq -c '{total, first_user: .users[0].email}'

# 3. List speeches
echo -e "\n[4] List Speeches (user_id=1)..."
SPEECHES=$(curl -s "$API/speeches?limit=3&api_key=$KEY")
echo "$SPEECHES" | jq -c '{count: (.speeches | length), first: .speeches[0].id}'

# 4. Get single speech
SPEECH_ID=$(echo "$SPEECHES" | jq -r '.speeches[0].id')
echo -e "\n[5] Get Speech $SPEECH_ID..."
curl -s "$API/speeches/$SPEECH_ID?api_key=$KEY" | jq -c '{id: .speech.id, title: .speech.title, has_audio}'

# 5. Get speech analysis
echo -e "\n[6] Get Analysis for Speech $SPEECH_ID..."
curl -s "$API/speeches/$SPEECH_ID/analysis?api_key=$KEY" | jq -c '{has_result: (.result != null), wpm: .result.audio_analysis.speaking_rate.overall_wpm}'

# 6. Get audio (check if accessible)
echo -e "\n[7] Get Audio for Speech $SPEECH_ID..."
AUDIO_RESP=$(curl -s -w "\n%{http_code}" "$API/speeches/$SPEECH_ID/audio?api_key=$KEY" -o /tmp/test_audio.opus)
HTTP_CODE=$(echo "$AUDIO_RESP" | tail -1)
if [ "$HTTP_CODE" = "200" ]; then
    SIZE=$(stat -f%z /tmp/test_audio.opus 2>/dev/null || stat -c%s /tmp/test_audio.opus 2>/dev/null)
    echo "Audio OK: $SIZE bytes"
else
    echo "Audio HTTP: $HTTP_CODE"
fi

# 7. Count speeches
echo -e "\n[8] Count Speeches..."
curl -s "$API/speeches/count?api_key=$KEY" | jq -c '.'

# 8. Projects
echo -e "\n[9] List Projects..."
curl -s "$API/projects?api_key=$KEY" | jq -c '{count: (. | length)}'

# 9. Coach messages
echo -e "\n[10] Coach Messages..."
curl -s "$API/coach/messages?limit=3&api_key=$KEY" | jq -c '{count: (.messages | length)}'

# 10. Version endpoint
echo -e "\n[11] API Version..."
curl -s "$API/version" | jq -c '.'

# 11. Test speech creation (dry run - create and delete)
echo -e "\n[12] Test Speech Create/Delete Flow..."
# Create a test speech via analyze endpoint would require audio
# Instead, test update on existing speech
echo "Skipping create test (requires audio upload)"

# 12. Update speech test
echo -e "\n[13] Update Speech $SPEECH_ID (add tag)..."
UPDATE_RESP=$(curl -s -X PATCH "$API/speeches/$SPEECH_ID?api_key=$KEY" \
  -H "Content-Type: application/json" \
  -d '{"tags": ["test-tag"]}')
echo "$UPDATE_RESP" | jq -c '{id: .id, tags}'

# Revert the tag
curl -s -X PATCH "$API/speeches/$SPEECH_ID?api_key=$KEY" \
  -H "Content-Type: application/json" \
  -d '{"tags": []}' > /dev/null

echo -e "\n=========================================="
echo "Test Complete!"
echo "=========================================="
