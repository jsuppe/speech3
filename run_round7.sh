#!/bin/bash
# Process round 7 speeches one at a time (separate process per speech = no OOM)
set -e
cd /home/jsuppe/speech3
source venv/bin/activate

DIR="test_speeches"

process_speech() {
    local clip="$1" title="$2" speaker="$3" category="$4" products="$5"
    echo ""
    echo "============================================"
    echo "Processing: $title"
    echo "============================================"
    python add_one_speech.py "$clip" "$title" "$speaker" "$category" "$products"
    echo "Sleeping 5s for GPU cooldown..."
    sleep 5
}

# 1. English Fairy Tales - Tom Tit Tot
process_speech \
    "$DIR/english_fairy_tales_-_tom_tit_tot_clip.mp3" \
    "English Fairy Tales - Tom Tit Tot" \
    "Joy Chan" \
    "childrens_reading" \
    '["SpeechScore", "ReadAloud"]'

# 2. Andersen's Fairy Tales - The Emperor's New Clothes
process_speech \
    "$DIR/andersens_fairy_tales_-_the_emperors_new_clothes_clip.mp3" \
    "Andersen's Fairy Tales - The Emperor's New Clothes" \
    "LibriVox Volunteer" \
    "childrens_reading" \
    '["SpeechScore", "ReadAloud"]'

# 3. Malcolm X - House Negro and Field Negro
process_speech \
    "$DIR/house_negro_and_field_negro_clip.mp3" \
    "House Negro and Field Negro" \
    "Malcolm X" \
    "civil_rights" \
    '["SpeechScore", "OratoScore"]'

# 4. Malcolm X - On Black Nationalism
process_speech \
    "$DIR/on_black_nationalism_clip.mp3" \
    "On Black Nationalism" \
    "Malcolm X" \
    "civil_rights" \
    '["SpeechScore", "OratoScore"]'

# 5. Obama Inauguration 2009
process_speech \
    "$DIR/2009_presidential_inauguration_clip.mp3" \
    "2009 Presidential Inauguration" \
    "Barack Obama" \
    "political" \
    '["SpeechScore", "OratoScore", "PitchPerfect"]'

# 6. Malcolm X Oxford Debate
process_speech \
    "$DIR/oxford_debate_1964_clip.mp3" \
    "Oxford Debate 1964" \
    "Malcolm X" \
    "academic_debate" \
    '["SpeechScore", "OratoScore", "PitchPerfect"]'

echo ""
echo "============================================"
echo "ALL DONE"
echo "============================================"
python -c "
from speech_db import SpeechDB
db = SpeechDB()
speeches = db.list_speeches(limit=500)
new = [s for s in speeches if s['id'] > 38]
print(f'Total speeches: {len(speeches)}')
print(f'New speeches added: {len(new)}')
for s in new:
    print(f'  #{s[\"id\"]}: {s[\"title\"]}')
"
