#!/bin/bash
set -e
cd /home/jsuppe/speech3
source venv/bin/activate

DIR="test_speeches/round8"

echo "=== Round 8: Processing 6 speeches ==="
echo ""

echo "[1/6] Margaret Thatcher - Election Victory 1979"
python add_one_speech.py \
  "$DIR/thatcher_election_clip.mp3" \
  "Margaret Thatcher - Election Victory Speech 1979" \
  "Margaret Thatcher" \
  "political_victory" \
  '["SpeechScore","PitchPerfect","OratoScore"]'
echo ""

echo "[2/6] Margaret Thatcher - National Press Club 1995"
python add_one_speech.py \
  "$DIR/thatcher_press_club_clip.mp3" \
  "Margaret Thatcher - National Press Club Address 1995" \
  "Margaret Thatcher" \
  "political_address" \
  '["SpeechScore","PitchPerfect","OratoScore"]'
echo ""

echo "[3/6] Yale Civil War Lecture - Prof. David Blight"
python add_one_speech.py \
  "$DIR/yale_civil_war_clip.mp3" \
  "Yale Civil War Lecture 1 - Prof. David Blight" \
  "David Blight" \
  "academic_lecture" \
  '["SpeechScore","PitchPerfect"]'
echo ""

echo "[4/6] C. Spencer Pompey Oral History 1996"
python add_one_speech.py \
  "$DIR/pompey_oral_history_clip.mp3" \
  "C. Spencer Pompey - Oral History Interview 1996" \
  "C. Spencer Pompey" \
  "oral_history_elderly" \
  '["SpeechScore","VoiceVitals"]'
echo ""

echo "[5/6] Lucille & Otley Scott Oral History 1992"
python add_one_speech.py \
  "$DIR/scott_oral_history_clip.mp3" \
  "Lucille & Otley Scott - Oral History Interview 1992" \
  "Lucille & Otley Scott" \
  "oral_history_elderly" \
  '["SpeechScore","VoiceVitals"]'
echo ""

echo "[6/6] Nette Shultz & C.R. Ratliff Oral History"
python add_one_speech.py \
  "$DIR/shultz_ratliff_clip.mp3" \
  "Nette Shultz & C.R. Ratliff - Oral History Interview" \
  "Nette Shultz, C.R. Ratliff" \
  "oral_history_elderly" \
  '["SpeechScore","VoiceVitals"]'
echo ""

echo "=== Round 8 Complete ==="
