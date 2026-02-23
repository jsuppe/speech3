#!/bin/bash
cd /home/melchior/speech3
source venv/bin/activate

while true; do
    echo "$(date): Starting analyzer..."
    python -u analyze_oratory_batch.py 2>&1
    ANALYZED=$(python -c "import json; d=json.load(open('oratory_analysis_state.json')); print(len(d['analyzed']))")
    echo "$(date): Analyzer stopped. $ANALYZED files analyzed so far."
    
    if [ "$ANALYZED" -ge 400 ]; then
        echo "Analysis complete!"
        break
    fi
    
    sleep 2
done
