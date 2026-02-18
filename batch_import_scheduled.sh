#!/bin/bash
# Run continuously between 1am and 5am ET, processing 10 files per batch
# with a short pause between batches to let the API breathe

cd /home/melchior/speech3
source venv/bin/activate

while true; do
    HOUR=$(TZ="America/New_York" date +%H)
    
    # Only run between 1am and 5am ET
    if [ "$HOUR" -ge 1 ] && [ "$HOUR" -lt 5 ]; then
        echo "$(date): Starting batch of 10..."
        python batch_import_all.py
        
        # Pause 30 seconds between batches to reduce DB pressure
        echo "$(date): Batch complete. Pausing 30s..."
        sleep 30
    else
        echo "$(date): Outside batch window (1am-5am ET). Exiting."
        exit 0
    fi
done
