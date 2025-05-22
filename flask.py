# host_api.py
from flask import Flask, request
import subprocess
import os

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    transcript_path = data['transcript_path']
    output_path = data['output_path']
    
    result = subprocess.run(
        [os.path.expanduser('~/dev/speech3/run-json.sh'), transcript_path, output_path],
        capture_output=True, text=True
    )
    
    return {
        'stdout': result.stdout,
        'stderr': result.stderr,
        'success': result.returncode == 0
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
