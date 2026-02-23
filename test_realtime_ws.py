#!/usr/bin/env python3
"""Quick test of the real-time pronunciation WebSocket"""

import asyncio
import json
import numpy as np
import websockets

async def test_pronunciation_stream():
    uri = "ws://localhost:8000/ws/pronunciation-stream"
    
    async with websockets.connect(uri) as ws:
        # 1. Initialize with target text
        await ws.send(json.dumps({
            "type": "init",
            "target_text": "She sells sea shells"
        }))
        
        response = await ws.recv()
        print(f"Init response: {response}")
        
        # 2. Simulate audio chunks (silence + fake "speech")
        # In reality, this would come from a microphone
        sample_rate = 16000
        chunk_duration = 0.1  # 100ms chunks
        chunk_samples = int(sample_rate * chunk_duration)
        
        # Send a few chunks of "silence"
        print("Sending silence chunks...")
        for _ in range(5):
            silence = np.zeros(chunk_samples, dtype=np.float32)
            await ws.send(silence.tobytes())
            await asyncio.sleep(0.1)
        
        # Send "speech" (just noise for testing - won't produce good results)
        print("Sending noise (simulated speech)...")
        for _ in range(10):
            noise = np.random.randn(chunk_samples).astype(np.float32) * 0.3
            await ws.send(noise.tobytes())
            await asyncio.sleep(0.1)
        
        # Send silence again (word boundary)
        print("Sending silence (word boundary)...")
        for _ in range(5):
            silence = np.zeros(chunk_samples, dtype=np.float32)
            await ws.send(silence.tobytes())
            await asyncio.sleep(0.1)
        
        # Check for any word results
        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=0.5)
                print(f"Word result: {response}")
        except asyncio.TimeoutError:
            pass
        
        # 3. End session
        await ws.send(json.dumps({"type": "end"}))
        summary = await ws.recv()
        print(f"Summary: {summary}")

if __name__ == "__main__":
    asyncio.run(test_pronunciation_stream())
