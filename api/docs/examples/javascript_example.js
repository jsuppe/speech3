/**
 * SpeechScore API — JavaScript/Node.js examples
 * Uses native fetch (Node 18+) or install node-fetch for older versions.
 */

const API_URL = "http://localhost:8000";
const API_KEY = "sk-YOUR-KEY";

const headers = { Authorization: `Bearer ${API_KEY}` };

// ── Health Check ─────────────────────────────────────────────────

async function healthCheck() {
  const resp = await fetch(`${API_URL}/v1/health`);
  const data = await resp.json();
  console.log("Health:", data);
}

// ── Sync Analysis ────────────────────────────────────────────────

async function analyzeSync(filePath) {
  const fs = await import("fs");
  const blob = new Blob([fs.readFileSync(filePath)]);
  const form = new FormData();
  form.append("audio", blob, filePath.split("/").pop());

  const resp = await fetch(`${API_URL}/v1/analyze`, {
    method: "POST",
    headers,
    body: form,
  });

  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(`${err.error.code}: ${err.error.message}`);
  }

  return resp.json();
}

// ── Async Analysis with Polling ──────────────────────────────────

async function analyzeAsync(filePath) {
  const fs = await import("fs");
  const blob = new Blob([fs.readFileSync(filePath)]);
  const form = new FormData();
  form.append("audio", blob, filePath.split("/").pop());

  // Submit
  const submitResp = await fetch(`${API_URL}/v1/analyze?mode=async`, {
    method: "POST",
    headers,
    body: form,
  });

  const { job_id } = await submitResp.json();
  console.log(`Job submitted: ${job_id}`);

  // Poll
  while (true) {
    const pollResp = await fetch(`${API_URL}/v1/jobs/${job_id}`, { headers });
    const job = await pollResp.json();

    if (job.status === "complete") {
      console.log("Complete!", job.result.transcript?.substring(0, 80));
      return job;
    }
    if (job.status === "failed") {
      throw new Error(`Job failed: ${job.error}`);
    }

    console.log(`Status: ${job.status} (${job.progress || "waiting..."})`);
    await new Promise((r) => setTimeout(r, 2000));
  }
}

// ── Quick Preset ─────────────────────────────────────────────────

async function quickTranscribe(filePath) {
  const fs = await import("fs");
  const blob = new Blob([fs.readFileSync(filePath)]);
  const form = new FormData();
  form.append("audio", blob, filePath.split("/").pop());

  const resp = await fetch(`${API_URL}/v1/analyze?preset=quick`, {
    method: "POST",
    headers,
    body: form,
  });

  const data = await resp.json();
  console.log("Transcript:", data.transcript);
  console.log("SNR:", data.audio_quality?.snr_db, "dB");
  return data;
}

// ── Usage Check ──────────────────────────────────────────────────

async function checkUsage() {
  const resp = await fetch(`${API_URL}/v1/usage`, { headers });
  const usage = await resp.json();
  console.log("Usage:", usage);
}

// ── Run Examples ─────────────────────────────────────────────────

(async () => {
  await healthCheck();

  // Uncomment to run:
  // const result = await analyzeSync("speech.mp3");
  // console.log("WPM:", result.audio_analysis?.speaking_rate?.overall_wpm);

  // const job = await analyzeAsync("long_speech.mp3");

  // await quickTranscribe("speech.mp3");

  // await checkUsage();
})();
