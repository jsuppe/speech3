# SpeechScore Changelog

All notable changes to the SpeechScore platform (API, Web Dashboard, Mobile App).

## [Unreleased]

### 2026-02-17

#### Mobile App (Flutter) - Project Recording Reminders
- **Added** Project recording reminders — get notified to record regularly
  - ReminderSettings widget in create project dialog
  - Daily, weekly, monthly frequency options
  - Time picker for reminder time
  - Day selector for weekly reminders
- **Added** NotificationService for scheduling local notifications
- **Added** Deep link handling — notification tap opens record screen with project pre-selected
- **Added** Global notification toggle in Settings screen
- **Added** API support for reminder fields: `reminder_enabled`, `reminder_frequency`, `reminder_days`, `reminder_time`

#### API (SpeakFit)
- **Added** Metric Exemplar Library — audio samples demonstrating each grade (A-F) for each metric
  - Generated from 64 curated test speeches (MLK, JFK, FDR, Churchill, etc.)
  - JSON and Markdown reference docs at `/workspace/METRIC_EXEMPLARS.md`
  - Useful for calibrating scoring expectations
- **Added** `/v1/metrics/exemplars` endpoint — returns audio URLs for each grade per metric
- **Added** `/v1/metrics/exemplars/{metric}` endpoint — single metric exemplars
- **Added** Static serving of exemplar audio files at `/static/exemplars/`

#### Mobile App (Flutter) - Metric Help Enhancements
- **Added** "🎧 Hear Examples" section in metric help modal
  - Shows A/B/C/D/F grades with play buttons
  - Famous speakers as examples (MLK, JFK, FDR, Churchill, Reagan, etc.)
  - Streams audio directly from API

### 2026-02-13 (Parity Check)

#### Mobile App (Flutter) - Patch 46
- **Fixed** Group project badge not appearing immediately after creation — now updates UI instantly before full refresh

#### Notes
- Investigated folder icon bug (Trello card) — code has `Icons.folder`, appears to be a non-issue or already fixed
- Shorebird build shows DEX differences due to environment drift; used `--allow-native-diffs` for pure Dart patch

### 2026-02-12 (Parity Check)

#### Web Dashboard (speechscore.dev)
- **Added** Group meeting project view with participant summaries
  - Expandable participant cards showing word count, recording count, duration
  - LLM-generated summaries for each participant (speaking style, key points, opinions, action items, to-dos)
  - "Regenerate" button to refresh AI summaries
  - Purple "Group Meeting" badge on group project headers
  - Parity with mobile app's group meeting view (Patch 18)
- **Added** API methods for `getMeetingData` and `getParticipantSummary` to auth-api.ts
- **Added** Type definitions for `MeetingData`, `ParticipantSummary`, `ParticipantLLMSummary` to api.ts

### 2026-02-10

#### Mobile App (Flutter) - Patch 42
- **Fixed** TypeError crash in ProjectsScreen when widget unmounted during load — added mounted check before setState

#### API (api.speechscore.dev)
- **Fixed** Speaker detail endpoint (`/v1/speakers/{name}/detail`) — was failing with "no such column: aud.audio_path" error
- **Fixed** Group analysis endpoint (`/v1/speeches/{id}/group`) now uses diarized segments when available, showing actual speaker labels instead of "Speaker 1"
- **Fixed** Diarization dependency issue — downgraded setuptools to <81 to restore pkg_resources compatibility with webrtcvad
- **Added** `unidentified_speakers` count to `/v1/projects` endpoint for group meeting projects

#### Mobile App (Flutter) - Patch 37
- **Fixed** TypeError crash in ProjectDetailScreen when widget unmounted during load — added mounted check before setState

#### Mobile App (Flutter) - Patch 36
- **Added** Orange badge on group project icons showing count of unidentified speakers
- **Added** "X unidentified voice(s)" text under group project names when speakers need identification
- **Fixed** Storage timeout issues — SharedPreferences, FlutterSecureStorage, Play Integrity all have 3s timeouts
- **Disabled** Play Integrity API until GCP project number is configured

#### Infrastructure
- **Re-ran** diarization on 4 recordings (9560-9564) that were analyzed before the profile bug fix — now showing correct multi-speaker data
- **Hosted** Shorebird APK at api.speechscore.dev/static/speechscore-1.0.24+25.apk

#### Mobile App (Flutter) - Patch 35
- **Fixed** Recordings from projects now inherit the project's profile (group meeting → `group` profile for diarization)
- **Fixed** Timestamps display in user's local timezone instead of UTC

#### Mobile App (Flutter) - Patch 34
- **Fixed** WakelockPlus platform channel crash — wrapped calls in try-catch to handle app startup/shutdown timing

#### Mobile App (Flutter) - Patch 33
- **Fixed** SharedPreferences platform channel error on cold start — added retry mechanism with delays to TierService.init()
- **Improved** Error handling for 422 server errors — now displays actual server error message instead of generic exception text

### 2026-02-09

#### API (api.speechscore.dev)
- **Added** `GET /v1/speakers/{name}/detail` — Get speaker detail with speeches and projects
- **Added** `POST /v1/speakers/merge` — Merge two speaker profiles into one
- **Enhanced** `GET /v1/projects/{id}/meeting` — Now includes `participant_summaries` with per-speaker stats and full text

#### Mobile App (Flutter) - Patch 18
- **Enhanced** Group meeting project view:
  - **Participant summaries** — Expandable cards showing each speaker's word count, duration, and full transcript
  - **Timeline transcript** — Full chronological transcript with speaker avatars and connecting lines
  - Visual improvements: speaker-colored avatars, timestamps, duration badges

#### Mobile App (Flutter) - Patch 17
- **Added** "Participants" button in Quick Actions on Home screen
- **Added** Participants list screen — view all identified speakers
- **Added** Participant detail screen:
  - View speaker info (sample count, date added)
  - Rename speaker
  - Delete speaker profile
  - **Merge** speaker profiles (combine two speakers if they're the same person)
  - List of projects featuring this speaker
  - List of recordings by this speaker (with playback)

#### Web Dashboard (speechscore.dev) - Parity Check
- **Added** Merge speaker profiles button on Speakers page
  - Select source speaker → choose target → merge voice embeddings
  - Source profile deleted after merge, samples combined into target
  - Parity with mobile Patch 17 merge feature

---

### 2026-02-08 (Parity Check)

#### Web Dashboard (speechscore.dev)
- **Added** Speaker Profiles page (`/speakers`) — view, rename, delete voice fingerprints
- **Fixed** Feature parity matrix — mobile already had delete/edit/compare, matrix was outdated

#### Parity Matrix Update
- Confirmed mobile delete, edit, compare features already implemented in prior patches
- Updated matrix to reflect true feature parity

---

### 2026-02-08

#### API (api.speechscore.dev)
- **Added** "Group / Meeting" analysis profile — multi-party focus with speaker tracking
  - Diarization-focused: transcription + speaker identification, no individual scoring
  - New endpoint: `GET /v1/projects/{id}/analysis` — AI summary, calendar extraction, conflict detection
  - New endpoint: `POST /v1/projects/{id}/query` — Ask questions about project recordings
- **Added** Project analysis features:
  - LLM-powered summary of all sessions
  - Calendar extraction (dates, deadlines, events mentioned)
  - Action items extraction
  - Conflict detection between sessions (e.g., conflicting deadlines)
  - Speaker participation analysis

#### Mobile App (Flutter) - Patch 7
- **Added** "Group / Meeting" profile option (👥)
- **Added** Project selector on New Recording screen
- **Added** Analysis profile shown in project dropdown
- **Changed** Primary speaker defaults to user's name (or "Primary Speaker" if not available)

#### Web Dashboard (speechscore.dev)
- **Added** "Group / Meeting" profile option
- **Added** Project selector on Record/Upload page
- **Added** Analysis profile shown in project dropdown
- **Changed** Primary speaker defaults to user's name (or "Primary Speaker" if not available)

---

### 2026-02-07

#### API (api.speechscore.dev)
- **Added** Speaker voice fingerprinting — auto-identify speakers from voice embeddings
  - New `speaker_embeddings` table in database
  - New endpoints: `GET /v1/speakers`, `POST /v1/speakers/register`, `POST /v1/speakers/identify`, `DELETE /v1/speakers/{name}`
  - Pipeline integration: auto-detects speaker if ≥80% voice match
- **Added** Pipeline documentation under Methodology page (admin-only Section 7)
  - Architecture diagram (PlantUML)
  - Module reference cards
  - API module/preset documentation
  - Tech stack overview

#### Web Dashboard (speechscore.dev)
- **Added** Projects feature — organize recordings into projects
  - `/projects` page with create/list functionality
  - `/projects/[id]` detail page with add existing recordings
  - Default scoring profile per project
- **Added** Coach message logging to database (`coach_messages` table)
- **Added** Admin dashboard with platform stats, charts, top users

#### Mobile App (Flutter)
- **Added** Projects feature (Patch 4 for 1.0.24+25)
  - Projects screen with create/list
  - Project detail with "Add Existing" and "Record New" buttons
  - Assign recordings to projects
- **Added** Progress tracking screen with score trends over time
- **Added** Search on History screen (title, transcript, speaker)

---

## Feature Parity Matrix

| Feature | API | Web | Mobile | Notes |
|---------|-----|-----|--------|-------|
| User authentication (Google) | ✅ | ✅ | ✅ | |
| Record & analyze speech | ✅ | ✅ | ✅ | |
| View recordings/history | ✅ | ✅ | ✅ | |
| Search recordings | ✅ | ✅ | ✅ | |
| Projects | ✅ | ✅ | ✅ | |
| Progress tracking | ✅ | ✅ | ✅ | |
| Speech Coach AI | ✅ | ✅ | ✅ | |
| Compare recordings | ✅ | ✅ | ✅ | |
| Speaker fingerprinting | ✅ | ✅ | ✅ | |
| Speaker merge | ✅ | ✅ | ✅ | Combine duplicate voice profiles |
| Delete recordings | ✅ | ✅ | ✅ | |
| Edit recording metadata | ✅ | ✅ | ✅ | |
| Scoring profiles | ✅ | ✅ | ✅ | |
| Group meeting view | ✅ | ✅ | ✅ | Participant list + LLM summaries |
| Admin dashboard | ✅ | ✅ | N/A | Web-only feature |

---

## Pending Parity Items

*No pending parity gaps — all platforms at feature parity!* 🎉


## 2026-02-11

### API
- Added score caching to speeches table (cached_score, cached_score_profile columns)
- Progress endpoint now uses cached scores - O(1) instead of O(n) score calculations
- Added ?unassigned=true parameter to /speeches endpoint for efficient filtering

- Added LLM participant summary endpoint: GET /projects/{id}/participant/{name}/summary
- Uses llama3.1:8b for context-aware analysis of meeting contributions
- Summaries are cached and auto-regenerate when meeting is updated

