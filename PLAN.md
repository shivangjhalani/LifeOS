# LifeOS v0 — Build Plan

Audio journaling app. Record → transcribe → summarize → search.
Single app, no server. All AI via external APIs (BYOK). All data local on device.

## Constraints

- React Native + Expo (already scaffolded in `lifeos/`)
- One provider to start: OpenAI (whisper-1, gpt-4o-mini, text-embedding-3-small)
- User supplies their own API key
- No background processing — pipeline runs while app is foregrounded
- Minimal UI — functional, not polished

## Dependencies

Install inside `lifeos/`:

```
bun add expo-audio expo-sqlite expo-file-system expo-secure-store ai @ai-sdk/openai zod
```

Enable sqlite-vec and FTS in `app.json` plugins array (see config section).

No other dependencies. Chunking is a hand-written function (~20 lines). No state management library — use React context for settings, hooks for everything else.

## app.json Changes

Add to the `plugins` array:

```json
["expo-audio"],
["expo-sqlite", { "enableFTS": true, "withSQLiteVecExtension": true }]
```

Add to `ios`:

```json
"infoPlist": {
  "NSMicrophoneUsageDescription": "LifeOS needs microphone access to record journal entries."
}
```

Add to `android`:

```json
"permissions": ["android.permission.RECORD_AUDIO"]
```

## File Structure

```
lifeos/
  app/
    _layout.tsx               # Tab navigator: Record, Journals, Search, Settings
    index.tsx                  # Record screen (home tab)
    journals/
      index.tsx                # Journal list (timeline)
      [id].tsx                 # Journal detail
    search.tsx                 # Search screen
    settings.tsx               # API key input
  src/
    types.ts                   # All domain types
    db/
      schema.ts                # SQL schema string + migration runner
      database.ts              # Open db, run migrations, CRUD functions
    services/
      ai.ts                    # transcribe(), summarize(), embed()
      recorder.ts              # Start/stop recording, save audio file
      search.ts                # indexJournal(), searchJournals()
      pipeline.ts              # processJournal() orchestrator
    lib/
      chunker.ts               # splitIntoChunks()
    hooks/
      use-database.ts          # DB provider + context
      use-recorder.ts          # Recording state machine
      use-journals.ts          # Journal list/detail queries
      use-search.ts            # Search query + results state
      use-settings.ts          # API key get/set via SecureStore
```

## Types — `src/types.ts`

```typescript
export type JournalStatus = 'recorded' | 'transcribed' | 'summarized' | 'indexed';

export interface Journal {
  id: string;
  audioUri: string;
  transcript: string | null;
  title: string | null;
  mood: string | null;
  topics: string[] | null;        // stored as JSON in SQLite
  memorableQuotes: string[] | null;
  keyLearnings: string[] | null;
  activeQuestions: string[] | null;
  durationMs: number | null;
  status: JournalStatus;
  createdAt: string;              // ISO 8601
  updatedAt: string;
}

export interface Chunk {
  id: number;
  journalId: string;
  text: string;
  chunkIndex: number;
}

export interface SummaryDoc {
  id: number;
  journalId: string;
  fieldType: 'overview' | 'learnings' | 'questions' | 'quotes';
  text: string;
}

export interface SearchResult {
  journalId: string;
  score: number;
  journal: Journal;
}
```

## Phase 1: Database — `src/db/`

### Schema — `src/db/schema.ts`

One `migrate(db)` function that runs `CREATE TABLE IF NOT EXISTS` statements. Tables:

```sql
CREATE TABLE IF NOT EXISTS journals (
  id TEXT PRIMARY KEY,
  audio_uri TEXT NOT NULL,
  transcript TEXT,
  title TEXT,
  mood TEXT,
  topics TEXT,
  memorable_quotes TEXT,
  key_learnings TEXT,
  active_questions TEXT,
  duration_ms INTEGER,
  status TEXT NOT NULL DEFAULT 'recorded',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  journal_id TEXT NOT NULL,
  text TEXT NOT NULL,
  chunk_index INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS summary_docs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  journal_id TEXT NOT NULL,
  field_type TEXT NOT NULL,
  text TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS summary_vec USING vec0(
  embedding float[1536]
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(
  embedding float[1536]
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, journal_id UNINDEXED
);
```

`1536` = dimension of `text-embedding-3-small`.

The `rowid` of `summary_vec` corresponds to the `id` of `summary_docs`.
The `rowid` of `chunk_vec` corresponds to the `id` of `chunks`.

### Database Service — `src/db/database.ts`

Open with `expo-sqlite`:

```typescript
import * as SQLite from 'expo-sqlite';

const db = await SQLite.openDatabaseAsync('lifeos.db', {
  enableFTS: true,
  extensions: [SQLite.SQLiteBundledExtensions.sqliteVec],
});
```

Wrap in a React context provider so all hooks can access `db`. Expose via `useDatabase()` hook.

CRUD functions (all in this file or split as needed):

- `insertJournal(journal: Partial<Journal>): Promise<void>`
- `updateJournal(id: string, fields: Partial<Journal>): Promise<void>`
- `getJournal(id: string): Promise<Journal>`
- `listJournals(): Promise<Journal[]>` — ordered by `created_at DESC`
- `getUnprocessedJournals(): Promise<Journal[]>` — where `status != 'indexed'`
- `insertChunks(chunks: Omit<Chunk, 'id'>[]): Promise<number[]>` — returns inserted rowids
- `insertSummaryDocs(docs: Omit<SummaryDoc, 'id'>[]): Promise<number[]>` — returns inserted rowids
- `insertVectors(table: 'summary_vec' | 'chunk_vec', rows: { rowid: number; embedding: number[] }[]): Promise<void>`
- `insertFTS(chunks: { rowid: number; text: string; journalId: string }[]): Promise<void>`
- `queryVec(table: 'summary_vec' | 'chunk_vec', embedding: number[], k: number): Promise<{ rowid: number; distance: number }[]>`
- `queryFTS(query: string): Promise<{ rowid: number; rank: number; journalId: string }[]>`

Vector insert SQL:
```sql
INSERT INTO summary_vec(rowid, embedding) VALUES (?, ?)
```
Pass embedding as JSON string: `JSON.stringify(vector)`.

Vector query SQL:
```sql
SELECT rowid, distance FROM summary_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance
```

FTS query SQL:
```sql
SELECT rowid, rank, journal_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank
```

## Phase 2: Settings — `src/hooks/use-settings.ts`

Use `expo-secure-store` to store the OpenAI API key:

```typescript
import * as SecureStore from 'expo-secure-store';

function useSettings() {
  const [apiKey, setApiKeyState] = useState<string | null>(null);

  useEffect(() => {
    SecureStore.getItemAsync('openai_api_key').then(setApiKeyState);
  }, []);

  const setApiKey = async (key: string) => {
    await SecureStore.setItemAsync('openai_api_key', key);
    setApiKeyState(key);
  };

  return { apiKey, setApiKey };
}
```

Wrap in a context provider so all services can access it.

## Phase 3: AI Service — `src/services/ai.ts`

Three functions. All take `apiKey` as a parameter.

### transcribe(apiKey, audioUri) → string

Direct `fetch` to OpenAI Whisper API. No SDK needed for this one.

```typescript
async function transcribe(apiKey: string, audioUri: string): Promise<string> {
  const formData = new FormData();
  formData.append('file', {
    uri: audioUri,
    type: 'audio/m4a',
    name: 'recording.m4a',
  } as any);
  formData.append('model', 'whisper-1');

  const res = await fetch('https://api.openai.com/v1/audio/transcriptions', {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}` },
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.error?.message || 'Transcription failed');
  return data.text;
}
```

### summarize(apiKey, transcript) → Summary

Use AI SDK `generateObject` with Zod schema.

```typescript
import { generateObject } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { z } from 'zod';

const SummarySchema = z.object({
  title: z.string().describe('A concise title for this journal entry'),
  mood: z.string().describe('A single word or short phrase capturing the overall emotional tone'),
  topics: z.array(z.string()).describe('2-5 main themes discussed'),
  memorableQuotes: z.array(z.string()).describe('1-3 notable direct quotes from the transcript'),
  keyLearnings: z.array(z.string()).describe('1-3 insights or realizations expressed'),
  activeQuestions: z.array(z.string()).describe('Unresolved questions or decisions mentioned'),
});

type Summary = z.infer<typeof SummarySchema>;

async function summarize(apiKey: string, transcript: string): Promise<Summary> {
  const openai = createOpenAI({ apiKey });
  const { object } = await generateObject({
    model: openai('gpt-4o-mini'),
    schema: SummarySchema,
    prompt: `Analyze this personal journal transcript. Be faithful to what was said. Use the speaker's own words where possible.\n\nTranscript:\n${transcript}`,
  });
  return object;
}
```

### embed(apiKey, texts) → number[][]

Use AI SDK `embedMany`.

```typescript
import { embedMany } from 'ai';

async function embed(apiKey: string, texts: string[]): Promise<number[][]> {
  const openai = createOpenAI({ apiKey });
  const { embeddings } = await embedMany({
    model: openai.embedding('text-embedding-3-small'),
    values: texts,
  });
  return embeddings;
}
```

## Phase 4: Audio Recorder — `src/services/recorder.ts`

Use `expo-audio` for recording and `expo-file-system` to persist the file.

Flow:
1. Request microphone permission via `Audio.requestPermissionsAsync()`
2. Start recording with `useAudioRecorder` hook (high quality preset, m4a output)
3. On stop, copy the temp file to `${FileSystem.documentDirectory}audio/${journalId}.m4a`
4. Return the permanent URI

Expose as a `useRecorder()` hook in `src/hooks/use-recorder.ts` that wraps the `expo-audio` hook and adds:
- `isRecording` state
- `startRecording()` — requests permission if needed, starts
- `stopRecording()` → `{ uri: string, durationMs: number }`

## Phase 5: Chunker — `src/lib/chunker.ts`

Simple sentence-boundary chunking. No dependencies.

```typescript
function splitIntoChunks(text: string, maxChars = 1500): string[] {
  const sentences = text.split(/(?<=[.!?])\s+/);
  const chunks: string[] = [];
  let current = '';

  for (const sentence of sentences) {
    if (current.length + sentence.length > maxChars && current) {
      chunks.push(current.trim());
      current = sentence;
    } else {
      current += (current ? ' ' : '') + sentence;
    }
  }
  if (current.trim()) chunks.push(current.trim());
  return chunks;
}
```

## Phase 6: Search — `src/services/search.ts`

Two responsibilities: **indexing** and **retrieval**.

### indexJournal(db, apiKey, journalId)

Called after summarization. Steps:

1. Load journal from DB (needs title, topics, transcript, summary fields).
2. **Build summary docs** (Layer 1) — 4 text documents per journal:
   - `overview`: `"${title}. Topics: ${topics.join(', ')}"`
   - `learnings`: `keyLearnings.join('. ')`
   - `questions`: `activeQuestions.join('. ')`
   - `quotes`: `"${memorableQuotes.join('. ')} Mood: ${mood}"`
3. Insert summary docs into `summary_docs` table. Keep returned rowids.
4. **Build transcript chunks** (Layer 2):
   - `splitIntoChunks(transcript)`
   - Prefix each chunk: `"[${title} | ${topics.join(', ')}] ${chunkText}"`
5. Insert chunks into `chunks` table (store the *unprefixed* text). Keep returned rowids.
6. Insert chunk text into `chunks_fts` table (unprefixed text + journal_id).
7. **Embed everything in one batch**:
   - Combine summary doc texts + prefixed chunk texts into one array.
   - Call `embed(apiKey, allTexts)`.
   - Split returned embeddings back into summary embeddings and chunk embeddings.
8. Insert embeddings into `summary_vec` and `chunk_vec` using the matching rowids.

### searchJournals(db, apiKey, query, limit = 10) → SearchResult[]

Two-layer search fused into journal-level results.

1. **Embed query**: `embed(apiKey, [query])` → `queryVec`
2. **Dense retrieval** (Layer 1):
   - `queryVec('summary_vec', queryVec, 50)` → list of `{ rowid, distance }`
   - Join with `summary_docs` to get `journalId` for each rowid.
   - Group by `journalId`, keep the *minimum distance* per journal.
3. **BM25 retrieval**:
   - `queryFTS(query)` → list of `{ rowid, rank, journalId }`
   - Group by `journalId`, keep the *minimum rank* (best BM25) per journal.
4. **Score fusion**:
   - Normalize dense distances to `[0, 1]` (min-max across results, then `1 - normalized` so higher = better).
   - Normalize BM25 ranks to `[0, 1]` (FTS5 rank is negative; take absolute value, then min-max normalize, then `1 - normalized`).
   - Fused score = `0.6 * denseScore + 0.4 * bm25Score`.
   - For journals that appear in only one result set, use `0` for the missing score.
5. Sort by fused score descending, return top `limit` with journal data.

## Phase 7: Processing Pipeline — `src/services/pipeline.ts`

### processJournal(db, apiKey, journalId)

Orchestrates the full pipeline. Resumable via status field.

```
async function processJournal(db, apiKey, journalId):
  journal = getJournal(journalId)

  if status == 'recorded':
    transcript = transcribe(apiKey, journal.audioUri)
    updateJournal(journalId, { transcript, status: 'transcribed' })

  if status == 'recorded' or 'transcribed':
    summary = summarize(apiKey, journal.transcript)
    updateJournal(journalId, { ...summary, status: 'summarized' })

  if status in ('recorded', 'transcribed', 'summarized'):
    indexJournal(db, apiKey, journalId)
    updateJournal(journalId, { status: 'indexed' })
```

Each step reads the latest journal state from DB, so if the app dies mid-pipeline, it resumes from the last completed step on next open.

### On App Start

In the root layout or a startup hook:

```typescript
const unprocessed = await getUnprocessedJournals();
for (const journal of unprocessed) {
  await processJournal(db, apiKey, journal.id);
}
```

Process sequentially. Show a small indicator in the UI if processing is happening.

## Phase 8: UI — Minimal Screens

### Root Layout (`app/_layout.tsx`)

Tab navigator with 4 tabs: Record, Journals, Search, Settings.

Wrap everything in:
1. `DatabaseProvider` — opens DB, runs migrations, provides via context
2. `SettingsProvider` — loads API key, provides via context

If no API key is set, show the Settings screen as a blocking modal.

### Record Screen (`app/index.tsx`)

- Large circular record button in center.
- Tap to start, tap to stop.
- On stop: generate UUID, save audio, insert journal row (status = 'recorded'), kick off `processJournal` in the background (non-blocking, fire-and-forget with error catch).
- Show recording duration while active.
- Show a small status toast/indicator when processing starts.

### Journals Screen (`app/journals/index.tsx`)

- FlatList of all journals, sorted newest first.
- Each row: title (or "Processing..." if null), date, mood pill, status indicator.
- Tap to navigate to journal detail.

### Journal Detail (`app/journals/[id].tsx`)

- Title, mood, date at top.
- Topics as pills/tags.
- Summary sections: key learnings, memorable quotes, active questions.
- Expandable transcript section at bottom.
- If journal is still processing, show which step is in progress.

### Search Screen (`app/search.tsx`)

- Text input at top.
- On submit: call `searchJournals()`, display results as a list.
- Each result: title, date, relevance score.
- Tap to navigate to journal detail.
- Show empty state when no query or no results.

### Settings Screen (`app/settings.tsx`)

- Text input for OpenAI API key (masked).
- Save button.
- Show current key status (set / not set).

## Implementation Order

Build and test in this sequence. Each phase produces something testable.

1. **`src/types.ts`** — Define all types.
2. **`src/db/schema.ts`** + **`src/db/database.ts`** — Schema, migrations, CRUD. Test by inserting/reading a dummy journal.
3. **`src/hooks/use-database.ts`** — Context provider.
4. **`src/hooks/use-settings.ts`** — SecureStore wrapper. Build `app/settings.tsx` to test.
5. **`app/_layout.tsx`** — Tab navigator with providers.
6. **`src/services/ai.ts`** — All three functions. Test `summarize` with a hardcoded transcript string.
7. **`src/services/recorder.ts`** + **`src/hooks/use-recorder.ts`** — Recording. Build `app/index.tsx` (record button) to test. Verify audio file is saved.
8. **`src/lib/chunker.ts`** — Pure function, trivial to test.
9. **`src/services/search.ts`** (indexing only) — `indexJournal`. Test with a manually inserted journal that has transcript + summary.
10. **`src/services/pipeline.ts`** — Wire together transcribe → summarize → index. Test end-to-end: record → process → verify journal has all fields.
11. **`src/services/search.ts`** (retrieval) — `searchJournals`. Build `app/search.tsx` to test.
12. **`app/journals/index.tsx`** + **`app/journals/[id].tsx`** — Journal list and detail views.
13. **`src/hooks/use-journals.ts`** + **`src/hooks/use-search.ts`** — Remaining hooks.

## Notes for the Implementing Agent

- **Use bun** as the package manager (already set up).
- **Expo SDK 55** is already installed. Check `expo-audio` and `expo-sqlite` API docs for the exact hook/function signatures at this version.
- **Do not over-abstract.** If a function is only called in one place, inline it. Extract only when there's actual reuse.
- **JSON columns**: `topics`, `memorable_quotes`, `key_learnings`, `active_questions` are stored as `JSON.stringify(array)` in SQLite and parsed with `JSON.parse()` on read.
- **IDs**: Use `crypto.randomUUID()` for journal IDs.
- **Error handling**: Try/catch around each pipeline step. If a step fails, leave the journal at its current status. The user can retry by reopening the app (the startup hook picks up unprocessed journals).
- **No tests for v0.** Get it working first.
- **Styling**: Use inline styles or a minimal `StyleSheet.create`. Dark-mode compatible (respect system theme via `useColorScheme`). No design system, no component library.
