# yt-searchapi

An interviewed, budget-safe YouTube research crawler built on the existing
[SearchAPI](https://www.searchapi.io/docs/youtube) client. It discovers videos
through search results, channels, channel videos, and related-video edges;
classifies every evaluated candidate with strict OpenAI Responses API schemas;
and writes an append-only JSONL audit trail plus target-language transcripts.

The crawler does **not** use the official YouTube Data API.

## Runs and budgets

- `start` creates a project. `resume` continues that project or expands its
  crawl limits.
- `--max-credits` is the project's initial, cumulative SearchAPI credit grant.
  Every later `--add-credits N` increases that lifetime grant by exactly `N`.
  Dispatched requests are charged to the local ledger, including ambiguous
  failures.
- Before either session starts, the CLI checks SearchAPI's live account balance.
  The account must cover every credit the project could still dispatch in that
  session. An underfunded plan stops before a crawl request is sent or a new
  grant is committed.
- The transcript reserve is a protected floor against discovery, not a
  transcript ceiling: transcript collection may immediately transfer currently
  unspent discovery capacity when needed.
- Independent SearchAPI discovery, detail, and transcript requests run in
  bounded parallel batches. Pagination within one result chain remains
  sequential and is budgeted page by page. SearchAPI client retries are disabled.
- Each project has its own persistent request cache at
  `<project>/.cache/searchapi`. It has no TTL: identical SearchAPI requests in
  later sessions are served locally, audited as cache hits, and cost no local
  credits. Projects never share cached responses.
- The crawler checkpoints its frontier and credit ledger throughout discovery.
  A budget stop, network failure, OpenAI rate/quota error, terminal interruption,
  or accidental shutdown can be resumed without repeating completed SearchAPI
  work. A request whose outcome became ambiguous during an interruption remains
  conservatively charged.
- Final relevance uses the publication-date check and the selected-language
  transcript. Missing or relative-only dates are recorded as
  `publication_date_not_proven`.

## Install and configure

Python 3.13 and [uv](https://docs.astral.sh/uv/) are expected.

```bash
uv sync --all-groups
cp .env.example .env
```

Fill in `SEARCHAPI_API_KEY` and `OPENAI_API_KEY` in the untracked `.env` file.
No credentials are committed to this repository.

OpenAI usage and estimated GPT-5.6 Luna cost are recorded in the project audit,
but they are not a local budget.

Environment settings:

| Variable | Required | Meaning |
| --- | --- | --- |
| `SEARCHAPI_API_KEY` | For `start` and `resume`. | SearchAPI credential used for the balance preflight and crawl requests. |
| `OPENAI_API_KEY` | For `start` and `resume`. | OpenAI credential used for interview suggestions, query expansion, and classification. |
| `LOGFIRE_TOKEN` | Optional. | Enables remote Logfire telemetry without an interactive local login. |
| `YT_SEARCHAPI_DISABLE_TELEMETRY=1` | Optional. | Explicitly disables sending telemetry to Logfire. Local JSONL auditing remains enabled. |

### Optional Logfire observability

The CLI links to [Logfire](https://logfire.pydantic.dev/) for remote runtime
logs. To connect a development machine, authenticate and select a Logfire
project once:

```bash
uv run logfire auth
uv run logfire projects use <project-name>
```

For a deployed environment, set `LOGFIRE_TOKEN` instead. Telemetry is best
effort: missing credentials, an unreachable Logfire service, or an exporter
failure never stops the crawl. The link still opens Logfire so you can sign in
or finish setup. The local `.logfire/` credentials directory remains ignored;
do not commit it.

Runtime telemetry may include research queries, video IDs, project paths, run
controls, and error details. Logfire's OpenAI auto-instrumentation sends full
model conversations by default. This can include system instructions, research
topics and definitions, generated queries, video metadata, transcript excerpts
passed to classifiers, model output (including incomplete output), model and
response IDs, token usage, request duration, and exceptions. See Logfire's
[OpenAI integration documentation](https://logfire.pydantic.dev/docs/integrations/llms/openai/)
before enabling remote telemetry for sensitive research. Raw HTTP header/body
instrumentation is not separately enabled, and the application does not
deliberately attach API keys. `crawl_state.json` and the append-only JSONL audit
files remain the durable local source of truth; Logfire is remote operational
visibility, not a recovery mechanism.

## Start a new project

For a fully interactive setup, run:

```bash
uv run yt-crawl start
```

Every omitted setting is asked in the terminal. You can instead supply any
subset as flags; supplied values are not asked again. To skip all settings
prompts, supply them together:

```bash
uv run yt-crawl start \
  --topic "heat-pump retrofits in apartment buildings" \
  --language de \
  --max-credits 20 \
  --start-date 2024-01-01 \
  --project runs/heat-pump-retrofits \
  --max-queries 8 \
  --max-search-pages 1 \
  --max-channel-pages 1 \
  --max-depth 2 \
  --country de \
  --interface-language de \
  --searchapi-timeout 90 \
  --searchapi-workers 8 \
  --llm-workers 16
```

The three research-scope questions still follow; the command only makes the run
settings non-interactive.

The project path must be new or empty. If a non-empty path has a valid
`crawl_state.json`, `start` points to `resume`. If preparation was interrupted
before that checkpoint existed, both commands say that the project is not
resumable and direct you to rerun `start` with a different new/empty project
path. The project name becomes its stable run ID.

The setup prompts show progress as `Question X/N`. Once settings are resolved,
three scope questions collect the research goal and positive/negative examples.
Use Space to toggle an example, `E` to edit the highlighted example, and Enter
to accept the screen. You can also add custom examples. The review screen lets
you change any answer before the crawl begins. Confirmed scope answers are saved
in `interview_answer.jsonl`.

## Configuration reference

Every omitted `start` option is collected interactively. “Saved” below means
the value is written to `crawl_state.json` and reused by later sessions.

### Project and research settings

| Option | Start behavior | Resume behavior | Meaning |
| --- | --- | --- | --- |
| `--project PATH` | Required; the directory must be new or empty. | Required; the directory must contain a valid checkpoint. | Project location and stable run ID. Relative paths are resolved to absolute paths. |
| `--topic TEXT` | Required. | Saved; cannot be changed. | Research topic used by the interview, query expansion, and relevance classifier. |
| `--language CODE` | Required, for example `de` or `en`. | Saved; cannot be changed. | Required video/transcript language. It is also the default SearchAPI interface language. The crawler does not fall back to another transcript language. |
| `--start-date YYYY-MM-DD` | Required. | May stay unchanged or move earlier, never later. | Inclusive publication boundary. It is applied after video discovery using an exact video-detail date. Moving it earlier reopens eligible videos already discovered; it does **not** fetch older search or channel pages by itself. |
| `--country CODE` | Default `us`; two-letter SearchAPI `gl` code. | Saved; cannot be changed. | Geographic context for YouTube search, video-detail, and channel requests. It can affect ranking and localization; it is not a publication-country filter. |
| `--interface-language CODE` | Defaults to `--language`; SearchAPI `hl` code. | Saved; cannot be changed. | Language requested for YouTube interface text and localized metadata. It does not replace the transcript-language gate. |

See SearchAPI's [`hl` parameter](https://www.searchapi.io/docs/parameters/youtube/hl)
and [transcript language behavior](https://www.searchapi.io/docs/youtube-transcripts).

### Budget and frontier settings

| Option | Range/default | Resume rule | Meaning and cost behavior |
| --- | --- | --- | --- |
| `--max-credits N` | Minimum 4; required when starting. | Start only. | Initial lifetime SearchAPI grant. It is a hard local accounting limit, not a per-session allowance. |
| `--add-credits N` | Integer at least 0; default 0. | Resume only; additive. | Adds exactly `N` to the saved lifetime grant. Zero uses existing unspent capacity. Credits alone do not expose new work after a completed frontier. |
| `--max-queries N` | 2–18; default 8. | May only increase. | Enables the first `N` variants in the query plan prepared at project creation. Each enabled query has its own pagination state. Raising this beyond the number of prepared variants has no effect. |
| `--max-search-pages N` | 1–10; default 1. | May only increase. | Maximum pages fetched **for each enabled search query**, not across the whole project. For example, 8 queries × 2 pages permits at most 16 search-page requests. Raising 1 → 2 resumes each non-exhausted query from its saved continuation token; it does not repeat page 1. Each uncached page normally costs one SearchAPI credit. |
| `--max-channel-pages N` | 1–10; default 1. | May only increase. | Maximum pages fetched **for each discovered channel**. This can fan out much more than `--max-search-pages` because many channels may be discovered. Raising the limit continues every non-exhausted channel from its saved token. Each uncached page normally costs one credit. |
| `--max-depth N` | 0–5; default 2. | May only increase. | Maximum discovery-graph depth. Search-result videos start at depth 0. Depth 0 evaluates only those videos; depth 1 also admits their related videos and channels; higher values continue outward from relevant videos. Increasing depth exposes already saved deeper nodes and permits further related/channel discovery. |

Search pages and channel pages discover candidate IDs. Candidate video-detail
requests and requested transcripts can each consume additional credits, so the
page-request bounds are not total-run cost bounds. Duplicate candidates are
deduplicated by video ID, identical cached requests cost no local credits, and a
provider result chain may exhaust before reaching its configured page maximum.

The transcript reserve is derived automatically from the lifetime grant. It
protects capacity from discovery fan-out, but it is not a transcript maximum:
transcripts may transfer still-unused discovery capacity when necessary.

### Runtime settings

| Option | Range/default | Resume behavior | Meaning |
| --- | --- | --- | --- |
| `--searchapi-timeout SECONDS` | Greater than 0; default 90. | Saved; cannot currently be changed on resume. | Timeout for one SearchAPI request. A timeout is recorded as an error; ambiguous dispatched work remains conservatively charged. |
| `--searchapi-workers N` | 1–32; default 8. | Optional replacement; saved for later resumes. | Maximum concurrent independent SearchAPI requests. Pagination within one query or channel remains sequential. This affects throughput, not frontier size. |
| `--llm-workers N` | 1–32; default 16. | Optional replacement; saved for later resumes. | Maximum concurrent OpenAI video-classification requests. This affects throughput, not which candidates are eligible. |

The classifier model is currently fixed at `gpt-5.6-luna`. Transcript-stage
classification receives at most 12,000 sampled transcript characters; neither
value is currently exposed as a CLI setting.

The interview and topic expansion happen before the first resumable checkpoint
is created. The live dashboard appears while the query plan is compiling, but
the project becomes resumable only when that preparation finishes and the
crawler initializes its saved frontier. If preparation is interrupted, rerun
`start` with a different new/empty `--project` path.

## During a run

The terminal switches from the interview to a live dashboard. Its header says
either `START NEW PROJECT` or `RESUME EXISTING PROJECT`, followed by the current
task, elapsed session time, crawl totals, query and channel progress, SearchAPI
grant usage, cumulative OpenAI token and estimated-dollar usage, active/maximum
requests for each provider, cache hits, and API errors.

SearchAPI batches independent search, detail, channel, and transcript work up to
`--searchapi-workers`; a stalled request stops after `--searchapi-timeout`
seconds. Candidate classifications run concurrently up to `--llm-workers`.
Both limits are saved in the project checkpoint and restored by `resume`.
To change an existing project's parallelism, pass one or both options, for
example:

```sh
uv run yt-crawl resume --project runs/heat-pump-retrofits \
  --add-credits 0 --searchapi-workers 4 --llm-workers 8
```

On resume, the dashboard loads prior counts from `crawl_state.json` and
`api_call.jsonl` before new work begins. Routine diagnostics and Python warnings
are sent to Logfire when it is configured; they do not write a project log file
or interrupt the Rich display. The terminal is reserved for the interview, live
dashboard, final summary, and concise failure messages. The display uses
standard box drawing and is intended for a monospaced terminal font.

The cost estimate uses current GPT-5.6 Luna Standard rates: $0.20 per million
input tokens, $0.02 per million cached input tokens, $0.25 per million
cache-write tokens, and $1.20 per million output tokens. Requests over 272K
input tokens use the model's documented long-context multipliers. See the
current [GPT-5.6 Luna model page](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
for pricing and limitations.

Classification calls use explicit prompt caching. Automatic checkpointing is
disabled, and the sole manual breakpoint is after the fixed classifier
instructions and few-shot examples, before video-specific content. Other LLM
steps do not repeat a sufficiently regular prefix and therefore do not request
prompt caching. Cached and cache-write tokens are read from provider usage and
included in the dashboard estimate. See OpenAI's
[prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching).

## Start small, inspect, then expand

This is a complete small-budget workflow. The second crawl session keeps the
same interview, query plan, completed-video set, pagination tokens, credit
ledger, audit files, and project-local request cache.

```bash
# 1. Start a deliberately small project.
uv run yt-crawl start \
  --topic "heat-pump retrofits in apartment buildings" \
  --language de \
  --start-date 2024-01-01 \
  --max-credits 12 \
  --max-queries 3 \
  --max-search-pages 1 \
  --max-depth 1 \
  --project runs/heat-pump-retrofits

# 2. Inspect the cumulative project results offline.
uv run yt-crawl dashboard \
  --run-dir runs/heat-pump-retrofits

# 3. Add 30 credits, widen the same project, and adjust its parallelism.
uv run yt-crawl resume \
  --project runs/heat-pump-retrofits \
  --add-credits 30 \
  --start-date 2023-01-01 \
  --max-queries 8 \
  --max-search-pages 2 \
  --max-depth 2 \
  --searchapi-workers 4 \
  --llm-workers 8

# 4. Regenerate the dashboard to see both sessions and cumulative totals.
uv run yt-crawl dashboard \
  --run-dir runs/heat-pump-retrofits
```

`--add-credits` is additive: the example's lifetime project grant becomes
`12 + 30 = 42` credits, minus credits already spent. It does not reset the cap
to 30. Resume frontier controls are monotonic: omit a control to keep its current
value, or provide a higher value to expose additional queries, pages, or graph
depth. Worker counts are runtime controls and may move in either direction.
The publication boundary is monotonic in the other direction: `--start-date`
may move the saved date earlier, but never later. Previously discovered videos
rejected only because they preceded the old boundary are reopened when their
exact publication date falls within the newly admitted interval. Their original
decisions remain in the append-only audit stream, and resumed evaluation writes
newer decisions.
Use `--add-credits 0` to retry a failed session when the existing grant still
has unspent credits across its pools; unspent discovery credits can also fund
transcripts whenever needed, including after the current discovery work is
exhausted. Discovery can never use transcript capacity. Work marked
`deferred_budget` is
provisional, not final: resume retries it when capacity becomes available. A
project that completed under its current controls must either increase at least
one `--max-*` scope control or move `--start-date` earlier when resumed. Adding
credits alone cannot discover more work, so the CLI rejects that no-op before
checking funding or changing the saved grant. The terminal and dashboard
therefore recommend `--add-credits 0` plus one eligible scope increase while
completed projects still have credits; they recommend added credits only after
the aggregate SearchAPI balance reaches zero. Failed or budget-stopped projects
instead resume their current frontier before widening scope.

The old multi-engine example was renamed to `example.py`. It performs real,
billable requests and is not part of the crawler workflow.

## Project data

Each project is one directory, such as `runs/heat-pump-retrofits/`. All resumed
sessions append to the same audit streams. After a resume funding check passes,
the expanded grant and monotonic scope controls are atomically committed to
`crawl_state.json` before the resume audit row or any crawler provider is
initialized. `.cache/searchapi/` is the unlimited-TTL, project-local request
cache. Files are split by record type so
DuckDB and line-oriented tools can ingest them independently:

- `run_config.jsonl`, `run_status.jsonl`, `run_metric.jsonl`
- `interview_answer.jsonl`, `query.jsonl`
- `discovery_edge.jsonl`, `video_candidate.jsonl`, `channel.jsonl`
- `relevance_decision.jsonl`, `transcript.jsonl`
- `api_call.jsonl`, `budget_event.jsonl`, `run_error.jsonl`
- `raw/*.jsonl` for typed SearchAPI response provenance

Duplicate discovery edges are retained. Each video ID is normally evaluated and
transcribed at most once, but moving `--start-date` earlier can reevaluate a
candidate whose prior terminal decision was solely the old date boundary. Both
decisions remain in the audit, and downstream views use the latest one. The
model always returns a binary `relevant` or `irrelevant` decision. A metadata-stage
`relevant` is persisted as the operational `needs_transcript` disposition while
its transcript is reserved and fetched; only transcript-stage decisions are
final inclusions.
Unresolved candidates can also retain `deferred_budget` or `error`. The exact
compiled system prompt, expansion, prompt hash, and Responses API IDs are
retained for decision auditability.

## Dashboard

Generate a self-contained HTML dashboard for a project:

```bash
uv run yt-crawl dashboard --run-dir runs/heat-pump-retrofits
```

The generated dashboard opens in your default browser automatically. Use
`--no-open` when running in automation or on a headless machine.

Static dashboard options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--run-dir PATH` | Required. | Existing project directory containing the cumulative JSONL audit. |
| `--output PATH` | `<run-dir>/dashboard.html` | HTML file to create or replace. |
| `--open` / `--no-open` | `--open` | Whether to open the generated file in the default browser. Generation itself is offline. |

Generate the committed mock example without credentials or network calls:

```bash
uv run yt-crawl dashboard \
  --run-dir examples/mock-run \
  --output examples/mock-dashboard.html
```

DuckDB reads and transforms the JSONL files; the generated HTML contains its
CSS, charts, tables, and data inline and does not need a CDN.
It shows the latest project status and controls, cumulative SearchAPI grant,
spent and remaining credits, cache hits, pending/deferred work, session history,
and a copyable resume/expansion command when more work is possible. Query scope
is split into planned variants, variants executed at least once, variants held
back by the current `--max-queries`, and unfinished variants still within the
current page limit. The live terminal shows cumulative OpenAI token usage and
estimated Luna cost; both are recorded for visibility and are not an active
budget.

The repository includes a generated example at `examples/mock-dashboard.html`
backed by fictional, schema-validated data in `examples/mock-run/`.

### Live DuckDB explorer

For interactive analysis, build the React client once and run the local
read-only query layer alongside it:

```bash
cd dashboard-app
bun install
bun run build
cd ..
uv run yt-crawl dashboard-live --run-dir runs/personal-finance-de
```

`dashboard-live` serves the built React/Tailwind application and exposes a
small JSON API backed by an in-memory DuckDB database. The crawler's JSONL
files remain the source of truth: the server fingerprints them, rebuilds the
DuckDB relations when they change, and does not create a second persistent
database. Use `--no-open` for automation, or `--port` to choose another local
port. During dashboard development, run `bun run dev` in `dashboard-app` and
keep the Python command running for the `/api` proxy.

Live dashboard options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--run-dir PATH` | Required. | Existing project directory read by DuckDB. |
| `--host ADDRESS` | `127.0.0.1` | Interface to bind. The default is local-only; binding a public interface exposes the dashboard without application authentication. |
| `--port N` | `8765`; range 1–65535. | Local HTTP port. |
| `--web-dir PATH` | `dashboard-app/dist` | Directory containing the built frontend assets. |
| `--open` / `--no-open` | `--open` | Whether to open the live dashboard automatically. |

The analysis is intentionally transparent and DuckDB-only. It includes
candidate and decision funnels, observed views/likes distributions, channel
aggregates, source provenance, metadata coverage, query-plan counts, title and
description keyword frequencies, transcript search, and monthly publication
trends. Topic hits use manually reviewed keyword definitions in
`src/yt_searchapi/analysis.py`; the current map covers saving/budgeting,
investing/ETFs, retirement/pensions, debt/credit, insurance, taxes/policy,
housing, and tools/apps. Topics overlap by design, and all video-level
results can be filtered server-side and opened in a transcript/detail drawer.

The stack is React + TypeScript + Vite + Tailwind, Recharts for the charts,
and TanStack Table for the drill-down table. A Next.js backend would add
deployment and routing machinery without improving this local, append-only
workflow, so the small standard-library Python HTTP layer is the simpler
live boundary for now.

## Offline verification

```bash
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check \
  main.py example.py src/yt_searchapi tests \
  --exclude 'src/yt_searchapi/models/*'
```

All tests use injected fake clients and mock data. They never execute a
SearchAPI, OpenAI, or YouTube request.

## Design references

- [SearchAPI YouTube search and pagination](https://www.searchapi.io/docs/youtube)
- [SearchAPI video details and related videos](https://www.searchapi.io/docs/youtube-video)
- [SearchAPI channel-video pagination](https://www.searchapi.io/docs/youtube-channel-videos-api)
- [SearchAPI transcript language behavior](https://www.searchapi.io/docs/youtube-transcripts)
- [SearchAPI YouTube interface language](https://www.searchapi.io/docs/parameters/youtube/hl)
- [OpenAI Responses API migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses)
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
- [GPT-5.6 Luna pricing](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
- [GPT-5.6 prompting guidance](https://developers.openai.com/api/docs/guides/latest-model)
