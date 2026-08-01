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

Useful crawl controls:

```text
--max-queries 8
--max-search-pages 1
--max-channel-pages 1
--max-depth 2
--country us
--interface-language <selected video language>
--searchapi-timeout 90
--searchapi-workers 8
--llm-workers 16
```

New projects suggest 8 concurrent SearchAPI requests and 16 concurrent OpenAI
classifications. Pass lower values when you intentionally want to throttle a
run. Resumed projects keep their saved worker counts unless you pass either
worker option again; the selected value is saved for later resumes.

The selected video language is passed to SearchAPI as the default YouTube
interface language, which prevents localized titles such as German titles from
being requested in English. `--interface-language` can override it. The country
is passed as `gl`, and transcript requests use the selected language as `lang`.
See SearchAPI's [`hl` parameter](https://www.searchapi.io/docs/parameters/youtube/hl)
and [transcript language behavior](https://www.searchapi.io/docs/youtube-transcripts).

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
to 30. Resume controls are monotonic: omit a control to keep its current value,
or provide a higher value to expose additional queries, pages, or graph depth.
Use `--add-credits 0` to retry a failed session when the existing grant still
has unspent credits across its pools; unspent discovery credits can also fund
transcripts whenever needed, including after the current discovery work is
exhausted. Discovery can never use transcript capacity. Work marked
`deferred_budget` is
provisional, not final: resume retries it when capacity becomes available. A
project that completed under its current controls
must increase at least one `--max-*` scope control when resumed. Adding credits
alone cannot discover more work, so the CLI rejects that no-op before checking
funding or changing the saved grant. The terminal and dashboard therefore
recommend `--add-credits 0` plus one eligible scope increase while completed
projects still have credits; they recommend added credits only after the
aggregate SearchAPI balance reaches zero. Failed or budget-stopped projects
instead resume their current frontier before widening scope.

The old multi-engine example was renamed to `example.py`. It performs real,
billable requests and is not part of the crawler workflow.

## Project data

Each project is one directory, such as `runs/heat-pump-retrofits/`. All resumed
sessions append to the same audit streams. After a resume funding check passes,
the expanded grant and monotonic controls are atomically committed to
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

Duplicate discovery edges are retained, while each video ID is evaluated and
transcribed at most once across the project. The model always returns a binary
`relevant` or `irrelevant` decision. A metadata-stage `relevant` is persisted
as the operational `needs_transcript` disposition while its transcript is
reserved and fetched; only transcript-stage decisions are final inclusions.
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
