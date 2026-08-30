# yt-searchapi

A YouTube research crawler using [SearchAPI](https://www.searchapi.io/docs/youtube) and OpenAI. It finds videos for a topic, keeps those that match the research scope, and saves their target-language transcripts and an append-only audit trail.

![Crawl steps](diagram.png)

This is research code accompanying the related article: link. It is AI-generated and will not be maintained.

## Run the crawler

Python 3.13 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --all-groups
cp .env.example .env
```

Add `SEARCHAPI_API_KEY` and `OPENAI_API_KEY` to `.env`, then start a project:

```bash
uv run yt-crawl start
```

The terminal asks for the topic, language, publication-date boundary, budget, project folder, and crawl limits. It also asks a few questions to define which videos count as relevant. The crawler then shows live progress and writes the project data to the folder you chose.

To run without the settings prompts, pass the required values yourself:

```bash
uv run yt-crawl start \
  --topic "personal finance" \
  --language de \
  --start-date 2024-01-01 \
  --max-credits 12 \
  --project runs/personal-finance-de
```

The project folder must be new or empty. The research-scope questions still appear, even when run settings are supplied as flags.

## Resume or expand a project

Use the same project folder to continue after a budget stop, interruption, or failure. Completed work and cached SearchAPI responses are reused.

```bash
uv run yt-crawl resume \
  --project runs/personal-finance-de \
  --add-credits 30 \
  --max-search-pages 2
```

`--add-credits` adds to the project's lifetime SearchAPI allowance; it does not replace it. To find more material after a completed run, also widen at least one scope limit (`--max-queries`, `--max-search-pages`, `--max-channel-pages`, or `--max-depth`) or move `--start-date` earlier. These scope limits can only increase on resume.

Use `--add-credits 0` when unspent project credits remain. Before a session starts, the CLI checks that the SearchAPI account balance can cover its remaining possible requests.

## What you get

Each project is self-contained and resumable. Its main files are:

| File                                                             | Contents                                                               |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `crawl_state.json`                                               | Checkpoint, saved settings, and credit ledger.                         |
| `transcript.jsonl`                                               | Collected target-language transcripts.                                 |
| `relevance_decision.jsonl`                                       | Metadata and transcript relevance decisions.                           |
| `video_candidate.jsonl`, `channel.jsonl`, `discovery_edge.jsonl` | Discovered videos, channels, and their links.                          |
| `api_call.jsonl`, `budget_event.jsonl`, `run_error.jsonl`        | Provider activity, credit use, and errors.                             |
| `.cache/searchapi/`                                              | Project-local request cache; identical later requests cost no credits. |

All session records append to these files.

## Common controls

`--max-credits` is the hard SearchAPI budget for a new project. Search-result pages, channel pages, video details, and transcripts can all consume credits. OpenAI usage is recorded in the audit but has no local budget cap.

`--language` is the required video and transcript language. `--start-date` is the inclusive earliest publication date. `--country` and `--interface-language` control SearchAPI's YouTube context.

`--searchapi-workers` and `--llm-workers` control speed, not crawl scope. `--searchapi-retries` controls additional attempts for transient SearchAPI failures; every dispatched attempt is audited and charged.

For the full option reference, use:

```bash
uv run yt-crawl start --help
uv run yt-crawl resume --help
```

## Logfire telemetry

Local JSONL auditing is always on. `LOGFIRE_TOKEN` optionally enables remote Logfire telemetry; set `YT_SEARCHAPI_DISABLE_TELEMETRY=1` to disable it explicitly.

## Development checks

```bash
uv run --frozen pytest -q
uv run --frozen ruff check .
```

Tests use fake clients and make no SearchAPI, OpenAI, or YouTube requests.
