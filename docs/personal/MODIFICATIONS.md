# Personal fork modifications

This file records changes made on top of the public Zed repository. Every
change supports T3 Code's Zed provider through the `zed-acp-server` binary in
`crates/eval_cli`. Releases are tagged `personal-vX.Y.Z` so they don't collide
with upstream's `vX.Y.Z` tags.

## zed-acp-server

- Runs Zed's `NativeAgent` and `AcpThread` headlessly over ACP (JSON-RPC on
  standard input and output): initialization, sessions, prompts, cancellation,
  permission requests, elicitation, streamed updates and `zed/invoke_skill`.
- Options: `--worktree`, `--data-dir`, `--model`, `--list-models` and
  `--worker-mode`.
- Uses the stored Zed sign-in without an interactive prompt, waits for the
  selected model to become available, then sets `agent.default_model`.
- `--list-models` prints every hosted model on the account as JSON, with its
  upstream provider, whether it is the account default, and its effort levels.
  Models Zed has disabled are left out.
- Thinking effort uses each model's native levels.
- `--worker-mode` runs T3 coordination roles with a fixed tool profile and no
  native delegation. `review` is read-only; `edit` and `parent` can edit. Only
  `parent` gets T3's orchestration and thread tools; workers can report to the
  parent and read threads without asking.
- Web search and native subagents keep running inside a session.

## Upstream sync

Merge upstream's latest stable release tag (`vX.Y.Z`, not `-pre`) on a
`sync/upstream-YYYY-MM-DD` branch; don't rebase. Fast-forward `main` and delete
the branch. Each sync bumps the patch version. Fork code follows upstream's
APIs rather than patching them back.

- v1.24.2: `LanguageModel` became a struct, and `acp_thread` moved to ACP v2.
  Permission requests have their own `PermissionRequestId`; the bridge maps
  each to its tool call with `permission_request(..).legacy_tool_call_id()`.
  Prompts go in as v2 content blocks, and the native agent still answers with
  `SubmissionResponse::LegacyCompleted` carrying the stop reason. The fork's
  `FakeLanguageModel::with_effort_levels` is gone; tests set fields through
  `FakeLanguageModelProvider::update_model`.

## Releases

- 0.2.2 (2026-10-17): synced with upstream v1.24.2.
- 0.2.1 (2026-10-17): `--version`, and `initialize` reports the crate version.
- 0.2.0 (2026-10-17): first tagged release. Merges the bridge into `main`;
  `integration/zed-abacus` is retired. Adds provider labels, the account
  default and disabled-model filtering to `--list-models`.
- 0.1.0: untagged builds from `integration/zed-abacus`, 2026-09-19 onward.
