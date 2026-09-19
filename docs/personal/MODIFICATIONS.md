# Personal fork modifications

This file records changes made on top of the public Zed repository.

## 2026-09-19: Headless ACP server

- Added the `zed-acp-server` binary to `crates/eval_cli`.
- Added JSON-RPC over standard input and output for ACP initialization, session creation, prompts, cancellation, permission requests, elicitation, streamed updates, and `zed/invoke_skill`.
- Reused Zed's `NativeAgent` and `AcpThread` so T3 Code can run agent sessions without the desktop app.
- Added `--worktree`, `--data-dir`, and `--model` options.
- Added headless Zed model authentication and readiness checks before session creation. The server waits for stored Zed credentials to make the selected model available, then sets `agent.default_model`.
- The implementation supports the T3 Code model choices `zed.dev/claude-sonnet-5` and `zed.dev/gpt-5.6-luna`.
- Built and tested the debug binary at `target/debug/zed-acp-server` against T3 Code's ACP integration tests.
