//! Headless Zed ACP (Agent Client Protocol) Server.
//!
//! Provides a headless runtime for Zed agents speaking ACP over standard input/output.
//! T3 Code (or any ACP client) can spawn this binary as a background process to run
//! Zed NativeAgent, execute tools, request permissions, and invoke skills.

mod headless;

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash as _, Hasher as _};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use acp_thread::{
    AcpThread, AcpThreadEvent, AgentConnection as _, AgentThreadEntry, AssistantMessageChunk,
    ElicitationEntryId, PermissionOptions, SelectedPermissionOutcome, ToolCallStatus,
};
use agent::{NativeAgent, NativeAgentConnection, Templates, ThreadStore};
use agent_client_protocol::schema::v1 as acp;
use anyhow::Context as _;
use clap::Parser;
use futures::StreamExt;
use futures::channel::mpsc;
use gpui::{App, AppContext as _, AsyncApp, Entity, Subscription, UpdateGlobal as _};
use language_model::{ConfiguredModel, LanguageModelRegistry, SelectedModel};
use project::Project;
use release_channel::{AppCommitSha, AppVersion};
use reqwest_client::ReqwestClient;
use serde_json::{Value, json};
use settings::SettingsStore;
use util::path_list::PathList;

use headless::AgentCliAppState;

#[derive(Parser, Debug)]
#[command(
    name = "zed-acp-server",
    about = "Headless Zed Agent Client Protocol (ACP) Server"
)]
struct Args {
    /// Working directory / worktree path
    #[arg(long)]
    worktree: Option<PathBuf>,

    /// Custom Zed data directory
    #[arg(long)]
    data_dir: Option<String>,

    /// Model identifier to select (e.g. anthropic/claude-3-7-sonnet)
    #[arg(long)]
    model: Option<String>,

    /// Print the current hosted model catalog as JSON and exit.
    #[arg(long)]
    list_models: bool,

    /// Disable native delegation, with an optional read-only tool profile.
    #[arg(long, value_parser = ["parent", "review", "edit"])]
    worker_mode: Option<String>,
}

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_id(prefix: &str) -> String {
    let id = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}_{}", prefix, id)
}

#[cfg(test)]
thread_local! {
    static RPC_MESSAGES: RefCell<Vec<Value>> = const { RefCell::new(Vec::new()) };
}

fn send_rpc(msg: &Value) {
    #[cfg(test)]
    RPC_MESSAGES.with(|messages| messages.borrow_mut().push(msg.clone()));
    if let Ok(serialized) = serde_json::to_string(msg) {
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        use std::io::Write;
        let _ = writeln!(lock, "{}", serialized);
        let _ = lock.flush();
    }
}

fn send_response(id: &Value, result: Value) {
    send_rpc(&json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    }));
}

fn send_error(id: &Value, code: i64, message: &str) {
    send_rpc(&json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message
        }
    }));
}

fn prompt_result(outcome: anyhow::Result<Option<acp::PromptResponse>>) -> anyhow::Result<Value> {
    let stop_reason = match outcome {
        Ok(Some(response)) => match response.stop_reason {
            acp::StopReason::EndTurn => "end_turn",
            acp::StopReason::MaxTokens => "max_tokens",
            acp::StopReason::Cancelled => "cancelled",
            _ => "end_turn",
        },
        Ok(None) => "end_turn",
        Err(error) => return Err(error),
    };
    Ok(json!({ "stopReason": stop_reason }))
}

fn send_notification(method: &str, params: Value) {
    send_rpc(&json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params
    }));
}

#[allow(dead_code)]
struct SessionEntry {
    session_id: String,
    client_session_id: String,
    project: Entity<Project>,
    connection: Rc<NativeAgentConnection>,
    acp_thread: Entity<AcpThread>,
    workdir: PathBuf,
    // (entry_index, chunk_index) -> emitted text byte length
    emitted_chunk_lengths: HashMap<(usize, usize), usize>,
    // Keep fingerprints rather than retaining a second copy of tool output.
    emitted_tool_calls: HashMap<acp::ToolCallId, u64>,
}

struct PendingPermission {
    session_id: String,
    tool_call_id: acp::ToolCallId,
    options: Vec<acp::PermissionOption>,
}

struct PendingElicitation {
    session_id: String,
    entry_id: ElicitationEntryId,
}

struct ServerState {
    app_state: Arc<AgentCliAppState>,
    default_worktree: Option<PathBuf>,
    default_model: Option<String>,
    worker_mode: Option<String>,
    model_ready: bool,
    sessions: HashMap<String, SessionEntry>,
    pending_permissions: HashMap<String, PendingPermission>,
    pending_elicitations: HashMap<String, PendingElicitation>,
}

impl ServerState {
    fn new(
        app_state: Arc<AgentCliAppState>,
        default_worktree: Option<PathBuf>,
        default_model: Option<String>,
    ) -> Self {
        Self {
            app_state,
            default_worktree,
            default_model,
            worker_mode: None,
            model_ready: false,
            sessions: HashMap::new(),
            pending_permissions: HashMap::new(),
            pending_elicitations: HashMap::new(),
        }
    }

    fn on_thread_event(
        &mut self,
        session_id: &str,
        thread: &Entity<AcpThread>,
        event: &AcpThreadEvent,
        cx: &mut App,
    ) {
        let client_session_id = self
            .sessions
            .get(session_id)
            .map(|session| session.client_session_id.clone())
            .unwrap_or_else(|| session_id.to_string());
        match event {
            // `AcpThread` flushes its buffered streaming Markdown immediately
            // before emitting `Stopped`. The flush updates the Markdown entity
            // without another `EntryUpdated`, so inspect the entries here or
            // the final tail never reaches ACP clients.
            AcpThreadEvent::NewEntry
            | AcpThreadEvent::EntryUpdated(_)
            | AcpThreadEvent::Stopped(_) => {
                let Some(session) = self.sessions.get_mut(session_id) else {
                    return;
                };
                let thread_read = thread.read(cx);
                let entries = thread_read.entries();

                for (entry_ix, entry) in entries.iter().enumerate() {
                    if let AcpThreadEvent::EntryUpdated(updated_ix) = event
                        && entry_ix != *updated_ix
                    {
                        continue;
                    }
                    match entry {
                        AgentThreadEntry::AssistantMessage(msg)
                            if session_id == client_session_id =>
                        {
                            for (chunk_ix, chunk) in msg.chunks.iter().enumerate() {
                                match chunk {
                                    AssistantMessageChunk::Message { block, id } => {
                                        let markdown = block.to_markdown(cx);
                                        let emitted = session
                                            .emitted_chunk_lengths
                                            .entry((entry_ix, chunk_ix))
                                            .or_insert(0);
                                        if markdown.len() > *emitted {
                                            let delta = &markdown[*emitted..];
                                            *emitted = markdown.len();
                                            send_notification(
                                                "session/update",
                                                json!({
                                                    "sessionId": client_session_id,
                                                    "update": {
                                                        "sessionUpdate": "agent_message_chunk",
                                                        "messageId": id.as_ref().map(ToString::to_string),
                                                        "content": {
                                                            "type": "text",
                                                            "text": delta
                                                        }
                                                    }
                                                }),
                                            );
                                        }
                                    }
                                    AssistantMessageChunk::Thought { block, id } => {
                                        let markdown = block.to_markdown(cx);
                                        let emitted = session
                                            .emitted_chunk_lengths
                                            .entry((entry_ix, chunk_ix))
                                            .or_insert(0);
                                        if markdown.len() > *emitted {
                                            let delta = &markdown[*emitted..];
                                            *emitted = markdown.len();
                                            send_notification(
                                                "session/update",
                                                json!({
                                                    "sessionId": client_session_id,
                                                    "update": {
                                                        "sessionUpdate": "agent_thought_chunk",
                                                        "messageId": id.as_ref().map(ToString::to_string),
                                                        "content": {
                                                            "type": "text",
                                                            "text": delta
                                                        }
                                                    }
                                                }),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        AgentThreadEntry::ToolCall(call) => {
                            let status_str = match &call.status {
                                ToolCallStatus::Pending => "pending",
                                ToolCallStatus::WaitingForConfirmation { .. } => "pending",
                                ToolCallStatus::InProgress => "in_progress",
                                ToolCallStatus::Completed => "completed",
                                ToolCallStatus::Failed => "failed",
                                ToolCallStatus::Rejected => "failed",
                                ToolCallStatus::Canceled => "failed",
                            };

                            let kind_str = match call.kind {
                                acp::ToolKind::Read => "read",
                                acp::ToolKind::Edit => "edit",
                                acp::ToolKind::Delete => "delete",
                                acp::ToolKind::Move => "move",
                                acp::ToolKind::Search => "search",
                                acp::ToolKind::Execute => "execute",
                                acp::ToolKind::Think => "think",
                                acp::ToolKind::Fetch => "fetch",
                                acp::ToolKind::SwitchMode => "switch_mode",
                                _ => "other",
                            };

                            let mut update = json!({
                                "toolCallId": client_tool_call_id(session_id, &client_session_id, &call.id),
                                "title": call.label.read(cx).source(),
                                "kind": kind_str,
                                "status": status_str,
                                "rawInput": call.raw_input,
                                "rawOutput": call.raw_output,
                                "locations": call.locations,
                                "content": call.content.iter().map(|content| json!({
                                    "type": "content",
                                    "content": { "type": "text", "text": content.to_markdown(cx) },
                                })).collect::<Vec<_>>(),
                            });
                            if let Some(name) = &call.tool_name {
                                update["name"] = json!(name.as_ref());
                            }
                            if let Some(info) = &call.subagent_session_info {
                                update["_meta"] = json!({ "subagent_session_info": info });
                            }
                            let mut hasher = std::collections::hash_map::DefaultHasher::new();
                            update.hash(&mut hasher);
                            let fingerprint = hasher.finish();
                            let previous = session
                                .emitted_tool_calls
                                .insert(call.id.clone(), fingerprint);
                            if previous == Some(fingerprint) {
                                continue;
                            }
                            update["sessionUpdate"] = json!(if previous.is_none() {
                                "tool_call"
                            } else {
                                "tool_call_update"
                            });
                            send_notification(
                                "session/update",
                                json!({
                                    "sessionId": client_session_id,
                                    "update": update,
                                }),
                            );
                        }
                        _ => {}
                    }
                }
            }
            AcpThreadEvent::ToolAuthorizationRequested(tool_call_id) => {
                if self.pending_permissions.values().any(|pending| {
                    pending.session_id == session_id && &pending.tool_call_id == tool_call_id
                }) {
                    return;
                }
                let thread_read = thread.read(cx);
                if let Some((_, call)) = thread_read.tool_call(tool_call_id) {
                    if let ToolCallStatus::WaitingForConfirmation { options, .. } = &call.status {
                        let acp_options = match options {
                            PermissionOptions::Flat(opts) => opts.clone(),
                            PermissionOptions::Dropdown(choices) => choices
                                .iter()
                                .flat_map(|c| [c.allow.clone(), c.deny.clone()])
                                .collect(),
                            PermissionOptions::DropdownWithPatterns { choices, .. } => choices
                                .iter()
                                .flat_map(|c| [c.allow.clone(), c.deny.clone()])
                                .collect(),
                        };

                        let options_vec = if acp_options.is_empty() {
                            vec![
                                acp::PermissionOption::new(
                                    acp::PermissionOptionId::new("allow"),
                                    "Allow",
                                    acp::PermissionOptionKind::AllowOnce,
                                ),
                                acp::PermissionOption::new(
                                    acp::PermissionOptionId::new("deny"),
                                    "Deny",
                                    acp::PermissionOptionKind::RejectOnce,
                                ),
                            ]
                        } else {
                            acp_options
                        };

                        let req_id = next_id("perm");
                        self.pending_permissions.insert(
                            req_id.clone(),
                            PendingPermission {
                                session_id: session_id.to_string(),
                                tool_call_id: tool_call_id.clone(),
                                options: options_vec.clone(),
                            },
                        );

                        let options_json: Vec<Value> = options_vec
                            .iter()
                            .map(|opt| {
                                json!({
                                    "optionId": opt.option_id.to_string(),
                                    "name": opt.name.to_string(),
                                    "kind": match opt.kind {
                                        acp::PermissionOptionKind::AllowOnce => "allow_once",
                                        acp::PermissionOptionKind::AllowAlways => "allow_always",
                                        acp::PermissionOptionKind::RejectOnce => "reject_once",
                                        acp::PermissionOptionKind::RejectAlways => "reject_always",
                                        _ => "allow_once",
                                    }
                                })
                            })
                            .collect();

                        let title = call.label.read(cx).source().to_string();

                        send_rpc(&json!({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "method": "session/request_permission",
                            "params": {
                                "sessionId": client_session_id,
                                "toolCall": {
                                    "toolCallId": client_tool_call_id(session_id, &client_session_id, tool_call_id),
                                    "title": title,
                                    "status": "pending",
                                    "rawInput": call.raw_input
                                },
                                "options": options_json
                            }
                        }));
                    }
                }
            }
            AcpThreadEvent::ElicitationRequested(entry_id) => {
                let req_id = next_id("elicit");
                self.pending_elicitations.insert(
                    req_id.clone(),
                    PendingElicitation {
                        session_id: session_id.to_string(),
                        entry_id: entry_id.clone(),
                    },
                );

                send_rpc(&json!({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": "elicitation/create",
                    "params": {
                        "sessionId": client_session_id,
                        "elicitationId": entry_id.0.to_string(),
                    }
                }));
            }
            AcpThreadEvent::AvailableCommandsUpdated(commands)
                if session_id == client_session_id =>
            {
                let cmds_json: Vec<Value> = commands
                    .iter()
                    .map(|cmd| {
                        json!({
                            "name": cmd.name,
                            "description": cmd.description
                        })
                    })
                    .collect();
                send_notification(
                    "session/update",
                    json!({
                        "sessionId": client_session_id,
                        "update": {
                            "sessionUpdate": "available_commands_update",
                            "availableCommands": cmds_json
                        }
                    }),
                );
            }
            _ => {}
        }
    }
}

const MODEL_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(30);
const MODEL_DISCOVERY_POLL_INTERVAL: Duration = Duration::from_millis(100);

fn find_configured_model(selected: &SelectedModel, cx: &App) -> Option<ConfiguredModel> {
    let registry = LanguageModelRegistry::global(cx);
    let provider = registry.read(cx).provider(&selected.provider)?;
    let model = provider
        .provided_models(cx)
        .into_iter()
        .find(|model| model.id() == selected.model)?;
    Some(ConfiguredModel { provider, model })
}

async fn ensure_model_ready(
    app_state: &AgentCliAppState,
    model: Option<&str>,
    cx: &mut AsyncApp,
) -> anyhow::Result<()> {
    let Some(model) = model else {
        return Ok(());
    };
    let selected = model
        .parse::<SelectedModel>()
        .map_err(|error| anyhow::anyhow!(error))?;

    // The desktop client signs in before its cloud language-model provider
    // discovers zed.dev models. The headless ACP server has no desktop
    // startup path, so it must perform that step before creating a session.
    if selected.provider.0.as_ref() == "zed.dev" {
        app_state
            .client
            .sign_in_with_optional_connect(true, cx)
            .await
            .context("authenticating the Zed language-model provider")?;
    }

    let started_at = Instant::now();
    loop {
        let configured = cx.update(|cx| find_configured_model(&selected, cx));
        if let Some(configured) = configured {
            cx.update(|cx| {
                LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                    registry.set_default_model(Some(configured), cx);
                });
            });
            return Ok(());
        }

        if started_at.elapsed() >= MODEL_DISCOVERY_TIMEOUT {
            anyhow::bail!("Zed language model {model} was not discovered");
        }

        cx.background_executor()
            .timer(MODEL_DISCOVERY_POLL_INTERVAL)
            .await;
    }
}

fn main() {
    let args = Args::parse();

    if let Some(data_dir) = &args.data_dir {
        paths::set_custom_data_dir(data_dir);
    }

    let app_commit_sha = option_env!("ZED_COMMIT_SHA").map(|s| AppCommitSha::new(s.to_owned()));
    let app_version = AppVersion::load(
        env!("ZED_PKG_VERSION"),
        option_env!("ZED_BUILD_ID"),
        app_commit_sha,
    );

    let user_agent = format!(
        "Zed Agent ACP Server/{} ({}; {})",
        app_version,
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    let http_client =
        Arc::new(ReqwestClient::user_agent(&user_agent).expect("could not start HTTP client"));

    let app = gpui_platform::headless().with_http_client(http_client);

    app.run(move |cx: &mut App| {
        let app_state = headless::init(cx);

        if args.list_models {
            cx.spawn(async move |cx| {
                let result = async {
                    app_state.client.sign_in_non_interactive(&cx).await?;
                    let started_at = Instant::now();
                    loop {
                        let models = cx.update(|cx| {
                            LanguageModelRegistry::global(cx)
                                .read(cx)
                                .provider(&language_model::ZED_CLOUD_PROVIDER_ID)
                                .map(|provider| provider.provided_models(cx))
                                .unwrap_or_default()
                        });
                        if !models.is_empty() {
                            let models = models.iter().map(|model| json!({
                                "slug": format!("zed.dev/{}", model.id().0),
                                "name": model.name().0.to_string(),
                            })).collect::<Vec<_>>();
                            println!("{}", serde_json::to_string(&models)?);
                            return Ok::<_, anyhow::Error>(());
                        }
                        if started_at.elapsed() >= MODEL_DISCOVERY_TIMEOUT {
                            anyhow::bail!("Timed out discovering Zed hosted models");
                        }
                        cx.background_executor().timer(MODEL_DISCOVERY_POLL_INTERVAL).await;
                    }
                }.await;
                let exit_code = match result {
                    Ok(()) => 0,
                    Err(error) => {
                        eprintln!("Failed to list Zed models: {error:#}");
                        1
                    }
                };
                cx.update(|cx| cx.quit());
                std::process::exit(exit_code);
            }).detach();
            return;
        }

        // Apply strict confirmation approval policy by default (Requirement 7)
        SettingsStore::update_global(cx, |store, cx| {
            let settings = r#"{
                "agent": {
                    "tool_permissions": {
                        "default": "confirm"
                    }
                },
                "autosave": "off",
                "format_on_save": "off"
            }"#;
            store.set_user_settings(settings, cx).result()
        })
        .expect("failed to configure default settings");

        if let Some(model_str) = &args.model {
            if let Some((provider, model)) = model_str.split_once('/') {
                let settings = format!(
                    r#"{{
                        "agent": {{
                            "default_model": {{
                                "provider": "{}",
                                "model": "{}"
                            }}
                        }}
                    }}"#,
                    provider, model
                );
                SettingsStore::update_global(cx, |store, cx| {
                    store.set_user_settings(&settings, cx).result()
                })
                .expect("failed to configure model setting");
            }
        }

        if let Some(worker_mode) = &args.worker_mode {
            let tools = if worker_mode == "review" {
                vec!["diagnostics", "find_path", "find_references", "go_to_definition", "list_directory", "read_file", "grep", "fetch", "search_web"]
            } else {
                vec!["copy_path", "create_directory", "delete_path", "diagnostics", "apply_code_action", "edit_file", "write_file", "fetch", "find_path", "find_references", "get_code_actions", "go_to_definition", "list_directory", "move_path", "rename_symbol", "read_file", "grep", "terminal", "search_web"]
            };
            let enabled_tools = tools.into_iter().map(|name| (name, true)).collect::<std::collections::BTreeMap<_, _>>();
            let settings = json!({ "agent": {
                "default_profile": "t3-worker",
                "profiles": { "t3-worker": { "name": "T3 worker", "enable_all_context_servers": false, "tools": enabled_tools } },
                "tool_permissions": { "default": "confirm" }
            }, "autosave": "off", "format_on_save": "off" }).to_string();
            SettingsStore::update_global(cx, |store, cx| store.set_user_settings(&settings, cx).result())
                .expect("failed to configure T3 worker profile");
        }

        let state = Rc::new(RefCell::new(ServerState::new(
            app_state,
            args.worktree.clone(),
            args.model.clone(),
        )));

        state.borrow_mut().worker_mode = args.worker_mode.clone();

        let (stdin_tx, mut stdin_rx) = mpsc::unbounded::<String>();
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            use std::io::BufRead;
            for line in stdin.lock().lines() {
                match line {
                    Ok(l) => {
                        if stdin_tx.unbounded_send(l).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        cx.spawn(async move |mut cx| {
            let state = state.clone();
                while let Some(line) = stdin_rx.next().await {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    let value: Value = match serde_json::from_str(trimmed) {
                        Ok(v) => v,
                        Err(err) => {
                            eprintln!("Failed to parse JSON line: {err}");
                            continue;
                        }
                    };

                    let state_clone = state.clone();
                    process_message(value, state_clone, &mut cx).await;
                }
                let _ = cx.update(|cx| cx.quit());
                std::process::exit(0);
        })
        .detach();
    });
}

fn client_tool_call_id(session_id: &str, client_session_id: &str, id: &acp::ToolCallId) -> String {
    if session_id == client_session_id {
        id.to_string()
    } else {
        format!("{session_id}:{}", id)
    }
}

fn subscribe_session(
    state: &Rc<RefCell<ServerState>>,
    session_id: &str,
    thread: &Entity<AcpThread>,
    cx: &mut App,
) {
    let state = state.clone();
    let session_id = session_id.to_string();
    let subscription: Subscription = cx.subscribe(thread, move |thread, event, cx| {
        state
            .borrow_mut()
            .on_thread_event(&session_id, &thread, event, cx);
        if let AcpThreadEvent::SubagentSpawned(child_id) = event {
            attach_subagent(&state, &session_id, child_id, cx);
        }
    });
    subscription.detach();
}

fn attach_subagent(
    state: &Rc<RefCell<ServerState>>,
    session_id: &str,
    child_id: &acp::SessionId,
    cx: &mut App,
) {
    let Some((connection, project, workdir, client_session_id)) =
        state.borrow().sessions.get(session_id).map(|parent| {
            (
                parent.connection.clone(),
                parent.project.clone(),
                parent.workdir.clone(),
                parent.client_session_id.clone(),
            )
        })
    else {
        return;
    };
    if state.borrow().sessions.contains_key(&child_id.to_string()) {
        return;
    }
    let child_task = connection.clone().load_session(
        child_id.clone(),
        project.clone(),
        PathList::new(&[&workdir]),
        None,
        cx,
    );
    let state = state.clone();
    cx.spawn(async move |cx| {
        let child = match child_task.await {
            Ok(child) => child,
            Err(error) => {
                eprintln!("Failed to attach subagent session: {error:#}");
                return;
            }
        };
        cx.update(|cx| {
            let child_id = child.read(cx).session_id().to_string();
            state.borrow_mut().sessions.insert(
                child_id.clone(),
                SessionEntry {
                    session_id: child_id.clone(),
                    client_session_id,
                    project,
                    connection,
                    acp_thread: child.clone(),
                    workdir,
                    emitted_chunk_lengths: HashMap::new(),
                    emitted_tool_calls: HashMap::new(),
                },
            );
            subscribe_session(&state, &child_id, &child, cx);
            // A child may already be waiting before its subscription attaches.
            state
                .borrow_mut()
                .on_thread_event(&child_id, &child, &AcpThreadEvent::NewEntry, cx);
            let waiting = child
                .read(cx)
                .entries()
                .iter()
                .filter_map(|entry| match entry {
                    AgentThreadEntry::ToolCall(call)
                        if matches!(call.status, ToolCallStatus::WaitingForConfirmation { .. }) =>
                    {
                        Some(call.id.clone())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            for id in waiting {
                state.borrow_mut().on_thread_event(
                    &child_id,
                    &child,
                    &AcpThreadEvent::ToolAuthorizationRequested(id),
                    cx,
                );
            }
            let descendants = child
                .read(cx)
                .entries()
                .iter()
                .filter_map(|entry| match entry {
                    AgentThreadEntry::ToolCall(call) => call
                        .subagent_session_info
                        .as_ref()
                        .map(|info| info.session_id.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            for descendant in descendants {
                attach_subagent(&state, &child_id, &descendant, cx);
            }
        });
    })
    .detach();
}

async fn process_message(value: Value, state: Rc<RefCell<ServerState>>, cx: &mut AsyncApp) {
    let method = value.get("method").and_then(|m| m.as_str());
    let id = value.get("id");

    if let Some(method_name) = method {
        if let Some(req_id) = id {
            // Request
            match method_name {
                "initialize" => {
                    send_response(
                        req_id,
                        json!({
                            "protocolVersion": 1,
                            "_meta": { "t3WorkerPolicy": state.borrow().worker_mode.clone(), "t3ThreadTools": true },
                            "serverInfo": {
                                "name": "zed-acp-server",
                                "version": "0.1.0"
                            },
                            "capabilities": {
                                "elicitation": { "form": true },
                                "tools": true
                            }
                        }),
                    );
                }
                "session/new" => {
                    let params = value.get("params");
                    let workdir = params
                        .and_then(|p| p.get("cwd"))
                        .and_then(|c| c.as_str())
                        .map(PathBuf::from)
                        .or_else(|| state.borrow().default_worktree.clone())
                        .unwrap_or_else(|| {
                            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
                        });

                    // Only T3's scoped HTTP server is forwarded. Worker profiles allow
                    // coordination tools explicitly; other MCP servers stay disabled.
                    if let Some(server) = params
                        .and_then(|p| p.get("mcpServers"))
                        .and_then(Value::as_array)
                        .and_then(|servers| {
                            servers.iter().find(|server| {
                                server.get("name").and_then(Value::as_str) == Some("t3-code")
                                    && server.get("type").and_then(Value::as_str) == Some("http")
                            })
                        })
                    {
                        if let Some(url) = server.get("url").and_then(Value::as_str) {
                            let headers = server
                                .get("headers")
                                .and_then(Value::as_array)
                                .into_iter()
                                .flatten()
                                .filter_map(|header| {
                                    Some((
                                        header.get("name")?.as_str()?.to_owned(),
                                        header.get("value")?.as_str()?.to_owned(),
                                    ))
                                })
                                .collect::<HashMap<_, _>>();
                            let worker_mode = state.borrow().worker_mode.clone();
                            let result = cx.update(|cx| SettingsStore::update_global(cx, |store, cx| {
                                let mut settings = serde_json::to_value(store.raw_user_settings().cloned().unwrap_or_default())?;
                                settings["context_servers"]["t3-code"] = json!({ "url": url, "headers": headers, "enabled": true, "timeout": 30 });
                                if let Some(mode) = &worker_mode {
                                    let names: &[&str] = if mode == "parent" {
                                        &["start_orchestration_layer", "spawn_child", "assign_child", "report_to_parent", "wait_for_children", "read_orchestration_layer", "control_orchestration_layer", "refresh_coordination_policy", "list_thread_models", "create_thread", "read_thread", "send_message_to_thread", "interrupt_thread"]
                                    } else { &["report_to_parent", "read_orchestration_layer", "read_thread"] };
                                    let tools = names.iter().map(|name| (name.to_string(), true)).collect::<std::collections::BTreeMap<_, _>>();
                                    settings["agent"]["profiles"]["t3-worker"]["context_servers"]["t3-code"] = json!({ "tools": tools });
                                    if mode != "parent" {
                                        let permissions = names.iter().map(|name| {
                                            (format!("mcp:t3-code:{name}"), json!({ "default": "allow" }))
                                        }).collect::<serde_json::Map<_, _>>();
                                        settings["agent"]["tool_permissions"]["tools"] = json!(permissions);
                                    }
                                }
                                store.set_user_settings(&settings.to_string(), cx).result()
                            }));
                            if let Err(error) = result {
                                send_error(
                                    req_id,
                                    -32000,
                                    &format!("Failed to configure T3 thread tools: {error:#}"),
                                );
                                return;
                            }
                        }
                    }

                    let (app_state, default_model, model_ready) = {
                        let state = state.borrow();
                        (
                            state.app_state.clone(),
                            state.default_model.clone(),
                            state.model_ready,
                        )
                    };
                    if !model_ready {
                        if let Err(error) =
                            ensure_model_ready(&app_state, default_model.as_deref(), cx).await
                        {
                            send_error(
                                req_id,
                                -32000,
                                &format!("Failed to prepare model: {error:#}"),
                            );
                            return;
                        }
                        state.borrow_mut().model_ready = true;
                    }

                    let project = cx.update(|cx| {
                        Project::local(
                            app_state.client.clone(),
                            app_state.node_runtime.clone(),
                            app_state.user_store.clone(),
                            app_state.languages.clone(),
                            app_state.fs.clone(),
                            None,
                            project::LocalProjectFlags {
                                init_worktree_trust: false,
                                ..Default::default()
                            },
                            cx,
                        )
                    });

                    let worktree_task =
                        project.update(cx, |p, cx| p.create_worktree(&workdir, true, cx));
                    let worktree = match worktree_task.await {
                        Ok(w) => w,
                        Err(e) => {
                            send_error(req_id, -32000, &format!("Failed to create worktree: {e}"));
                            return;
                        }
                    };

                    let scan_res = worktree.update(cx, |tree, _cx| {
                        tree.as_local()
                            .context("expected local worktree")
                            .map(|local| local.scan_complete())
                    });
                    if let Ok(scan_future) = scan_res {
                        let _ = scan_future.await;
                    }

                    let agent = cx.update(|cx| {
                        let thread_store = cx.new(|cx| ThreadStore::new(cx));
                        NativeAgent::new(thread_store, Templates::new(), app_state.fs.clone(), cx)
                    });

                    let connection = Rc::new(NativeAgentConnection(agent));
                    let workdir_clone = workdir.clone();
                    let conn_clone = connection.clone();
                    let project_clone = project.clone();

                    let acp_thread = match cx
                        .update(|cx| {
                            conn_clone.new_session(
                                project_clone,
                                PathList::new(&[&workdir_clone]),
                                cx,
                            )
                        })
                        .await
                    {
                        Ok(t) => t,
                        Err(e) => {
                            send_error(
                                req_id,
                                -32000,
                                &format!("Failed to create ACP session: {e}"),
                            );
                            return;
                        }
                    };

                    let session_id = acp_thread.read_with(cx, |t, _| t.session_id().to_string());

                    cx.update(|cx| subscribe_session(&state, &session_id, &acp_thread, cx));

                    state.borrow_mut().sessions.insert(
                        session_id.clone(),
                        SessionEntry {
                            session_id: session_id.clone(),
                            client_session_id: session_id.clone(),
                            project,
                            connection,
                            acp_thread,
                            workdir,
                            emitted_chunk_lengths: HashMap::new(),
                            emitted_tool_calls: HashMap::new(),
                        },
                    );

                    send_response(
                        req_id,
                        json!({
                            "sessionId": session_id
                        }),
                    );
                }
                "session/prompt" => {
                    let params = match value.get("params") {
                        Some(p) => p,
                        None => {
                            send_error(req_id, -32602, "Missing params");
                            return;
                        }
                    };
                    let session_id = match params.get("sessionId").and_then(|s| s.as_str()) {
                        Some(s) => s,
                        None => {
                            send_error(req_id, -32602, "Missing sessionId");
                            return;
                        }
                    };

                    let session_thread = {
                        let state_borrow = state.borrow();
                        state_borrow
                            .sessions
                            .get(session_id)
                            .map(|s| s.acp_thread.clone())
                    };

                    let acp_thread = match session_thread {
                        Some(t) => t,
                        None => {
                            send_error(req_id, -32001, "Session not found");
                            return;
                        }
                    };

                    let mut content_blocks = Vec::new();
                    if let Some(prompt_val) = params.get("prompt") {
                        if let Some(prompt_arr) = prompt_val.as_array() {
                            for item in prompt_arr {
                                if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                                    content_blocks.push(acp::ContentBlock::Text(
                                        acp::TextContent::new(text.to_string()),
                                    ));
                                }
                            }
                        } else if let Some(text) = prompt_val.as_str() {
                            content_blocks.push(acp::ContentBlock::Text(acp::TextContent::new(
                                text.to_string(),
                            )));
                        }
                    }

                    if content_blocks.is_empty() {
                        content_blocks.push(acp::ContentBlock::Text(acp::TextContent::new(
                            String::new(),
                        )));
                    }

                    let send_future =
                        acp_thread.update(cx, |thread, cx| thread.send(content_blocks, cx));

                    let req_id_clone = req_id.clone();
                    cx.spawn(async move |_cx| {
                        let outcome = send_future.await;
                        match prompt_result(outcome) {
                            Ok(result) => send_response(&req_id_clone, result),
                            Err(error) => send_error(
                                &req_id_clone,
                                -32000,
                                &format!("Prompt failed: {error:#}"),
                            ),
                        }
                    })
                    .detach();
                }
                "session/cancel" => {
                    let params = value.get("params");
                    let session_id = params
                        .and_then(|p| p.get("sessionId"))
                        .and_then(|s| s.as_str());

                    if let Some(session_id) = session_id {
                        let thread_opt = state
                            .borrow()
                            .sessions
                            .get(session_id)
                            .map(|s| s.acp_thread.clone());
                        if let Some(thread) = thread_opt {
                            let _ = thread.update(cx, |t, cx| t.cancel(cx));
                        }
                    }
                    send_response(req_id, json!({}));
                }
                "zed/invoke_skill" => {
                    let params = match value.get("params") {
                        Some(p) => p,
                        None => {
                            send_error(req_id, -32602, "Missing params");
                            return;
                        }
                    };
                    let session_id = match params.get("sessionId").and_then(|s| s.as_str()) {
                        Some(s) => s,
                        None => {
                            send_error(req_id, -32602, "Missing sessionId");
                            return;
                        }
                    };
                    let skill_name = match params.get("skillName").and_then(|s| s.as_str()) {
                        Some(s) => s,
                        None => {
                            send_error(req_id, -32602, "Missing skillName");
                            return;
                        }
                    };
                    let prompt_text = params
                        .get("prompt")
                        .and_then(|p| p.as_str())
                        .unwrap_or("")
                        .trim();

                    let command_text = if prompt_text.is_empty() {
                        format!("/{}", skill_name)
                    } else {
                        format!("/{} {}", skill_name, prompt_text)
                    };

                    let session_thread = {
                        let state_borrow = state.borrow();
                        state_borrow
                            .sessions
                            .get(session_id)
                            .map(|s| s.acp_thread.clone())
                    };

                    let acp_thread = match session_thread {
                        Some(t) => t,
                        None => {
                            send_error(req_id, -32001, "Session not found");
                            return;
                        }
                    };

                    let content_blocks =
                        vec![acp::ContentBlock::Text(acp::TextContent::new(command_text))];

                    let send_future =
                        acp_thread.update(cx, |thread, cx| thread.send(content_blocks, cx));

                    let req_id_clone = req_id.clone();
                    cx.spawn(async move |_cx| {
                        let outcome = send_future.await;
                        match prompt_result(outcome) {
                            Ok(result) => send_response(&req_id_clone, result),
                            Err(error) => send_error(
                                &req_id_clone,
                                -32000,
                                &format!("Prompt failed: {error:#}"),
                            ),
                        }
                    })
                    .detach();
                }
                other => {
                    send_error(req_id, -32601, &format!("Method not found: {other}"));
                }
            }
        } else {
            // Notification
            if method_name == "session/cancel" {
                if let Some(session_id) = value
                    .get("params")
                    .and_then(|p| p.get("sessionId"))
                    .and_then(|s| s.as_str())
                {
                    let thread_opt = state
                        .borrow()
                        .sessions
                        .get(session_id)
                        .map(|s| s.acp_thread.clone());
                    if let Some(thread) = thread_opt {
                        let _ = thread.update(cx, |t, cx| t.cancel(cx));
                    }
                }
            }
        }
    } else if let Some(resp_id) = id {
        // Inbound response to an outbound request (e.g. permission or elicitation response)
        let id_str = match resp_id.as_str() {
            Some(s) => s.to_string(),
            None => resp_id.to_string(),
        };

        let pending_perm = state.borrow_mut().pending_permissions.remove(&id_str);
        if let Some(pending) = pending_perm {
            let thread_opt = state
                .borrow()
                .sessions
                .get(&pending.session_id)
                .map(|s| s.acp_thread.clone());
            if let Some(thread) = thread_opt {
                let outcome_obj = value.get("result").and_then(|r| r.get("outcome"));
                let outcome_type = outcome_obj
                    .and_then(|o| o.get("outcome"))
                    .and_then(|s| s.as_str());
                let selected_option_id = outcome_obj
                    .and_then(|o| o.get("optionId"))
                    .and_then(|s| s.as_str());

                if outcome_type == Some("selected") {
                    if let Some(opt_id_str) = selected_option_id {
                        let option_id = acp::PermissionOptionId::new(opt_id_str);
                        let option_kind = pending
                            .options
                            .iter()
                            .find(|o| o.option_id == option_id)
                            .map(|o| o.kind)
                            .unwrap_or_else(|| {
                                if opt_id_str.contains("allow") {
                                    acp::PermissionOptionKind::AllowOnce
                                } else {
                                    acp::PermissionOptionKind::RejectOnce
                                }
                            });

                        let outcome = SelectedPermissionOutcome::new(option_id, option_kind);
                        let _ = thread.update(cx, |t, cx| {
                            t.authorize_tool_call(pending.tool_call_id, outcome, cx);
                        });
                    } else {
                        let _ = thread.update(cx, |t, cx| {
                            t.cancel_tool_call_authorization(&pending.tool_call_id, cx);
                        });
                    }
                } else {
                    let _ = thread.update(cx, |t, cx| {
                        t.cancel_tool_call_authorization(&pending.tool_call_id, cx);
                    });
                }
            }
            return;
        }

        let pending_elicit = state.borrow_mut().pending_elicitations.remove(&id_str);
        if let Some(pending) = pending_elicit {
            let thread_opt = state
                .borrow()
                .sessions
                .get(&pending.session_id)
                .map(|s| s.acp_thread.clone());
            if let Some(thread) = thread_opt {
                if let Some(result_val) = value.get("result") {
                    if let Ok(response) =
                        serde_json::from_value::<acp::CreateElicitationResponse>(result_val.clone())
                    {
                        let _ = thread.update(cx, |t, cx| {
                            t.respond_to_elicitation(&pending.entry_id, response, cx);
                        });
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod prompt_result_tests {
    use super::*;

    #[test]
    fn failed_prompt_is_not_reported_as_success() {
        let result = prompt_result(Err(anyhow::anyhow!("unsupported thinking mode")));
        assert!(result.is_err());
    }

    #[test]
    fn successful_prompt_preserves_stop_reason() {
        let result = prompt_result(Ok(Some(acp::PromptResponse::new(
            acp::StopReason::MaxTokens,
        ))))
        .unwrap();
        assert_eq!(result, json!({ "stopReason": "max_tokens" }));
    }
    async fn test_session(
        cx: &mut gpui::TestAppContext,
    ) -> (Rc<RefCell<ServerState>>, Entity<AcpThread>) {
        RPC_MESSAGES.with(|messages| messages.borrow_mut().clear());
        let app_state = headless::tests::init(cx).await;
        let project = cx.update(|cx| {
            Project::local(
                app_state.client.clone(),
                app_state.node_runtime.clone(),
                app_state.user_store.clone(),
                app_state.languages.clone(),
                app_state.fs.clone(),
                None,
                project::LocalProjectFlags {
                    init_worktree_trust: false,
                    ..Default::default()
                },
                cx,
            )
        });
        let connection = cx.update(|cx| {
            let store = cx.new(|cx| ThreadStore::new(cx));
            Rc::new(NativeAgentConnection(NativeAgent::new(
                store,
                Templates::new(),
                app_state.fs.clone(),
                cx,
            )))
        });
        let parent = cx
            .update(|cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[] as &[PathBuf]),
                    cx,
                )
            })
            .await
            .unwrap();
        let parent_id = parent.read_with(cx, |t, _| t.session_id().to_string());
        let state = Rc::new(RefCell::new(ServerState::new(app_state, None, None)));
        cx.update(|cx| {
            state.borrow_mut().sessions.insert(
                parent_id.clone(),
                SessionEntry {
                    session_id: parent_id.clone(),
                    client_session_id: parent_id.clone(),
                    project,
                    connection,
                    acp_thread: parent.clone(),
                    workdir: PathBuf::from("/tmp"),
                    emitted_chunk_lengths: HashMap::new(),
                    emitted_tool_calls: HashMap::new(),
                },
            );
            subscribe_session(&state, &parent_id, &parent, cx);
        });

        (state, parent)
    }

    #[gpui::test]
    async fn spawned_agent_permissions_reach_the_client(cx: &mut gpui::TestAppContext) {
        child_permission_roundtrip(cx, false, false).await;
    }

    #[gpui::test]
    async fn permissions_pending_before_subagent_attachment_reach_the_client(
        cx: &mut gpui::TestAppContext,
    ) {
        child_permission_roundtrip(cx, true, false).await;
    }

    #[gpui::test]
    async fn nested_subagent_permissions_pending_before_attachment_reach_the_client(
        cx: &mut gpui::TestAppContext,
    ) {
        child_permission_roundtrip(cx, true, true).await;
    }

    async fn child_permission_roundtrip(
        cx: &mut gpui::TestAppContext,
        already_waiting: bool,
        nested: bool,
    ) {
        let (state, parent) = test_session(cx).await;
        let parent_id = parent.read_with(cx, |t, _| t.session_id().to_string());
        let (connection, project) = state
            .borrow()
            .sessions
            .get(&parent_id)
            .map(|s| (s.connection.clone(), s.project.clone()))
            .unwrap();
        let child = cx
            .update(|cx| {
                connection.clone().new_session(
                    project.clone(),
                    PathList::new(&[] as &[PathBuf]),
                    cx,
                )
            })
            .await
            .unwrap();
        let child_id = child.read_with(cx, |t, _| t.session_id().clone());
        let child = if nested {
            let grandchild = cx
                .update(|cx| {
                    connection.clone().new_session(
                        project.clone(),
                        PathList::new(&[] as &[PathBuf]),
                        cx,
                    )
                })
                .await
                .unwrap();
            let grandchild_id = grandchild.read_with(cx, |t, _| t.session_id().clone());
            let mut spawn = acp::ToolCall::new(acp::ToolCallId::new("spawn-child"), "Spawn agent");
            spawn.meta = Some(acp::Meta::from_iter([(
                acp_thread::SUBAGENT_SESSION_INFO_META_KEY.into(),
                json!({ "session_id": grandchild_id, "message_start_index": 0, "message_end_index": null }),
            )]));
            child.update(cx, |t, cx| {
                t.upsert_tool_call(spawn, cx).unwrap();
                t.subagent_spawned(grandchild_id, cx);
            });
            grandchild
        } else {
            child
        };
        if !already_waiting {
            parent.update(cx, |t, cx| t.subagent_spawned(child_id.clone(), cx));
            cx.run_until_parked();
        }
        let authorization = child
            .update(cx, |t, cx| {
                t.request_tool_call_authorization(
                    acp::ToolCallUpdate::new(
                        acp::ToolCallId::new("child-search"),
                        acp::ToolCallUpdateFields::new().title("Search the web"),
                    ),
                    PermissionOptions::Flat(vec![acp::PermissionOption::new(
                        acp::PermissionOptionId::new("allow"),
                        "Allow",
                        acp::PermissionOptionKind::AllowOnce,
                    )]),
                    acp_thread::AuthorizationKind::PermissionGrant,
                    cx,
                )
            })
            .unwrap();
        if already_waiting {
            parent.update(cx, |t, cx| t.subagent_spawned(child_id.clone(), cx));
        }
        cx.run_until_parked();
        assert_eq!(
            state.borrow().pending_permissions.len(),
            1,
            "a child tool permission must be forwarded instead of silently stalling"
        );
        let request = RPC_MESSAGES.with(|messages| {
            messages
                .borrow()
                .iter()
                .find(|m| m["method"] == "session/request_permission")
                .cloned()
                .unwrap()
        });
        assert_eq!(request["params"]["sessionId"], parent_id);
        process_message(
            json!({ "jsonrpc": "2.0", "id": request["id"],
                "result": { "outcome": { "outcome": "selected", "optionId": "allow" } }
            }),
            state.clone(),
            &mut cx.to_async(),
        )
        .await;
        assert!(matches!(
            authorization.await,
            acp_thread::RequestPermissionOutcome::Selected(_)
        ));
        assert!(state.borrow().pending_permissions.is_empty());
    }

    #[gpui::test]
    async fn streaming_tool_details_are_not_frozen_at_the_first_fragment(
        cx: &mut gpui::TestAppContext,
    ) {
        let (state, parent) = test_session(cx).await;
        let id = acp::ToolCallId::new("read-file");
        parent
            .update(cx, |t, cx| {
                t.upsert_tool_call(
                    acp::ToolCall::new(id.clone(), "Read file")
                        .raw_input(json!({ "path": "work" })),
                    cx,
                )
            })
            .unwrap();
        cx.run_until_parked();
        parent
            .update(cx, |t, cx| {
                t.update_tool_call(
                    acp::ToolCallUpdate::new(
                        id.clone(),
                        acp::ToolCallUpdateFields::new()
                            .raw_input(json!({ "path": "work_utilities" })),
                    ),
                    cx,
                )
            })
            .unwrap();
        cx.run_until_parked();
        let input = RPC_MESSAGES.with(|messages| {
            messages
                .borrow()
                .iter()
                .rev()
                .find(|m| m["params"]["update"]["toolCallId"] == "read-file")
                .unwrap()["params"]["update"]["rawInput"]
                .clone()
        });
        assert_eq!(input, json!({ "path": "work_utilities" }));
        parent
            .update(cx, |t, cx| {
                t.update_tool_call(
                    acp::ToolCallUpdate::new(
                        id.clone(),
                        acp::ToolCallUpdateFields::new()
                            .title("Read search.md")
                            .raw_input(
                                json!({ "path": "work_utilities/search.md", "start_line": 10 }),
                            )
                            .status(acp::ToolCallStatus::Completed)
                            .raw_output(json!({ "Text": "result" })),
                    ),
                    cx,
                )
            })
            .unwrap();
        cx.run_until_parked();
        let update = RPC_MESSAGES.with(|messages| {
            messages
                .borrow()
                .iter()
                .rev()
                .find(|m| m["params"]["update"]["sessionUpdate"] == "tool_call_update")
                .cloned()
                .unwrap()
        });
        assert_eq!(
            update["params"]["update"]["rawInput"],
            json!({ "path": "work_utilities/search.md", "start_line": 10 })
        );
        assert_eq!(update["params"]["update"]["title"], "Read search.md");
        let count = RPC_MESSAGES.with(|messages| messages.borrow().len());
        cx.update(|cx| {
            state.borrow_mut().on_thread_event(
                &parent.read(cx).session_id().to_string(),
                &parent,
                &AcpThreadEvent::NewEntry,
                cx,
            )
        });
        assert_eq!(
            RPC_MESSAGES.with(|messages| messages.borrow().len()),
            count,
            "unrelated entry updates must not resend completed tools"
        );
    }
}
