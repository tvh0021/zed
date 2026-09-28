//! Headless Zed ACP (Agent Client Protocol) Server.
//!
//! Provides a headless runtime for Zed agents speaking ACP over standard input/output.
//! T3 Code (or any ACP client) can spawn this binary as a background process to run
//! Zed NativeAgent, execute tools, request permissions, and invoke skills.

mod headless;

use std::cell::RefCell;
use std::collections::HashMap;
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

    /// Disable native delegation, with an optional read-only tool profile.
    #[arg(long, value_parser = ["parent", "review", "edit"])]
    worker_mode: Option<String>,
}

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_id(prefix: &str) -> String {
    let id = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}_{}", prefix, id)
}

fn send_rpc(msg: &Value) {
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
    project: Entity<Project>,
    connection: Rc<NativeAgentConnection>,
    acp_thread: Entity<AcpThread>,
    workdir: PathBuf,
    // (entry_index, chunk_index) -> emitted text byte length
    emitted_chunk_lengths: HashMap<(usize, usize), usize>,
    // tool_call_id -> known status
    emitted_tool_calls: HashMap<acp::ToolCallId, String>,
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
                    match entry {
                        AgentThreadEntry::AssistantMessage(msg) => {
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
                                                    "sessionId": session_id,
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
                                                    "sessionId": session_id,
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

                            let title = call.label.read(cx).source().to_string();

                            let prev_status = session.emitted_tool_calls.get(&call.id);
                            if prev_status.is_none() {
                                session
                                    .emitted_tool_calls
                                    .insert(call.id.clone(), status_str.to_string());
                                send_notification(
                                    "session/update",
                                    json!({
                                        "sessionId": session_id,
                                        "update": {
                                            "sessionUpdate": "tool_call",
                                            "toolCallId": call.id.to_string(),
                                            "title": title,
                                            "kind": kind_str,
                                            "status": status_str,
                                            "rawInput": call.raw_input
                                        }
                                    }),
                                );
                            } else if prev_status.map(|s| s.as_str()) != Some(status_str)
                                || call.raw_output.is_some()
                            {
                                session
                                    .emitted_tool_calls
                                    .insert(call.id.clone(), status_str.to_string());
                                send_notification(
                                    "session/update",
                                    json!({
                                        "sessionId": session_id,
                                        "update": {
                                            "sessionUpdate": "tool_call_update",
                                            "toolCallId": call.id.to_string(),
                                            "status": status_str,
                                            "rawOutput": call.raw_output
                                        }
                                    }),
                                );
                            }
                        }
                        _ => {}
                    }
                }
            }
            AcpThreadEvent::ToolAuthorizationRequested(tool_call_id) => {
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
                                "sessionId": session_id,
                                "toolCall": {
                                    "toolCallId": tool_call_id.to_string(),
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
                        "sessionId": session_id,
                        "elicitationId": entry_id.0.to_string(),
                    }
                }));
            }
            AcpThreadEvent::AvailableCommandsUpdated(commands) => {
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
                        "sessionId": session_id,
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
                                        &["start_thread_workflow", "spawn_child", "assign_child", "report_to_parent", "wait_for_children", "read_thread_workflow", "control_thread_workflow", "refresh_coordination_policy", "list_thread_models", "create_thread", "read_thread", "send_message_to_thread", "interrupt_thread"]
                                    } else { &["report_to_parent", "read_thread_workflow", "read_thread"] };
                                    let tools = names.iter().map(|name| (name.to_string(), true)).collect::<std::collections::BTreeMap<_, _>>();
                                    settings["agent"]["profiles"]["t3-worker"]["context_servers"]["t3-code"] = json!({ "tools": tools });
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

                    let sess_id_str = session_id.clone();
                    let state_for_sub = state.clone();

                    let subscription: Subscription = cx.update(|cx| {
                        cx.subscribe(&acp_thread, move |thread, event, cx| {
                            state_for_sub.borrow_mut().on_thread_event(
                                &sess_id_str,
                                &thread,
                                event,
                                cx,
                            );
                        })
                    });
                    subscription.detach();

                    state.borrow_mut().sessions.insert(
                        session_id.clone(),
                        SessionEntry {
                            session_id: session_id.clone(),
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
                        let stop_reason = match outcome {
                            Ok(Some(resp)) => match resp.stop_reason {
                                acp::StopReason::EndTurn => "end_turn",
                                acp::StopReason::MaxTokens => "max_tokens",
                                acp::StopReason::Cancelled => "cancelled",
                                _ => "end_turn",
                            },
                            Ok(None) => "end_turn",
                            Err(err) => {
                                eprintln!("Prompt turn finished with error: {err:?}");
                                "end_turn"
                            }
                        };
                        send_response(
                            &req_id_clone,
                            json!({
                                "stopReason": stop_reason
                            }),
                        );
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
                        let stop_reason = match outcome {
                            Ok(Some(resp)) => match resp.stop_reason {
                                acp::StopReason::EndTurn => "end_turn",
                                acp::StopReason::MaxTokens => "max_tokens",
                                acp::StopReason::Cancelled => "cancelled",
                                _ => "end_turn",
                            },
                            Ok(None) => "end_turn",
                            Err(err) => {
                                eprintln!("Skill invocation finished with error: {err:?}");
                                "end_turn"
                            }
                        };
                        send_response(
                            &req_id_clone,
                            json!({
                                "stopReason": stop_reason
                            }),
                        );
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
