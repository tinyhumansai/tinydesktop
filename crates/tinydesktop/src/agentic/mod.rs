//! Native Jev-backed observation, intent resolution, and goal execution.

mod policy;
mod screen;
mod task;
mod verify;

#[cfg(test)]
mod test;

use std::fmt::Write as _;
use std::{
    collections::HashMap,
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use serde_json::json;
use tinydesktop_bus::{
    DeliveryDisposition, DesktopError, DesktopResponse, GoalContinuation, JevConfig,
    JevConfiguration, JevDecision, JevDecisionKind, JevMetrics, JevObservation, JevOperation,
    JevProvider, JevRunResult, JevStopReason, JevTarget, JevTurn, RefRequest, ResolveIntentRequest,
    RunGoalRequest, ScrollRequest, SetValueRequest, WaitRequest,
};
use tinyjevclient::{
    Client, ClientConfig, Error as JevError, EvaluationFailure, EvaluationRequest, EvaluationResult,
};

use crate::Desktop;
use policy::{
    action_space, choice, deterministic_destructive, exact_named_match, gate_with_evidence, noul,
    parse_operation, playing_goal_satisfied, positional_match, shortlist, target,
};
use screen::{Candidate, Screen, fingerprint, observe};
use task::run_goal_fresh;
use verify::{exact_label, satisfied, verify};

/// Configured Jev transport and non-secret policy metadata.
#[derive(Clone)]
pub(crate) struct JevRuntime {
    client: Arc<dyn Evaluator>,
    configuration: JevConfiguration,
    pending: Arc<Mutex<HashMap<String, PendingRun>>>,
}

#[derive(Debug)]
struct PendingRun {
    created: Instant,
    started: Instant,
    request: RunGoalRequest,
    decision: JevDecision,
    screen: Screen,
    target: Candidate,
    turns: Vec<JevTurn>,
    history: Vec<String>,
    unchanged: u32,
    metrics: JevMetrics,
}

impl std::fmt::Debug for JevRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("JevRuntime")
            .field("client", &"[configured]")
            .field("configuration", &self.configuration)
            .field("pending", &"[redacted]")
            .finish()
    }
}

impl JevRuntime {
    pub(crate) fn configure(request: &JevConfig) -> Result<Self, Box<DesktopError>> {
        let mut config = match request.provider {
            JevProvider::TypeSafe => ClientConfig::new(request.api_key()),
            JevProvider::OpenRouter => ClientConfig::openrouter(request.api_key()),
            JevProvider::TinyHumansOpenRouter => {
                ClientConfig::tinyhumans_openrouter(request.api_key())
            }
            JevProvider::OpenJev => ClientConfig::new(request.api_key())
                .with_endpoint_url("https://api.openjev.sh/v1/systemone"),
        };
        if let Some(endpoint) = &request.endpoint_url {
            if !trusted_endpoint(request.provider, endpoint) {
                return Err(Box::new(DesktopError::new(
                    "JEV_INVALID_CONFIG",
                    "endpoint is not an approved Jev provider route",
                )));
            }
            config = config.with_endpoint_url(endpoint);
        }
        if let Some(timeout_ms) = request.timeout_ms {
            config.timeout = Duration::from_millis(timeout_ms);
        }
        if let Some(max_retries) = request.max_retries {
            config.retry.max_retries = max_retries;
        }
        if request.provider == JevProvider::TinyHumansOpenRouter
            && let Some(sdk_name) = request.sdk_name.as_deref()
        {
            config = config.with_sdk_name(sdk_name);
        }
        let client = Client::new(config).map_err(|error| config_error(&error))?;
        Ok(Self {
            client: Arc::new(client),
            configuration: JevConfiguration {
                provider: request.provider,
                model: request
                    .model
                    .clone()
                    .unwrap_or_else(|| default_model(request.provider).to_owned()),
                endpoint_url: request.endpoint_url.clone(),
            },
            pending: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

trait Evaluator: Send + Sync {
    fn evaluate<'a>(
        &'a self,
        request: &'a EvaluationRequest,
    ) -> Pin<
        Box<
            dyn Future<Output = std::result::Result<EvaluationResult, EvaluationFailure>>
                + Send
                + 'a,
        >,
    >;
}

impl Evaluator for Client {
    fn evaluate<'a>(
        &'a self,
        request: &'a EvaluationRequest,
    ) -> Pin<
        Box<
            dyn Future<Output = std::result::Result<EvaluationResult, EvaluationFailure>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(Client::evaluate(self, request))
    }
}

fn default_model(provider: JevProvider) -> &'static str {
    match provider {
        JevProvider::OpenJev => "openjev",
        _ => "jev-latest",
    }
}

fn trusted_endpoint(provider: JevProvider, endpoint: &str) -> bool {
    let approved = match provider {
        JevProvider::TypeSafe => "https://api.typesafe.ai/v1/systemone",
        JevProvider::OpenRouter => "https://openrouter.ai/api/alpha/decisions",
        JevProvider::TinyHumansOpenRouter => {
            "https://api.tinyhumans.ai/agent-integrations/openrouter/systemone"
        }
        JevProvider::OpenJev => "https://api.openjev.sh/v1/systemone",
    };
    if endpoint == approved {
        return true;
    }
    #[cfg(test)]
    return endpoint.starts_with("http://127.0.0.1:");
    #[cfg(not(test))]
    false
}

pub(crate) async fn resolve_intent(
    desktop: Desktop,
    runtime: JevRuntime,
    request: ResolveIntentRequest,
) -> DesktopResponse {
    resolve_intent_with(desktop, runtime, request).await
}

async fn resolve_intent_with<B: AgentBackend>(
    backend: B,
    runtime: JevRuntime,
    request: ResolveIntentRequest,
) -> DesktopResponse {
    let result = resolve(
        &backend,
        &runtime,
        &request.intent,
        &request.app,
        request.root.as_deref(),
        request.text.as_deref(),
        request.include_values,
        request.execute,
        &[],
        true,
    )
    .await;
    match result {
        Ok(outcome) => outcome
            .action_failure
            .unwrap_or_else(|| response("resolve-intent", &outcome.decision)),
        Err(error) => *error,
    }
}

pub(crate) async fn run_goal(
    desktop: Desktop,
    runtime: JevRuntime,
    request: RunGoalRequest,
) -> DesktopResponse {
    run_goal_with(desktop, runtime, request).await
}

async fn run_goal_with<B: AgentBackend>(
    backend: B,
    runtime: JevRuntime,
    request: RunGoalRequest,
) -> DesktopResponse {
    if let Some(continuation) = request.continuation.clone() {
        return continue_goal(backend, runtime, continuation).await;
    }
    run_goal_fresh(backend, runtime, request, Vec::new(), 0).await
}

fn selected_target(screen: &Screen, decision: &JevDecision) -> Option<Candidate> {
    decision
        .target
        .as_ref()
        .and_then(|target| {
            screen
                .candidates
                .iter()
                .find(|candidate| candidate.ref_id == target.ref_id)
        })
        .cloned()
}

fn within_scope(request: &RunGoalRequest, screen: &Screen) -> bool {
    request.app.eq_ignore_ascii_case(&screen.app)
        && request
            .window_id
            .as_deref()
            .is_none_or(|window_id| screen.window_id.as_deref() == Some(window_id))
        && request
            .window
            .as_deref()
            .is_none_or(|window| screen.window.as_deref() == Some(window))
}

fn mutates(operation: JevOperation) -> bool {
    matches!(
        operation,
        JevOperation::Click
            | JevOperation::TypeText
            | JevOperation::Check
            | JevOperation::Uncheck
            | JevOperation::Expand
            | JevOperation::Collapse
            | JevOperation::Scroll
    )
}

fn target_allowed(request: &RunGoalRequest, candidate: &Candidate) -> bool {
    request.allowed_targets.is_empty()
        || request
            .allowed_targets
            .iter()
            .any(|name| exact_label(candidate, name))
}

fn prepared_text(request: &RunGoalRequest, candidate: &Candidate) -> Option<String> {
    request
        .text_slots
        .iter()
        .find(|(name, _)| exact_label(candidate, name))
        .map(|(_, value)| value.clone())
}

fn queue_confirmation(runtime: &JevRuntime, run: PendingRun) -> DesktopResponse {
    let mut token = [0_u8; 16];
    if getrandom::fill(&mut token).is_err() {
        return internal_error("cannot create a confirmation handle");
    }
    let id = token
        .iter()
        .fold(String::with_capacity(32), |mut id, byte| {
            let _ = write!(id, "{byte:02x}");
            id
        });
    let Ok(mut pending) = runtime.pending.lock() else {
        return internal_error("confirmation state is unavailable");
    };
    pending.retain(|_, previous| previous.created.elapsed() < Duration::from_secs(600));
    if pending.len() >= 32 {
        return DesktopResponse::err(
            "run-goal",
            DesktopError::new(
                "CONFIRMATION_LIMIT",
                "too many desktop actions await confirmation",
            ),
        );
    }
    let response = run_response_with_id(
        JevStopReason::ConfirmationRequired,
        run.turns.clone(),
        Some(run.decision.clone()),
        run.metrics.clone(),
        Some(id.clone()),
    );
    pending.insert(id, run);
    response
}

fn record_turn(
    turns: &mut Vec<JevTurn>,
    history: &mut Vec<String>,
    decision: &JevDecision,
    changed: bool,
) {
    let turn = JevTurn {
        step: u32::try_from(turns.len())
            .unwrap_or(u32::MAX)
            .saturating_add(1),
        operation: decision.operation,
        target: decision.target.clone(),
        confidence: decision.confidence,
        ok: decision.executed,
        changed,
    };
    history.push(format!(
        "step {}: {:?} {} and changed={changed}",
        turn.step,
        turn.operation,
        turn.target
            .as_ref()
            .and_then(|target| target.name.as_deref())
            .unwrap_or("the selected element")
    ));
    turns.push(turn);
}

async fn continue_goal<B: AgentBackend>(
    backend: B,
    runtime: JevRuntime,
    continuation: GoalContinuation,
) -> DesktopResponse {
    let pending = match approved_pending(&runtime, &continuation) {
        Ok(pending) => pending,
        Err(reply) => return *reply,
    };
    let (fresh, delivered_unverified) = match execute_pending_action(&backend, &pending).await {
        Ok(executed) => executed,
        Err(reply) => return *reply,
    };
    let mut turns = pending.turns.clone();
    let mut history = pending.history.clone();
    let confirmed = JevDecision {
        executed: true,
        ..pending.decision.clone()
    };
    let Some(remaining) = remaining_goal_time(&pending) else {
        record_turn(&mut turns, &mut history, &confirmed, false);
        return run_response(
            JevStopReason::ActionUncertain,
            turns,
            Some(confirmed),
            pending.metrics,
        );
    };
    let after = tokio::time::timeout(
        remaining,
        observe_async(
            backend.clone(),
            pending.request.app.clone(),
            pending.request.window_id.clone(),
            pending.request.root.clone(),
        ),
    )
    .await;
    let changed = if let Ok(Ok(after)) = after {
        if !within_scope(&pending.request, &after) {
            return confirmed_scope_changed(&pending, &confirmed, &mut turns, &mut history);
        }
        if let Some(reply) = confirmed_unverified_result(
            &backend,
            &pending,
            &after,
            delivered_unverified,
            &confirmed,
            &mut turns,
            &mut history,
        )
        .await
        {
            return reply;
        }
        fingerprint(&after) != fingerprint(&fresh)
    } else {
        record_turn(&mut turns, &mut history, &confirmed, false);
        return run_response(
            JevStopReason::ActionUncertain,
            turns,
            Some(confirmed),
            pending.metrics,
        );
    };
    record_turn(&mut turns, &mut history, &confirmed, changed);
    let unchanged = if changed {
        0
    } else {
        pending.unchanged.saturating_add(1)
    };
    if unchanged >= 3 {
        return run_response(JevStopReason::Stalled, turns, None, pending.metrics);
    }
    let remaining_ms = remaining_goal_millis(&pending);
    let mut request = pending.request;
    request.max_elapsed_ms = remaining_ms;
    if request.max_elapsed_ms == 0 {
        return run_response(JevStopReason::TimeBudget, turns, None, pending.metrics);
    }
    request.continuation = None;
    request.max_steps = request.max_steps.saturating_sub(1);
    if pending.decision.operation == JevOperation::TypeText
        && request.text_slots.is_empty()
        && !request.text.is_empty()
    {
        request.text.remove(0);
    }
    if request.max_steps == 0 {
        return run_response(JevStopReason::ActionBudget, turns, None, pending.metrics);
    }
    if request.max_model_calls == 0 {
        return run_response(JevStopReason::ModelBudget, turns, None, pending.metrics);
    }
    merge_continuation(
        run_goal_fresh(backend, runtime, request, history, unchanged).await,
        turns,
        &pending.metrics,
    )
}

fn confirmed_scope_changed(
    pending: &PendingRun,
    confirmed: &JevDecision,
    turns: &mut Vec<JevTurn>,
    history: &mut Vec<String>,
) -> DesktopResponse {
    record_turn(turns, history, confirmed, false);
    run_response(
        JevStopReason::ScopeChanged,
        turns.clone(),
        Some(confirmed.clone()),
        pending.metrics.clone(),
    )
}

fn approved_pending(
    runtime: &JevRuntime,
    continuation: &GoalContinuation,
) -> Result<PendingRun, Box<DesktopResponse>> {
    let pending = take_pending(runtime, &continuation.id)?;
    if !continuation.approve {
        return Err(Box::new(run_response(
            JevStopReason::Cancelled,
            pending.turns,
            Some(pending.decision),
            pending.metrics,
        )));
    }
    Ok(pending)
}

fn remaining_goal_millis(pending: &PendingRun) -> u64 {
    remaining_goal_time(pending).map_or(0, |remaining| {
        u64::try_from(remaining.as_millis()).unwrap_or(u64::MAX)
    })
}

async fn confirmed_unverified_result<B: AgentBackend>(
    backend: &B,
    pending: &PendingRun,
    after: &Screen,
    delivered_unverified: bool,
    confirmed: &JevDecision,
    turns: &mut Vec<JevTurn>,
    history: &mut Vec<String>,
) -> Option<DesktopResponse> {
    if !delivered_unverified || confirmed.destructive < policy::DESTRUCTIVE {
        return None;
    }
    let mut evidence = verify(after, &pending.request.success);
    if !pending.request.success.is_empty() {
        let settle_until = Instant::now() + Duration::from_secs(2);
        while !satisfied(&evidence) && Instant::now() < settle_until {
            let Some(remaining) = remaining_goal_time(pending) else {
                break;
            };
            tokio::time::sleep(Duration::from_millis(200).min(remaining)).await;
            let Some(remaining) = remaining_goal_time(pending) else {
                break;
            };
            let observed = tokio::time::timeout(
                remaining,
                observe_async(
                    backend.clone(),
                    pending.request.app.clone(),
                    pending.request.window_id.clone(),
                    pending.request.root.clone(),
                ),
            )
            .await;
            let Ok(Ok(screen)) = observed else {
                continue;
            };
            if !within_scope(&pending.request, &screen) {
                record_turn(turns, history, confirmed, false);
                return Some(run_response(
                    JevStopReason::ScopeChanged,
                    turns.clone(),
                    Some(confirmed.clone()),
                    pending.metrics.clone(),
                ));
            }
            evidence = verify(&screen, &pending.request.success);
        }
    }
    let verified = satisfied(&evidence);
    record_turn(turns, history, confirmed, verified);
    Some(run_response_observed(
        if verified {
            JevStopReason::Done
        } else {
            JevStopReason::ActionUncertain
        },
        turns.clone(),
        (!verified).then_some(confirmed.clone()),
        pending.metrics.clone(),
        Some(evidence),
    ))
}

fn pending_stop(pending: &PendingRun, stop: JevStopReason) -> DesktopResponse {
    run_response(
        stop,
        pending.turns.clone(),
        Some(pending.decision.clone()),
        pending.metrics.clone(),
    )
}

async fn execute_pending_action<B: AgentBackend>(
    backend: &B,
    pending: &PendingRun,
) -> Result<(Screen, bool), Box<DesktopResponse>> {
    let remaining = remaining_goal_time(pending)
        .ok_or_else(|| Box::new(pending_stop(pending, JevStopReason::TimeBudget)))?;
    let fresh = match tokio::time::timeout(
        remaining,
        observe_async(
            backend.clone(),
            pending.request.app.clone(),
            pending.request.window_id.clone(),
            pending.request.root.clone(),
        ),
    )
    .await
    {
        Ok(Ok(screen)) => screen,
        Ok(Err(error)) => return Err(error),
        Err(_) => return Err(Box::new(pending_stop(pending, JevStopReason::TimeBudget))),
    };
    let target = current_target(pending, &fresh)
        .ok_or_else(|| Box::new(pending_stop(pending, JevStopReason::StaleTarget)))?;
    let text = (pending.decision.operation == JevOperation::TypeText)
        .then(|| {
            prepared_text(&pending.request, &target)
                .or_else(|| pending.request.text.first().cloned())
        })
        .flatten();
    if pending.decision.operation == JevOperation::TypeText && text.is_none() {
        return Err(Box::new(pending_stop(pending, JevStopReason::NeedsText)));
    }
    let remaining = remaining_goal_time(pending)
        .ok_or_else(|| Box::new(pending_stop(pending, JevStopReason::TimeBudget)))?;
    let reply = tokio::time::timeout(
        remaining,
        execute_operation(
            backend.clone(),
            pending.decision.operation,
            Some(target),
            text,
        ),
    )
    .await;
    let Ok(reply) = reply else {
        let mut turns = pending.turns.clone();
        turns.push(failed_turn(&turns, &pending.decision));
        return Err(Box::new(run_response(
            JevStopReason::ActionUncertain,
            turns,
            Some(pending.decision.clone()),
            pending.metrics.clone(),
        )));
    };
    if !reply.ok {
        return Err(Box::new(action_failed_response(
            pending.turns.clone(),
            pending.decision.clone(),
            pending.metrics.clone(),
            &reply,
        )));
    }
    let delivered_unverified = reply
        .data
        .as_ref()
        .and_then(|data| data.get("disposition"))
        .and_then(|disposition| disposition.get("delivery"))
        .and_then(serde_json::Value::as_str)
        == Some("delivered_unverified");
    Ok((fresh, delivered_unverified))
}

fn remaining_goal_time(pending: &PendingRun) -> Option<Duration> {
    Duration::from_millis(pending.request.max_elapsed_ms.clamp(1, 300_000))
        .checked_sub(pending.started.elapsed())
        .filter(|remaining| !remaining.is_zero())
}

fn take_pending(runtime: &JevRuntime, id: &str) -> Result<PendingRun, Box<DesktopResponse>> {
    let pending = runtime
        .pending
        .lock()
        .map_err(|_| Box::new(internal_error("confirmation state is unavailable")))?
        .remove(id)
        .ok_or_else(|| {
            Box::new(DesktopResponse::err(
                "run-goal",
                DesktopError::new(
                    "CONFIRMATION_EXPIRED",
                    "confirmation handle is absent or already consumed",
                ),
            ))
        })?;
    if pending.created.elapsed() >= Duration::from_secs(600) {
        return Err(Box::new(DesktopResponse::err(
            "run-goal",
            DesktopError::new("CONFIRMATION_EXPIRED", "confirmation handle expired"),
        )));
    }
    Ok(pending)
}

fn current_target(pending: &PendingRun, fresh: &Screen) -> Option<Candidate> {
    let mut matching = fresh.candidates.iter().filter(|candidate| {
        same_target(
            &pending.screen,
            fresh,
            &pending.target,
            candidate,
            pending.decision.operation,
        )
    });
    let target = matching.next()?;
    matching.next().is_none().then(|| target.clone())
}

fn merge_continuation(
    mut result: DesktopResponse,
    mut turns: Vec<JevTurn>,
    metrics: &JevMetrics,
) -> DesktopResponse {
    if let Some(data) = result.data.take() {
        if let Ok(mut continuation_result) = serde_json::from_value::<JevRunResult>(data.clone()) {
            turns.append(&mut continuation_result.turns);
            for (index, turn) in turns.iter_mut().enumerate() {
                turn.step = u32::try_from(index).unwrap_or(u32::MAX).saturating_add(1);
            }
            continuation_result.turns = turns;
            continuation_result.metrics.calls = continuation_result
                .metrics
                .calls
                .saturating_add(metrics.calls);
            continuation_result.metrics.attempts = continuation_result
                .metrics
                .attempts
                .saturating_add(metrics.attempts);
            continuation_result.metrics.latency_ms = continuation_result
                .metrics
                .latency_ms
                .saturating_add(metrics.latency_ms);
            continuation_result.metrics.input_tokens = continuation_result
                .metrics
                .input_tokens
                .saturating_add(metrics.input_tokens);
            continuation_result.metrics.output_tokens = continuation_result
                .metrics
                .output_tokens
                .saturating_add(metrics.output_tokens);
            result.data = serde_json::to_value(continuation_result).ok();
        } else {
            result.data = Some(data);
        }
    }
    result
}

fn same_target(
    before: &Screen,
    after: &Screen,
    old: &Candidate,
    current: &Candidate,
    operation: JevOperation,
) -> bool {
    let action = match operation {
        JevOperation::Click => "Click",
        JevOperation::TypeText => "SetValue",
        JevOperation::Check | JevOperation::Uncheck => "Toggle",
        JevOperation::Expand => "Expand",
        JevOperation::Collapse => "Collapse",
        JevOperation::Scroll => "Scroll",
        JevOperation::Drill => "Drill",
        _ => return false,
    };
    before.app == after.app
        && before.window == after.window
        && before.window_id == after.window_id
        && before.surface == after.surface
        && old.role == current.role
        && old.name == current.name
        && old.description == current.description
        && old.native_id == current.native_id
        && old.path == current.path
        && old.bounds == current.bounds
        && old.states == current.states
        && (old.label().is_some() || old.bounds.is_some())
        && (operation == JevOperation::Drill
            || current.available_actions.iter().any(|available| {
                available == action
                    || (operation == JevOperation::TypeText && available == "TypeText")
            }))
}

fn stop_reason(decision: JevDecisionKind) -> Option<JevStopReason> {
    match decision {
        JevDecisionKind::Done => Some(JevStopReason::Done),
        JevDecisionKind::Blocked => Some(JevStopReason::Blocked),
        JevDecisionKind::ConfirmationRequired => Some(JevStopReason::ConfirmationRequired),
        JevDecisionKind::Abstain => Some(JevStopReason::LowConfidence),
        JevDecisionKind::NeedsText => Some(JevStopReason::NeedsText),
        JevDecisionKind::Act => None,
    }
}

fn action_failed_response(
    mut turns: Vec<JevTurn>,
    decision: JevDecision,
    metrics: JevMetrics,
    failure: &DesktopResponse,
) -> DesktopResponse {
    turns.push(failed_turn(&turns, &decision));
    let stop = match failure
        .error
        .as_ref()
        .map(|error| error.disposition.delivery)
    {
        Some(DeliveryDisposition::NotDelivered) => JevStopReason::ActionFailed,
        _ => JevStopReason::ActionUncertain,
    };
    run_response(stop, turns, Some(decision), metrics)
}

fn failed_turn(turns: &[JevTurn], decision: &JevDecision) -> JevTurn {
    JevTurn {
        step: u32::try_from(turns.len())
            .unwrap_or(u32::MAX)
            .saturating_add(1),
        operation: decision.operation,
        target: decision.target.clone(),
        confidence: decision.confidence,
        ok: false,
        changed: false,
    }
}

fn run_response(
    stop: JevStopReason,
    turns: Vec<JevTurn>,
    pending: Option<JevDecision>,
    metrics: JevMetrics,
) -> DesktopResponse {
    run_response_with_id(stop, turns, pending, metrics, None)
}

fn run_response_observed(
    stop: JevStopReason,
    turns: Vec<JevTurn>,
    pending: Option<JevDecision>,
    metrics: JevMetrics,
    final_observation: Option<JevObservation>,
) -> DesktopResponse {
    response(
        "run-goal",
        &JevRunResult {
            verified: final_observation.as_ref().is_some_and(satisfied),
            final_observation,
            stop,
            turns,
            pending,
            confirmation_id: None,
            metrics,
        },
    )
}

fn run_response_with_id(
    stop: JevStopReason,
    turns: Vec<JevTurn>,
    pending: Option<JevDecision>,
    metrics: JevMetrics,
    confirmation_id: Option<String>,
) -> DesktopResponse {
    response(
        "run-goal",
        &JevRunResult {
            stop,
            verified: stop == JevStopReason::Done,
            final_observation: None,
            turns,
            pending,
            confirmation_id,
            metrics,
        },
    )
}

struct ResolveOutcome {
    decision: JevDecision,
    evaluations: Vec<EvaluationResult>,
    action_failure: Option<DesktopResponse>,
}

#[allow(clippy::too_many_arguments)]
async fn resolve<B: AgentBackend>(
    backend: &B,
    runtime: &JevRuntime,
    intent: &str,
    app: &str,
    root: Option<&str>,
    text: Option<&str>,
    include_values: bool,
    execute: bool,
    history: &[String],
    allow_rerank: bool,
) -> Result<ResolveOutcome, Box<DesktopResponse>> {
    let screen = observe_async(
        backend.clone(),
        app.to_owned(),
        None,
        root.map(str::to_owned),
    )
    .await?;
    resolve_on_screen(
        backend,
        runtime,
        intent,
        &screen,
        text,
        include_values,
        execute,
        history,
        allow_rerank,
        None,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn resolve_on_screen<B: AgentBackend>(
    backend: &B,
    runtime: &JevRuntime,
    intent: &str,
    screen: &Screen,
    text: Option<&str>,
    include_values: bool,
    execute: bool,
    history: &[String],
    allow_rerank: bool,
    scope: Option<&RunGoalRequest>,
) -> Result<ResolveOutcome, Box<DesktopResponse>> {
    if let Some(done) = visible_completion(intent, screen) {
        return Ok(done);
    }
    let space = scoped_action_space(screen, text.is_some(), scope);
    let evaluation = runtime
        .client
        .evaluate(&policy::request(
            &runtime.configuration.model,
            intent,
            screen,
            &space,
            history,
            include_values,
        ))
        .await
        .map_err(|error| provider_error(&error))?;
    let answers = &evaluation.response.answers;
    let (operation_name, operation_confidence) = choice(answers.get("operation"))
        .ok_or_else(|| invalid_response("operation answer was absent"))?;
    let operation_name = operation_name.to_owned();
    let operation = parse_operation(&operation_name)
        .ok_or_else(|| invalid_response("operation answer was unknown"))?;
    let mut destructive = noul(answers.get("destructive"));
    let target_answer_name = format!("{}_target", operation_name.to_ascii_lowercase());
    let mut selected = target(&space, &operation_name, answers.get(&target_answer_name))
        .map(|(candidate, confidence)| (candidate.clone(), confidence));
    let mut evaluations = vec![evaluation];
    if let Some((reranked_target, reranked)) = rerank(RerankInput {
        runtime,
        intent,
        screen,
        space: &space,
        operation: &operation_name,
        target_answer: &target_answer_name,
        selected: selected.as_ref(),
        include_values,
        first: &evaluations[0],
        allow: allow_rerank,
    })
    .await?
    {
        if let Some(reranked_target) = reranked_target {
            selected = Some(reranked_target);
        }
        evaluations.push(reranked);
    }
    let confidence = selected
        .as_ref()
        .map_or(operation_confidence, |(_, confidence)| *confidence);
    destructive = destructive.max(local_destructive_score(
        intent,
        operation,
        selected.as_ref(),
    ));
    let decision = gate_decision(&GateInput {
        intent,
        operation,
        operation_name: &operation_name,
        operation_confidence,
        confidence,
        destructive,
        selected: selected.as_ref(),
        space: &space,
        has_text: text.is_some(),
    });
    let target = selected
        .as_ref()
        .map(|(candidate, _)| target_payload(candidate));
    let mut out = JevDecision {
        decision,
        operation,
        target,
        confidence,
        destructive,
        reason: reason(decision, confidence, destructive),
        executed: false,
    };
    let (executed, action_failure) = execute_if_requested(
        backend,
        execute && decision == JevDecisionKind::Act,
        operation,
        selected.map(|(node, _)| node),
        text,
    )
    .await;
    out.executed = executed;
    Ok(ResolveOutcome {
        decision: out,
        evaluations,
        action_failure,
    })
}

fn scoped_action_space(
    screen: &Screen,
    has_text: bool,
    scope: Option<&RunGoalRequest>,
) -> policy::ActionSpace {
    let mut space = action_space(screen, has_text);
    if let Some(scope) = scope {
        space.targets.retain(|operation, candidates| {
            let Some(parsed) = parse_operation(operation) else {
                return false;
            };
            if mutates(parsed)
                && !scope.allowed_operations.is_empty()
                && !scope.allowed_operations.contains(&parsed)
            {
                return false;
            }
            candidates.retain(|_, candidate| target_allowed(scope, candidate));
            !candidates.is_empty()
        });
    }
    space
}

struct GateInput<'a> {
    intent: &'a str,
    operation: JevOperation,
    operation_name: &'a str,
    operation_confidence: f64,
    confidence: f64,
    destructive: f64,
    selected: Option<&'a (Candidate, f64)>,
    space: &'a policy::ActionSpace,
    has_text: bool,
}

fn gate_decision(input: &GateInput<'_>) -> JevDecisionKind {
    let candidate = input.selected.map(|(candidate, _)| candidate);
    let mut decision = gate_with_evidence(
        input.operation,
        input.confidence,
        input.destructive,
        exact_named_match(input.intent, candidate)
            || positional_match(
                input.intent,
                candidate,
                input.space.targets.get(input.operation_name),
            ),
    );
    if mutates(input.operation) && input.operation_confidence < policy::ACT {
        decision = JevDecisionKind::Abstain;
    }
    if input.space.targets.contains_key(input.operation_name) && input.selected.is_none() {
        decision = JevDecisionKind::Abstain;
    }
    if input.operation == JevOperation::TypeText && !input.has_text {
        decision = JevDecisionKind::NeedsText;
    }
    decision
}

fn local_destructive_score(
    intent: &str,
    operation: JevOperation,
    selected: Option<&(Candidate, f64)>,
) -> f64 {
    if deterministic_destructive(intent, operation, selected.map(|(candidate, _)| candidate)) {
        1.0
    } else {
        0.0
    }
}

async fn execute_if_requested<B: AgentBackend>(
    backend: &B,
    execute: bool,
    operation: JevOperation,
    target: Option<Candidate>,
    text: Option<&str>,
) -> (bool, Option<DesktopResponse>) {
    if !execute {
        return (false, None);
    }
    let response =
        execute_operation(backend.clone(), operation, target, text.map(str::to_owned)).await;
    if response.ok {
        (true, None)
    } else {
        (false, Some(response))
    }
}

fn visible_completion(intent: &str, screen: &Screen) -> Option<ResolveOutcome> {
    playing_goal_satisfied(intent, screen).then(|| ResolveOutcome {
        decision: JevDecision {
            decision: JevDecisionKind::Done,
            operation: JevOperation::Done,
            target: None,
            confidence: 1.0,
            destructive: 0.0,
            reason: "the requested playback state is visibly satisfied".to_owned(),
            executed: false,
        },
        evaluations: Vec::new(),
        action_failure: None,
    })
}

struct RerankInput<'a> {
    runtime: &'a JevRuntime,
    intent: &'a str,
    screen: &'a Screen,
    space: &'a policy::ActionSpace,
    operation: &'a str,
    target_answer: &'a str,
    selected: Option<&'a (Candidate, f64)>,
    include_values: bool,
    first: &'a EvaluationResult,
    allow: bool,
}

async fn rerank(
    input: RerankInput<'_>,
) -> Result<Option<(Option<(Candidate, f64)>, EvaluationResult)>, Box<DesktopResponse>> {
    if !input.allow
        || !input
            .selected
            .is_some_and(|(_, confidence)| *confidence < policy::ACT)
    {
        return Ok(None);
    }
    let candidates = shortlist(
        input.space,
        input.operation,
        input.first.response.answers.get(input.target_answer),
    );
    if candidates.len() <= 1 {
        return Ok(None);
    }
    let evaluation = input
        .runtime
        .client
        .evaluate(&policy::rerank_request(
            &input.runtime.configuration.model,
            input.intent,
            input.screen,
            input.operation,
            &candidates,
            input.include_values,
        ))
        .await
        .map_err(|error| provider_error(&error))?;
    let selected =
        choice(evaluation.response.answers.get("target")).and_then(|(choice, confidence)| {
            candidates
                .get(choice)
                .cloned()
                .map(|candidate| (candidate, confidence))
        });
    Ok(Some((selected, evaluation)))
}

trait AgentBackend: Clone + Send + 'static {
    fn observe(
        &self,
        app: &str,
        window_id: Option<&str>,
        root: Option<&str>,
    ) -> Result<Screen, Box<DesktopResponse>>;
    fn execute(
        &self,
        operation: JevOperation,
        target: Option<Candidate>,
        text: Option<String>,
    ) -> DesktopResponse;
}

impl AgentBackend for Desktop {
    fn observe(
        &self,
        app: &str,
        window_id: Option<&str>,
        root: Option<&str>,
    ) -> Result<Screen, Box<DesktopResponse>> {
        observe(self, app, window_id, root)
    }

    fn execute(
        &self,
        operation: JevOperation,
        target: Option<Candidate>,
        text: Option<String>,
    ) -> DesktopResponse {
        execute_desktop(self, operation, target.as_ref(), text)
    }
}

async fn observe_async<B: AgentBackend>(
    backend: B,
    app: String,
    window_id: Option<String>,
    root: Option<String>,
) -> Result<Screen, Box<DesktopResponse>> {
    tokio::task::spawn_blocking(move || {
        backend.observe(&app, window_id.as_deref(), root.as_deref())
    })
    .await
    .map_err(|error| {
        Box::new(internal_error(&format!(
            "desktop observation task failed: {error}"
        )))
    })?
}

async fn execute_operation<B: AgentBackend>(
    backend: B,
    operation: JevOperation,
    target: Option<Candidate>,
    text: Option<String>,
) -> DesktopResponse {
    tokio::task::spawn_blocking(move || backend.execute(operation, target, text))
        .await
        .unwrap_or_else(|error| internal_error(&format!("desktop action task failed: {error}")))
}

fn execute_desktop(
    desktop: &Desktop,
    operation: JevOperation,
    target: Option<&Candidate>,
    text: Option<String>,
) -> DesktopResponse {
    let ref_id = target.map(|node| node.ref_id.clone());
    match operation {
        JevOperation::Click => desktop.click(RefRequest::new(ref_id.unwrap_or_default())),
        JevOperation::TypeText => desktop.set_value(SetValueRequest {
            ref_id: ref_id.unwrap_or_default(),
            value: text.unwrap_or_default(),
            ..SetValueRequest::default()
        }),
        JevOperation::Check => desktop.check(RefRequest::new(ref_id.unwrap_or_default())),
        JevOperation::Uncheck => desktop.uncheck(RefRequest::new(ref_id.unwrap_or_default())),
        JevOperation::Expand => desktop.expand(RefRequest::new(ref_id.unwrap_or_default())),
        JevOperation::Collapse => desktop.collapse(RefRequest::new(ref_id.unwrap_or_default())),
        JevOperation::Scroll => desktop.scroll(ScrollRequest::new(
            ref_id.unwrap_or_default(),
            tinydesktop_bus::Direction::Down,
            3,
        )),
        JevOperation::Wait => desktop.wait(WaitRequest::sleep(500)),
        JevOperation::Drill | JevOperation::Widen => {
            DesktopResponse::ok("look", json!({"root": ref_id}))
        }
        JevOperation::Done | JevOperation::Blocked => {
            DesktopResponse::ok("resolve-intent", json!({}))
        }
    }
}

fn target_payload(candidate: &Candidate) -> JevTarget {
    JevTarget {
        ref_id: candidate.ref_id.clone(),
        role: candidate.role.clone(),
        name: candidate.label().map(str::to_owned),
    }
}

fn reason(decision: JevDecisionKind, confidence: f64, destructive: f64) -> String {
    match decision {
        JevDecisionKind::Act => "the target cleared the safe-action threshold".to_owned(),
        JevDecisionKind::ConfirmationRequired => {
            format!("the action is hard to undo ({destructive:.2})")
        }
        JevDecisionKind::Abstain => format!("target confidence {confidence:.2} is too low"),
        JevDecisionKind::NeedsText => {
            "the selected operation needs caller-supplied text".to_owned()
        }
        JevDecisionKind::Done => "the goal is visibly satisfied".to_owned(),
        JevDecisionKind::Blocked => "no offered operation can advance the goal".to_owned(),
    }
}

fn merge_metrics(metrics: &mut JevMetrics, evaluation: &EvaluationResult) {
    metrics.calls = metrics.calls.saturating_add(1);
    metrics.attempts = metrics.attempts.saturating_add(evaluation.attempts);
    metrics.latency_ms = metrics.latency_ms.saturating_add(
        evaluation
            .latency
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX),
    );
    metrics.input_tokens = metrics
        .input_tokens
        .saturating_add(evaluation.response.usage.input_tokens.unwrap_or_default());
    metrics.output_tokens = metrics
        .output_tokens
        .saturating_add(evaluation.response.usage.output_tokens.unwrap_or_default());
    metrics.model = Some(evaluation.response.model.clone());
}

fn response<T: serde::Serialize>(command: &str, value: &T) -> DesktopResponse {
    match serde_json::to_value(value) {
        Ok(data) => DesktopResponse::ok(command, data),
        Err(error) => internal_error(&format!("cannot encode Jev result: {error}")),
    }
}

fn config_error(error: &JevError) -> Box<DesktopError> {
    Box::new(DesktopError::new("JEV_INVALID_CONFIG", error.to_string()))
}

fn provider_error(error: &tinyjevclient::EvaluationFailure) -> Box<DesktopResponse> {
    let code = match &error.error {
        JevError::Authentication => "JEV_AUTHENTICATION",
        JevError::RateLimited => "JEV_RATE_LIMITED",
        JevError::Timeout => "JEV_TIMEOUT",
        JevError::InvalidResponse { .. } | JevError::Decode { .. } => "JEV_INVALID_RESPONSE",
        _ => "JEV_PROVIDER_FAILED",
    };
    Box::new(DesktopResponse::err(
        "jev-evaluate",
        DesktopError::new(code, error.to_string()),
    ))
}

fn invalid_response(message: &str) -> Box<DesktopResponse> {
    Box::new(DesktopResponse::err(
        "jev-evaluate",
        DesktopError::new("JEV_INVALID_RESPONSE", message),
    ))
}

fn internal_error(message: &str) -> DesktopResponse {
    DesktopResponse::err("jev-desktop", DesktopError::new("INTERNAL", message))
}
