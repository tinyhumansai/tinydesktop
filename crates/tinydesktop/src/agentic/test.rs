//! Tests for deterministic Jev desktop-control policy.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use super::{
    AgentBackend, Evaluator, JevRuntime, execute_desktop, internal_error,
    policy::{
        ACT, DESTRUCTIVE, FLOOR, action_space, choice, deterministic_destructive,
        exact_named_match, gate_with_evidence, noul, parse_operation, playing_goal_satisfied,
        positional_match, request, rerank_request, shortlist, target,
    },
    provider_error, reason, resolve_intent, resolve_intent_with, response as agent_response,
    run_goal, run_goal_with, same_target,
    screen::{
        Candidate, NativeId, Screen, describe, fingerprint, observe, parse_reply, snapshot_request,
    },
    target_payload, visible_completion,
};
use serde_json::json;
use tinydesktop_bus::{
    DesktopResponse, GoalContinuation, JevConfig, JevDecisionKind, JevOperation, JevProvider,
    JevStopReason, RunGoalRequest, VisiblePredicate,
};
use tinyjevclient::{Answer, ChoiceAnswer};

#[test]
fn execution_gates_on_selected_probability_not_distribution_concentration() {
    let answer = Answer::Choice(ChoiceAnswer {
        choice: "liked".to_owned(),
        probabilities: BTreeMap::from([("liked".to_owned(), 0.91), ("other".to_owned(), 0.09)]),
        confidence: 0.41,
    });

    assert_eq!(choice(Some(&answer)), Some(("liked", 0.91)));
}

#[test]
fn destructive_actions_always_require_confirmation() {
    assert_eq!(
        gate_with_evidence(JevOperation::Click, 0.99, DESTRUCTIVE, false),
        JevDecisionKind::ConfirmationRequired
    );
}

#[test]
fn only_safe_confident_actions_are_executable() {
    assert_eq!(
        gate_with_evidence(JevOperation::Click, ACT, DESTRUCTIVE - 0.01, false),
        JevDecisionKind::Act
    );
    assert_eq!(
        gate_with_evidence(JevOperation::Click, FLOOR - 0.01, 0.0, false),
        JevDecisionKind::Abstain
    );
}

#[test]
fn an_exact_multiword_accessible_name_is_strong_identity_evidence() {
    let candidate = Candidate {
        name: Some("Liked Songs".to_owned()),
        ..Candidate::default()
    };
    assert!(exact_named_match(
        "open Liked Songs and play the first track",
        Some(&candidate)
    ));
    assert_eq!(
        gate_with_evidence(JevOperation::Click, 0.52, 0.05, true),
        JevDecisionKind::Act
    );
}

#[test]
fn topmost_play_target_is_strong_positional_evidence() {
    let top = Candidate {
        ref_id: "@s:e1".to_owned(),
        name: Some("Play First Song by Artist".to_owned()),
        bounds: Some(serde_json::json!({"x": 10.0, "y": 100.0})),
        ..Candidate::default()
    };
    let lower = Candidate {
        ref_id: "@s:e2".to_owned(),
        name: Some("Play Second Song by Artist".to_owned()),
        bounds: Some(serde_json::json!({"x": 10.0, "y": 160.0})),
        ..Candidate::default()
    };
    let peers = BTreeMap::from([("1".to_owned(), top.clone()), ("2".to_owned(), lower)]);

    assert!(positional_match(
        "play the topmost song",
        Some(&top),
        Some(&peers)
    ));
    assert_eq!(
        gate_with_evidence(JevOperation::Click, 0.49, 0.05, true),
        JevDecisionKind::Act
    );
}

#[test]
fn visible_pause_on_the_top_track_completes_a_playing_goal() {
    let mut screen = two_candidate_screen();
    screen.candidates[0].name = Some("Pause First Song by Artist".to_owned());
    assert!(playing_goal_satisfied(
        "ensure the topmost song is playing",
        &screen
    ));
    assert!(visible_completion("ensure the topmost song is playing", &screen).is_some());
    assert!(!playing_goal_satisfied("open the playlist", &screen));
}

#[test]
fn terminal_operations_are_not_treated_as_actions() {
    assert_eq!(
        gate_with_evidence(JevOperation::Done, 1.0, 1.0, false),
        JevDecisionKind::Done
    );
    assert_eq!(
        gate_with_evidence(JevOperation::Blocked, 1.0, 1.0, false),
        JevDecisionKind::Blocked
    );
    assert_eq!(
        gate_with_evidence(JevOperation::Done, ACT - 0.01, 0.0, false),
        JevDecisionKind::Abstain
    );
}

#[test]
fn deterministic_risk_and_identity_checks_fail_closed() {
    let delete = Candidate {
        name: Some("Delete account".to_owned()),
        ..Candidate::default()
    };
    assert!(deterministic_destructive(
        "continue",
        JevOperation::Click,
        Some(&delete)
    ));
    assert!(!deterministic_destructive(
        "continue",
        JevOperation::Scroll,
        Some(&delete)
    ));
    let candidate = Candidate {
        name: Some("Liked Songs".to_owned()),
        ..Candidate::default()
    };
    assert!(!exact_named_match("open Disliked Songs", Some(&candidate)));
    let decorated = Candidate {
        name: Some("Liked Songs Pinned Downloaded Playlist".to_owned()),
        ..Candidate::default()
    };
    assert!(exact_named_match("open Liked Songs", Some(&decorated)));
}

#[derive(Clone)]
struct FakeBackend {
    screens: Arc<Mutex<VecDeque<Screen>>>,
    operations: Arc<Mutex<Vec<JevOperation>>>,
    fail_execute: bool,
}

#[derive(Clone)]
struct RecordingTextBackend {
    inner: FakeBackend,
    values: Arc<Mutex<Vec<Option<String>>>>,
}

#[derive(Clone)]
struct WindowBoundBackend {
    inner: FakeBackend,
    requested: Arc<Mutex<Vec<Option<String>>>>,
}

#[derive(Clone)]
struct ReadinessBackend {
    inner: FakeBackend,
    attempts: Arc<Mutex<u32>>,
    first_error: &'static str,
}

#[derive(Clone)]
struct UnverifiedClickBackend {
    inner: FakeBackend,
}

impl AgentBackend for UnverifiedClickBackend {
    fn observe(
        &self,
        app: &str,
        window_id: Option<&str>,
        root: Option<&str>,
    ) -> Result<Screen, Box<DesktopResponse>> {
        self.inner.observe(app, window_id, root)
    }

    fn execute(
        &self,
        operation: JevOperation,
        target: Option<Candidate>,
        text: Option<String>,
    ) -> DesktopResponse {
        let _ = self.inner.execute(operation, target, text);
        DesktopResponse::ok(
            "click",
            json!({"disposition":{"delivery":"delivered_unverified"}}),
        )
    }
}

impl AgentBackend for ReadinessBackend {
    fn observe(
        &self,
        app: &str,
        window_id: Option<&str>,
        root: Option<&str>,
    ) -> Result<Screen, Box<DesktopResponse>> {
        let mut attempts = self.attempts.lock().unwrap();
        *attempts += 1;
        if *attempts == 1 {
            return Err(Box::new(DesktopResponse::err(
                "snapshot",
                tinydesktop_bus::DesktopError::new(self.first_error, "not ready"),
            )));
        }
        drop(attempts);
        self.inner.observe(app, window_id, root)
    }

    fn execute(
        &self,
        operation: JevOperation,
        target: Option<Candidate>,
        text: Option<String>,
    ) -> DesktopResponse {
        self.inner.execute(operation, target, text)
    }
}

impl AgentBackend for WindowBoundBackend {
    fn observe(
        &self,
        app: &str,
        window_id: Option<&str>,
        root: Option<&str>,
    ) -> Result<Screen, Box<DesktopResponse>> {
        self.requested
            .lock()
            .unwrap()
            .push(window_id.map(str::to_owned));
        if window_id != Some("w-515619") {
            return Err(Box::new(DesktopResponse::err(
                "snapshot",
                tinydesktop_bus::DesktopError::new(
                    "WINDOW_NOT_FOUND",
                    "requested window is unavailable",
                ),
            )));
        }
        self.inner.observe(app, window_id, root)
    }

    fn execute(
        &self,
        operation: JevOperation,
        target: Option<Candidate>,
        text: Option<String>,
    ) -> DesktopResponse {
        self.inner.execute(operation, target, text)
    }
}

impl AgentBackend for RecordingTextBackend {
    fn observe(
        &self,
        app: &str,
        window_id: Option<&str>,
        root: Option<&str>,
    ) -> Result<Screen, Box<DesktopResponse>> {
        self.inner.observe(app, window_id, root)
    }

    fn execute(
        &self,
        operation: JevOperation,
        target: Option<Candidate>,
        text: Option<String>,
    ) -> DesktopResponse {
        self.values.lock().unwrap().push(text.clone());
        self.inner.execute(operation, target, text)
    }
}

impl AgentBackend for FakeBackend {
    fn observe(
        &self,
        _app: &str,
        _window_id: Option<&str>,
        _root: Option<&str>,
    ) -> Result<Screen, Box<DesktopResponse>> {
        self.screens
            .lock()
            .expect("screen lock")
            .pop_front()
            .ok_or_else(|| {
                Box::new(DesktopResponse::err(
                    "snapshot",
                    tinydesktop_bus::DesktopError::new("EMPTY", "no screen"),
                ))
            })
    }

    fn execute(
        &self,
        operation: JevOperation,
        _target: Option<Candidate>,
        _text: Option<String>,
    ) -> DesktopResponse {
        self.operations
            .lock()
            .expect("operation lock")
            .push(operation);
        if self.fail_execute {
            DesktopResponse::err(
                "fake",
                tinydesktop_bus::DesktopError::new("ACTION_FAILED", "fake failure"),
            )
        } else {
            DesktopResponse::ok("fake", json!({"delivery": "delivered_verified"}))
        }
    }
}

fn clickable_screen() -> Screen {
    Screen {
        app: "Spotify".to_owned(),
        window: Some("Liked Songs".to_owned()),
        window_id: None,
        surface: "window".to_owned(),
        root: None,
        candidates: vec![Candidate {
            ref_id: "@s1:e1".to_owned(),
            role: "button".to_owned(),
            name: Some("Play First Song by Artist".to_owned()),
            available_actions: vec!["Click".to_owned()],
            bounds: Some(json!({"x": 10.0, "y": 100.0})),
            ..Candidate::default()
        }],
        observed: Vec::new(),
    }
}

fn two_candidate_screen() -> Screen {
    let mut screen = clickable_screen();
    screen.candidates.push(Candidate {
        ref_id: "@s1:e2".to_owned(),
        role: "button".to_owned(),
        name: Some("Play Second Song by Artist".to_owned()),
        available_actions: vec!["Click".to_owned()],
        bounds: Some(json!({"x": 10.0, "y": 160.0})),
        ..Candidate::default()
    });
    screen
}

fn response(operation: &str, probability: f64, target: &str) -> tinyjevclient::EvaluationResult {
    response_with(operation, probability, target, 0.9, 0.05)
}

fn response_with(
    operation: &str,
    probability: f64,
    target: &str,
    selected_target_probability: f64,
    destructive: f64,
) -> tinyjevclient::EvaluationResult {
    let remainder = (1.0 - probability) / 3.0;
    let target_probability = if target == "1" {
        selected_target_probability
    } else {
        1.0 - selected_target_probability
    };
    evaluation(json!({
        "model": "typesafe/jev-1.13-20260917",
        "answers": {
            "operation": {
                "type": "choice", "choice": operation, "confidence": 0.4,
                "probabilities": {
                    "CLICK": if operation == "CLICK" { probability } else { remainder },
                    "WAIT": if operation == "WAIT" { probability } else { remainder },
                    "DONE": if operation == "DONE" { probability } else { remainder },
                    "BLOCKED": if operation == "BLOCKED" { probability } else { remainder }
                }
            },
            "click_target": {
                "type": "choice", "choice": target, "confidence": 0.4,
                "probabilities": {"1": target_probability, "none": 1.0 - target_probability}
            },
            "destructive": {"type": "noul", "noul": destructive}
        },
        "usage": {"input_tokens": 10, "output_tokens": 2}
    }))
}

fn evaluation(value: serde_json::Value) -> tinyjevclient::EvaluationResult {
    let response = serde_json::from_value(value).expect("mock response decodes");
    tinyjevclient::EvaluationResult {
        response,
        request_id: Some("mock-request".to_owned()),
        attempts: 1,
        latency: Duration::from_millis(1),
    }
}

struct MockEvaluator {
    results: Mutex<VecDeque<tinyjevclient::EvaluationResult>>,
    requests: Arc<Mutex<Vec<tinyjevclient::EvaluationRequest>>>,
}

impl Evaluator for MockEvaluator {
    fn evaluate<'a>(
        &'a self,
        request: &'a tinyjevclient::EvaluationRequest,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<
                        tinyjevclient::EvaluationResult,
                        tinyjevclient::EvaluationFailure,
                    >,
                > + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            self.requests
                .lock()
                .expect("request lock")
                .push(request.clone());
            Ok(self
                .results
                .lock()
                .expect("evaluation lock")
                .pop_front()
                .expect("mock evaluation"))
        })
    }
}

fn runtime(results: Vec<tinyjevclient::EvaluationResult>) -> JevRuntime {
    runtime_recording(results).0
}

fn runtime_recording(
    results: Vec<tinyjevclient::EvaluationResult>,
) -> (
    JevRuntime,
    Arc<Mutex<Vec<tinyjevclient::EvaluationRequest>>>,
) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    (
        JevRuntime {
            client: Arc::new(MockEvaluator {
                results: Mutex::new(VecDeque::from(results)),
                requests: Arc::clone(&requests),
            }),
            configuration: tinydesktop_bus::JevConfiguration {
                provider: tinydesktop_bus::JevProvider::OpenRouter,
                model: "jev-latest".to_owned(),
                endpoint_url: None,
            },
            pending: Arc::new(Mutex::new(std::collections::HashMap::new())),
        },
        requests,
    )
}

fn backend(screen_count: usize) -> (FakeBackend, Arc<Mutex<Vec<JevOperation>>>) {
    let operations = Arc::new(Mutex::new(Vec::new()));
    (
        FakeBackend {
            screens: Arc::new(Mutex::new(VecDeque::from(
                (0..screen_count)
                    .map(|_| clickable_screen())
                    .collect::<Vec<_>>(),
            ))),
            operations: Arc::clone(&operations),
            fail_execute: false,
        },
        operations,
    )
}

#[tokio::test]
async fn goal_loop_executes_a_safe_choice_then_stops_done() {
    let runtime = runtime(vec![
        response("CLICK", 0.9, "1"),
        response("DONE", 0.9, "none"),
    ]);
    let (backend, operations) = backend(4);
    let reply = run_goal_with(
        backend,
        runtime,
        RunGoalRequest {
            app: "Spotify".to_owned(),
            goal: "play the topmost song".to_owned(),
            max_steps: 3,
            max_model_calls: 3,
            ..RunGoalRequest::default()
        },
    )
    .await;
    assert!(reply.ok);
    let result: tinydesktop_bus::JevRunResult =
        serde_json::from_value(reply.data.expect("run returns data")).expect("result decodes");
    assert_eq!(result.stop, JevStopReason::Done);
    assert_eq!(result.turns.len(), 1);
    assert_eq!(
        *operations.lock().expect("operation lock"),
        vec![JevOperation::Click]
    );
    assert_eq!((result.metrics.calls, result.metrics.attempts), (2, 2));
}

#[tokio::test]
async fn scoped_task_executes_two_consequential_steps_in_one_call_without_confirmations() {
    let runtime = runtime(vec![
        response_with("CLICK", 0.94, "1", 0.94, 0.9),
        response_with("CLICK", 0.94, "1", 0.94, 0.9),
    ]);
    let (backend, operations) = backend(6);
    {
        let mut screens = backend.screens.lock().unwrap();
        for screen in screens.iter_mut().skip(2).take(3) {
            screen.candidates[0].name = Some("Second Step".into());
        }
        screens[5].candidates[0].name = Some("Finished".into());
    }
    let reply = run_goal_with(
        backend,
        runtime,
        RunGoalRequest {
            app: "Spotify".into(),
            goal: "complete the two step task".into(),
            window: Some("Liked Songs".into()),
            allowed_operations: vec![JevOperation::Click],
            allowed_targets: vec!["Play First Song by Artist".into(), "Second Step".into()],
            success: vec![VisiblePredicate::NamePresent {
                name: "Finished".into(),
            }],
            require_confirmations: false,
            ..RunGoalRequest::default()
        },
    )
    .await;
    let result: tinydesktop_bus::JevRunResult =
        serde_json::from_value(reply.data.unwrap()).unwrap();
    assert_eq!(result.stop, JevStopReason::Done);
    assert!(result.verified);
    assert_eq!(result.turns.len(), 2);
    assert!(result.confirmation_id.is_none());
    assert_eq!(
        *operations.lock().unwrap(),
        vec![JevOperation::Click, JevOperation::Click]
    );
}

#[test]
fn snapshot_request_binds_exact_window_id() {
    let request = snapshot_request("TextEdit", Some("w-515619"), None);
    assert_eq!(request.app.as_deref(), Some("TextEdit"));
    assert_eq!(request.window_id.as_deref(), Some("w-515619"));
    assert_eq!(snapshot_request("TextEdit", None, None).window_id, None);
}

#[tokio::test]
async fn goal_waits_for_a_starting_app_but_not_for_denied_permission() {
    for (code, expected_attempts, done) in [
        ("APP_NOT_FOUND", 2, true),
        ("WINDOW_NOT_FOUND", 2, true),
        ("PERM_DENIED", 1, false),
    ] {
        let (inner, _) = backend(1);
        let attempts = Arc::new(Mutex::new(0));
        let backend = ReadinessBackend {
            inner,
            attempts: Arc::clone(&attempts),
            first_error: code,
        };
        let reply = run_goal_with(
            backend,
            runtime(Vec::new()),
            RunGoalRequest {
                app: "Spotify".into(),
                goal: "verify the visible song".into(),
                success: vec![VisiblePredicate::NamePresent {
                    name: "Play First Song by Artist".into(),
                }],
                ..RunGoalRequest::default()
            },
        )
        .await;
        assert_eq!(*attempts.lock().unwrap(), expected_attempts, "{code}");
        if done {
            let result: tinydesktop_bus::JevRunResult =
                serde_json::from_value(reply.data.unwrap()).unwrap();
            assert_eq!(result.stop, JevStopReason::Done);
            assert!(result.verified);
        } else {
            assert_eq!(reply.error.unwrap().code, code);
        }
    }
}

#[tokio::test]
async fn goal_waits_for_delayed_success_without_replaying_an_unverified_click() {
    let (inner, operations) = backend(4);
    inner.screens.lock().unwrap()[3].candidates[0].name = Some("Finished".into());
    let reply = run_goal_with(
        UnverifiedClickBackend { inner },
        runtime(vec![response("CLICK", 0.95, "1")]),
        RunGoalRequest {
            app: "Spotify".into(),
            goal: "click play and verify finished".into(),
            allowed_operations: vec![JevOperation::Click],
            allowed_targets: vec!["Play First Song by Artist".into()],
            success: vec![VisiblePredicate::NamePresent {
                name: "Finished".into(),
            }],
            require_confirmations: false,
            ..RunGoalRequest::default()
        },
    )
    .await;
    let result: tinydesktop_bus::JevRunResult =
        serde_json::from_value(reply.data.unwrap()).unwrap();
    assert_eq!(result.stop, JevStopReason::Done);
    assert!(result.verified);
    assert_eq!(result.turns.len(), 1);
    assert_eq!(*operations.lock().unwrap(), vec![JevOperation::Click]);
}

#[tokio::test]
async fn unverified_consequential_action_stops_after_settle_without_replay() {
    let (inner, operations) = backend(3);
    let reply = run_goal_with(
        UnverifiedClickBackend { inner },
        runtime(vec![response_with("CLICK", 0.95, "1", 0.95, 0.9)]),
        RunGoalRequest {
            app: "Spotify".into(),
            goal: "send the selected item".into(),
            allowed_operations: vec![JevOperation::Click],
            allowed_targets: vec!["Play First Song by Artist".into()],
            success: vec![VisiblePredicate::NamePresent {
                name: "Finished".into(),
            }],
            max_elapsed_ms: 4_000,
            require_confirmations: false,
            ..RunGoalRequest::default()
        },
    )
    .await;
    let result: tinydesktop_bus::JevRunResult =
        serde_json::from_value(reply.data.unwrap()).unwrap();
    assert_eq!(result.stop, JevStopReason::ActionUncertain);
    assert_eq!(result.turns.len(), 1);
    assert_eq!(*operations.lock().unwrap(), vec![JevOperation::Click]);
}

#[tokio::test]
async fn approved_unverified_consequential_action_without_predicate_never_replays() {
    let (inner, operations) = backend(3);
    let backend = UnverifiedClickBackend { inner };
    let runtime = runtime(vec![response_with("CLICK", 0.95, "1", 0.95, 0.9)]);
    let stopped = run_goal_with(
        backend.clone(),
        runtime.clone(),
        RunGoalRequest {
            app: "Spotify".into(),
            goal: "send the selected item".into(),
            ..RunGoalRequest::default()
        },
    )
    .await;
    let stopped: tinydesktop_bus::JevRunResult =
        serde_json::from_value(stopped.data.unwrap()).unwrap();
    assert_eq!(stopped.stop, JevStopReason::ConfirmationRequired);
    let resumed = run_goal_with(
        backend,
        runtime,
        RunGoalRequest {
            continuation: Some(GoalContinuation {
                id: stopped.confirmation_id.unwrap(),
                approve: true,
            }),
            ..RunGoalRequest::default()
        },
    )
    .await;
    let result: tinydesktop_bus::JevRunResult =
        serde_json::from_value(resumed.data.unwrap()).unwrap();
    assert_eq!(result.stop, JevStopReason::ActionUncertain);
    assert_eq!(result.turns.len(), 1);
    assert_eq!(*operations.lock().unwrap(), vec![JevOperation::Click]);
}

#[tokio::test]
async fn approved_unverified_action_waits_for_delayed_visible_success() {
    let (inner, operations) = backend(4);
    inner.screens.lock().unwrap()[3].candidates[0].name = Some("Finished".into());
    let backend = UnverifiedClickBackend { inner };
    let runtime = runtime(vec![response_with("CLICK", 0.95, "1", 0.95, 0.9)]);
    let stopped = run_goal_with(
        backend.clone(),
        runtime.clone(),
        RunGoalRequest {
            app: "Spotify".into(),
            goal: "send the selected item".into(),
            success: vec![VisiblePredicate::NamePresent {
                name: "Finished".into(),
            }],
            ..RunGoalRequest::default()
        },
    )
    .await;
    let stopped: tinydesktop_bus::JevRunResult =
        serde_json::from_value(stopped.data.unwrap()).unwrap();
    assert_eq!(stopped.stop, JevStopReason::ConfirmationRequired);
    let resumed = run_goal_with(
        backend,
        runtime,
        RunGoalRequest {
            continuation: Some(GoalContinuation {
                id: stopped.confirmation_id.unwrap(),
                approve: true,
            }),
            ..RunGoalRequest::default()
        },
    )
    .await;
    let result: tinydesktop_bus::JevRunResult =
        serde_json::from_value(resumed.data.unwrap()).unwrap();
    assert_eq!(result.stop, JevStopReason::Done);
    assert!(result.verified);
    assert_eq!(result.turns.len(), 1);
    assert_eq!(*operations.lock().unwrap(), vec![JevOperation::Click]);
}

#[test]
fn goal_verifies_visible_static_text_without_an_action_ref() {
    let reply = DesktopResponse::ok(
        "snapshot",
        json!({
            "app": "Calculator",
            "window": {"title": "Calculator"},
            "tree": {
                "role": "window",
                "name": "Calculator",
                "children": [{
                    "role": "scrollarea",
                    "name": "Edit field",
                    "ref_id": "@s:e1",
                    "available_actions": ["Scroll"],
                    "children": [{
                        "role": "statictext",
                        "name": "\u{200e}12",
                        "value": "\u{200e}12"
                    }]
                }]
            }
        }),
    );
    let screen = parse_reply(&crate::Desktop::new(), "Calculator", None, None, reply).unwrap();
    let evidence = super::verify::verify(
        &screen,
        &[VisiblePredicate::NamePresent {
            name: "\u{200e}12".to_owned(),
        }],
    );
    assert!(super::verify::satisfied(&evidence));
    assert_eq!(screen.candidates.len(), 1);
}

#[test]
fn screen_change_detection_ignores_ephemeral_refs_and_sees_visible_text() {
    let mut before = clickable_screen();
    before.observed = vec![Candidate {
        role: "statictext".into(),
        name: Some("Old result".into()),
        ..Candidate::default()
    }];
    let mut after = before.clone();
    after.candidates[0].ref_id = "@new-snapshot:e1".into();
    assert_eq!(fingerprint(&before), fingerprint(&after));
    after.observed[0].name = Some("New result".into());
    assert_ne!(fingerprint(&before), fingerprint(&after));
}

#[tokio::test]
async fn goal_keeps_the_chosen_window_id_through_every_observation() {
    let runtime = runtime(vec![response("CLICK", 0.9, "1")]);
    let (inner, operations) = backend(3);
    {
        let mut screens = inner.screens.lock().unwrap();
        for screen in screens.iter_mut() {
            screen.window_id = Some("w-515619".into());
            screen.window = Some("desktop-e2e-noapproval.txt".into());
        }
        screens[2].candidates[0].name = Some("Finished".into());
    }
    let requested = Arc::new(Mutex::new(Vec::new()));
    let backend = WindowBoundBackend {
        inner,
        requested: Arc::clone(&requested),
    };
    let result = run_goal_with(
        backend,
        runtime,
        RunGoalRequest {
            app: "Spotify".into(),
            goal: "complete one action".into(),
            window: Some("desktop-e2e-noapproval.txt".into()),
            window_id: Some("w-515619".into()),
            allowed_operations: vec![JevOperation::Click],
            allowed_targets: vec!["Play First Song by Artist".into()],
            success: vec![VisiblePredicate::NamePresent {
                name: "Finished".into(),
            }],
            require_confirmations: false,
            ..RunGoalRequest::default()
        },
    )
    .await;
    let result: tinydesktop_bus::JevRunResult =
        serde_json::from_value(result.data.unwrap()).unwrap();
    assert!(result.verified);
    assert_eq!(*operations.lock().unwrap(), vec![JevOperation::Click]);
    assert_eq!(*requested.lock().unwrap(), vec![Some("w-515619".into()); 3]);
}

#[tokio::test]
async fn goal_rejects_a_snapshot_from_a_different_window_before_jev() {
    let (backend, operations) = backend(1);
    let result = run_goal_with(
        backend,
        runtime(Vec::new()),
        RunGoalRequest {
            app: "Spotify".into(),
            goal: "complete one action".into(),
            window_id: Some("w-515619".into()),
            ..RunGoalRequest::default()
        },
    )
    .await;
    let result: tinydesktop_bus::JevRunResult =
        serde_json::from_value(result.data.unwrap()).unwrap();
    assert_eq!(result.stop, JevStopReason::ScopeChanged);
    assert!(operations.lock().unwrap().is_empty());
}

#[tokio::test]
async fn missing_bound_window_does_not_fall_back_to_another_window() {
    let (inner, operations) = backend(1);
    let requested = Arc::new(Mutex::new(Vec::new()));
    let backend = WindowBoundBackend {
        inner,
        requested: Arc::clone(&requested),
    };
    let reply = run_goal_with(
        backend,
        runtime(Vec::new()),
        RunGoalRequest {
            app: "Spotify".into(),
            goal: "complete one action".into(),
            window_id: Some("w-missing".into()),
            ..RunGoalRequest::default()
        },
    )
    .await;
    assert_eq!(reply.error.unwrap().code, "WINDOW_NOT_FOUND");
    let requested = requested.lock().unwrap();
    assert!(requested.len() > 1 && requested.len() <= 20);
    assert!(
        requested
            .iter()
            .all(|window| window.as_deref() == Some("w-missing"))
    );
    assert!(operations.lock().unwrap().is_empty());
}

#[tokio::test]
async fn low_operation_probability_cannot_be_rescued_by_a_certain_target() {
    let runtime = runtime(vec![
        response_with("CLICK", 0.60, "1", 0.99, 0.0),
        response_with("CLICK", 0.60, "1", 0.99, 0.0),
    ]);
    let (backend, operations) = backend(2);
    let reply = run_goal_with(
        backend,
        runtime,
        RunGoalRequest {
            app: "Spotify".into(),
            goal: "click the play button".into(),
            ..RunGoalRequest::default()
        },
    )
    .await;
    let result: tinydesktop_bus::JevRunResult =
        serde_json::from_value(reply.data.unwrap()).unwrap();
    assert_eq!(result.stop, JevStopReason::LowConfidence);
    assert!(operations.lock().unwrap().is_empty());
}

#[tokio::test]
async fn fresh_target_change_prevents_a_mutation() {
    let runtime = runtime(vec![response("CLICK", 0.9, "1")]);
    let (backend, operations) = backend(2);
    backend.screens.lock().unwrap()[1].candidates[0].name = Some("Different button".into());
    let reply = run_goal_with(
        backend,
        runtime,
        RunGoalRequest {
            app: "Spotify".into(),
            goal: "click play".into(),
            ..RunGoalRequest::default()
        },
    )
    .await;
    let result: tinydesktop_bus::JevRunResult =
        serde_json::from_value(reply.data.unwrap()).unwrap();
    assert_eq!(result.stop, JevStopReason::StaleTarget);
    assert!(operations.lock().unwrap().is_empty());
}

#[tokio::test]
async fn jev_done_cannot_claim_completion_without_visible_evidence() {
    let (runtime, requests) = runtime_recording(vec![
        response("DONE", 0.9, "none"),
        response("DONE", 0.9, "none"),
    ]);
    let (backend, operations) = backend(2);
    let reply = run_goal_with(
        backend,
        runtime,
        RunGoalRequest {
            app: "Spotify".into(),
            goal: "finish the task".into(),
            success: vec![VisiblePredicate::NamePresent {
                name: "Finished".into(),
            }],
            ..RunGoalRequest::default()
        },
    )
    .await;
    let result: tinydesktop_bus::JevRunResult =
        serde_json::from_value(reply.data.unwrap()).unwrap();
    assert_eq!(result.stop, JevStopReason::VerificationFailed);
    assert!(!result.verified);
    assert!(operations.lock().unwrap().is_empty());
    assert!(
        requests.lock().unwrap()[1].state["recent_actions"][0]
            .as_str()
            .unwrap()
            .contains("required visible success conditions are not yet met")
    );
}

#[tokio::test]
async fn continuous_task_rejects_empty_or_blank_scope_before_jev() {
    let (backend, operations) = backend(1);
    let request = RunGoalRequest {
        app: "Spotify".into(),
        goal: "click play".into(),
        allowed_operations: vec![JevOperation::Click],
        allowed_targets: vec!["Play First Song by Artist".into()],
        success: vec![VisiblePredicate::NamePresent {
            name: "Finished".into(),
        }],
        require_confirmations: false,
        ..RunGoalRequest::default()
    };
    for bad in [
        RunGoalRequest {
            allowed_targets: vec![" ".into()],
            ..request.clone()
        },
        RunGoalRequest {
            success: vec![VisiblePredicate::ValueContains {
                name: "Document".into(),
                value: String::new(),
            }],
            ..request.clone()
        },
        RunGoalRequest {
            success: vec![VisiblePredicate::NameContains {
                fragment: String::new(),
                within: "Messages in chat with Alex Rivera".into(),
            }],
            ..request.clone()
        },
        RunGoalRequest {
            success: vec![VisiblePredicate::NameContains {
                fragment: "Hello".into(),
                within: "  ".into(),
            }],
            ..request.clone()
        },
        RunGoalRequest {
            allowed_operations: Vec::new(),
            ..request.clone()
        },
    ] {
        let reply = run_goal_with(backend.clone(), runtime(Vec::new()), bad).await;
        assert_eq!(reply.error.unwrap().code, "INVALID_TASK_SCOPE");
    }
    assert!(operations.lock().unwrap().is_empty());
}

async fn run_case(
    bodies: Vec<tinyjevclient::EvaluationResult>,
    screen_count: usize,
    max_steps: u32,
    max_model_calls: u32,
    goal: &str,
) -> tinydesktop_bus::JevRunResult {
    let runtime = runtime(bodies);
    let (backend, _) = backend(screen_count * 3);
    let reply = run_goal_with(
        backend,
        runtime,
        RunGoalRequest {
            app: "Spotify".to_owned(),
            goal: goal.to_owned(),
            max_steps,
            max_model_calls,
            ..RunGoalRequest::default()
        },
    )
    .await;
    serde_json::from_value(reply.data.expect("run returns data")).expect("result decodes")
}

#[tokio::test]
async fn goal_loop_reports_terminal_policy_outcomes() {
    let blocked = run_case(
        vec![response("BLOCKED", 0.9, "none")],
        1,
        3,
        3,
        "impossible",
    )
    .await;
    assert_eq!(blocked.stop, JevStopReason::Blocked);

    let confirmation = run_case(
        vec![response_with("CLICK", 0.9, "1", 0.9, 0.9)],
        1,
        3,
        3,
        "play the topmost song",
    )
    .await;
    assert_eq!(confirmation.stop, JevStopReason::ConfirmationRequired);

    let no_target = run_case(
        vec![response("CLICK", 0.9, "none")],
        1,
        3,
        3,
        "activate something",
    )
    .await;
    assert_eq!(no_target.stop, JevStopReason::LowConfidence);
}

#[tokio::test]
async fn approved_goal_action_reobserves_then_continues_once() {
    let (runtime, requests) = runtime_recording(vec![
        response_with("CLICK", 0.9, "1", 0.9, 0.9),
        response("DONE", 0.9, "none"),
    ]);
    let (backend, operations) = backend(4);
    {
        let mut screens = backend.screens.lock().unwrap();
        for index in [2, 3] {
            screens[index].candidates[0].name = Some("Sent First Song by Artist".to_owned());
        }
    }
    let request = RunGoalRequest {
        app: "Spotify".to_owned(),
        goal: "send the selected item".to_owned(),
        max_steps: 3,
        max_model_calls: 3,
        ..RunGoalRequest::default()
    };
    let stopped = run_goal_with(backend.clone(), runtime.clone(), request).await;
    let stopped: tinydesktop_bus::JevRunResult =
        serde_json::from_value(stopped.data.unwrap()).unwrap();
    assert_eq!(stopped.stop, JevStopReason::ConfirmationRequired);
    assert!(operations.lock().unwrap().is_empty());
    let id = stopped.confirmation_id.expect("confirmation handle");
    let resumed = run_goal_with(
        backend.clone(),
        runtime.clone(),
        RunGoalRequest {
            continuation: Some(GoalContinuation {
                id: id.clone(),
                approve: true,
            }),
            ..RunGoalRequest::default()
        },
    )
    .await;
    let resumed: tinydesktop_bus::JevRunResult =
        serde_json::from_value(resumed.data.unwrap()).unwrap();
    assert_eq!(resumed.stop, JevStopReason::Done);
    assert_eq!(resumed.turns.len(), 1);
    assert!(resumed.turns[0].changed);
    assert_eq!(resumed.metrics.calls, 2);
    assert!(
        requests.lock().unwrap()[1].state["recent_actions"][0]
            .as_str()
            .unwrap()
            .contains("changed=true")
    );
    assert_eq!(*operations.lock().unwrap(), vec![JevOperation::Click]);
    let replay = run_goal_with(
        backend,
        runtime,
        RunGoalRequest {
            continuation: Some(GoalContinuation { id, approve: true }),
            ..RunGoalRequest::default()
        },
    )
    .await;
    assert_eq!(replay.error.unwrap().code, "CONFIRMATION_EXPIRED");
}

#[tokio::test]
async fn continuation_uses_prepared_named_text_instead_of_an_empty_value() {
    let first = evaluation(json!({
        "model":"typesafe/jev-1.13-20260917",
        "answers":{
            "operation":{"type":"choice","choice":"TYPE_TEXT","confidence":0.9,
                "probabilities":{"TYPE_TEXT":0.95,"DONE":0.03,"BLOCKED":0.02}},
            "type_text_target":{"type":"choice","choice":"1","confidence":0.9,
                "probabilities":{"1":0.95,"none":0.05}},
            "destructive":{"type":"noul","noul":0.9}
        }, "usage":{"input_tokens":10,"output_tokens":2}
    }));
    let runtime = runtime(vec![first, response("DONE", 0.9, "none")]);
    let (inner, _) = backend(4);
    for screen in inner.screens.lock().unwrap().iter_mut() {
        screen.candidates[0].role = "text field".into();
        screen.candidates[0].name = Some("Document".into());
        screen.candidates[0].available_actions = vec!["SetValue".into()];
    }
    let values = Arc::new(Mutex::new(Vec::new()));
    let backend = RecordingTextBackend {
        inner,
        values: Arc::clone(&values),
    };
    let stopped = run_goal_with(
        backend.clone(),
        runtime.clone(),
        RunGoalRequest {
            app: "Spotify".into(),
            goal: "fill the document".into(),
            text_slots: BTreeMap::from([("Document".into(), "marker".into())]),
            ..RunGoalRequest::default()
        },
    )
    .await;
    let stopped: tinydesktop_bus::JevRunResult =
        serde_json::from_value(stopped.data.unwrap()).unwrap();
    assert_eq!(stopped.stop, JevStopReason::ConfirmationRequired);
    let resumed = run_goal_with(
        backend,
        runtime,
        RunGoalRequest {
            continuation: Some(GoalContinuation {
                id: stopped.confirmation_id.unwrap(),
                approve: true,
            }),
            ..RunGoalRequest::default()
        },
    )
    .await;
    let resumed: tinydesktop_bus::JevRunResult =
        serde_json::from_value(resumed.data.unwrap()).unwrap();
    assert_eq!(resumed.stop, JevStopReason::Done);
    assert_eq!(*values.lock().unwrap(), vec![Some("marker".into())]);
}

#[tokio::test]
async fn continuation_never_replays_after_post_action_observation_is_lost() {
    let runtime = runtime(vec![response_with("CLICK", 0.9, "1", 0.9, 0.9)]);
    let (backend, operations) = backend(2);
    let stopped = run_goal_with(
        backend.clone(),
        runtime.clone(),
        RunGoalRequest {
            app: "Spotify".into(),
            goal: "send the selected item".into(),
            ..RunGoalRequest::default()
        },
    )
    .await;
    let stopped: tinydesktop_bus::JevRunResult =
        serde_json::from_value(stopped.data.unwrap()).unwrap();
    let result = run_goal_with(
        backend,
        runtime,
        RunGoalRequest {
            continuation: Some(GoalContinuation {
                id: stopped.confirmation_id.unwrap(),
                approve: true,
            }),
            ..RunGoalRequest::default()
        },
    )
    .await;
    let result: tinydesktop_bus::JevRunResult =
        serde_json::from_value(result.data.unwrap()).unwrap();
    assert_eq!(result.stop, JevStopReason::ActionUncertain);
    assert_eq!(result.turns.len(), 1);
    assert!(result.turns[0].ok);
    assert_eq!(*operations.lock().unwrap(), vec![JevOperation::Click]);
}

#[tokio::test]
async fn confirmation_wait_consumes_the_original_elapsed_budget() {
    let runtime = runtime(vec![response_with("CLICK", 0.9, "1", 0.9, 0.9)]);
    let (backend, operations) = backend(1);
    let stopped = run_goal_with(
        backend.clone(),
        runtime.clone(),
        RunGoalRequest {
            app: "Spotify".into(),
            goal: "send the selected item".into(),
            max_elapsed_ms: 50,
            ..RunGoalRequest::default()
        },
    )
    .await;
    let stopped: tinydesktop_bus::JevRunResult =
        serde_json::from_value(stopped.data.unwrap()).unwrap();
    let id = stopped.confirmation_id.unwrap();
    runtime
        .pending
        .lock()
        .unwrap()
        .get_mut(&id)
        .unwrap()
        .started = Instant::now()
        .checked_sub(Duration::from_millis(100))
        .unwrap();
    let result = run_goal_with(
        backend,
        runtime,
        RunGoalRequest {
            continuation: Some(GoalContinuation { id, approve: true }),
            ..RunGoalRequest::default()
        },
    )
    .await;
    let result: tinydesktop_bus::JevRunResult =
        serde_json::from_value(result.data.unwrap()).unwrap();
    assert_eq!(result.stop, JevStopReason::TimeBudget);
    assert!(operations.lock().unwrap().is_empty());
}

#[tokio::test]
async fn declined_and_stale_goal_actions_never_execute() {
    let runtime = runtime(vec![
        response_with("CLICK", 0.9, "1", 0.9, 0.9),
        response_with("CLICK", 0.9, "1", 0.9, 0.9),
    ]);
    let (backend, operations) = backend(4);
    let request = RunGoalRequest {
        app: "Spotify".to_owned(),
        goal: "send the selected item".to_owned(),
        ..RunGoalRequest::default()
    };
    let declined: tinydesktop_bus::JevRunResult = serde_json::from_value(
        run_goal_with(backend.clone(), runtime.clone(), request.clone())
            .await
            .data
            .unwrap(),
    )
    .unwrap();
    let decline: tinydesktop_bus::JevRunResult = serde_json::from_value(
        run_goal_with(
            backend.clone(),
            runtime.clone(),
            RunGoalRequest {
                continuation: Some(GoalContinuation {
                    id: declined.confirmation_id.unwrap(),
                    approve: false,
                }),
                ..RunGoalRequest::default()
            },
        )
        .await
        .data
        .unwrap(),
    )
    .unwrap();
    assert_eq!(decline.stop, JevStopReason::Cancelled);

    let stopped: tinydesktop_bus::JevRunResult = serde_json::from_value(
        run_goal_with(backend.clone(), runtime.clone(), request)
            .await
            .data
            .unwrap(),
    )
    .unwrap();
    let mut changed = clickable_screen();
    changed.candidates[0].name = Some("Different destructive button".to_owned());
    backend.screens.lock().unwrap().push_front(changed);
    let stale: tinydesktop_bus::JevRunResult = serde_json::from_value(
        run_goal_with(
            backend,
            runtime,
            RunGoalRequest {
                continuation: Some(GoalContinuation {
                    id: stopped.confirmation_id.unwrap(),
                    approve: true,
                }),
                ..RunGoalRequest::default()
            },
        )
        .await
        .data
        .unwrap(),
    )
    .unwrap();
    assert_eq!(stale.stop, JevStopReason::StaleTarget);
    assert!(operations.lock().unwrap().is_empty());
}

#[tokio::test]
async fn expired_confirmation_handle_never_executes() {
    let runtime = runtime(vec![response_with("CLICK", 0.9, "1", 0.9, 0.9)]);
    let (backend, operations) = backend(1);
    let stopped: tinydesktop_bus::JevRunResult = serde_json::from_value(
        run_goal_with(
            backend.clone(),
            runtime.clone(),
            RunGoalRequest {
                app: "Spotify".to_owned(),
                goal: "send the selected item".to_owned(),
                ..RunGoalRequest::default()
            },
        )
        .await
        .data
        .unwrap(),
    )
    .unwrap();
    let id = stopped.confirmation_id.unwrap();
    runtime
        .pending
        .lock()
        .unwrap()
        .get_mut(&id)
        .unwrap()
        .created = Instant::now()
        .checked_sub(Duration::from_secs(601))
        .unwrap();
    let reply = run_goal_with(
        backend,
        runtime,
        RunGoalRequest {
            continuation: Some(GoalContinuation { id, approve: true }),
            ..RunGoalRequest::default()
        },
    )
    .await;
    assert_eq!(reply.error.unwrap().code, "CONFIRMATION_EXPIRED");
    assert!(operations.lock().unwrap().is_empty());
}

#[tokio::test]
async fn goal_loop_enforces_action_model_and_stall_budgets() {
    let action_budget = run_case(
        vec![response("CLICK", 0.9, "1")],
        2,
        1,
        3,
        "play the topmost song",
    )
    .await;
    assert_eq!(action_budget.stop, JevStopReason::ActionBudget);

    let model_budget = run_case(
        vec![response("CLICK", 0.9, "1")],
        2,
        4,
        1,
        "play the topmost song",
    )
    .await;
    assert_eq!(model_budget.stop, JevStopReason::ModelBudget);

    let stalled = run_case(
        vec![
            response("CLICK", 0.9, "1"),
            response("CLICK", 0.9, "1"),
            response("CLICK", 0.9, "1"),
        ],
        6,
        4,
        4,
        "play the topmost song",
    )
    .await;
    assert_eq!(stalled.stop, JevStopReason::Stalled);
}

#[tokio::test]
async fn goal_loop_preserves_failed_actions_and_post_action_observation_failures() {
    let failed_runtime = runtime(vec![response("CLICK", 0.9, "1")]);
    let operations = Arc::new(Mutex::new(Vec::new()));
    let failed_backend = FakeBackend {
        screens: Arc::new(Mutex::new(VecDeque::from([
            clickable_screen(),
            clickable_screen(),
        ]))),
        operations: Arc::clone(&operations),
        fail_execute: true,
    };
    let failed = run_goal_with(
        failed_backend,
        failed_runtime,
        RunGoalRequest {
            app: "Spotify".to_owned(),
            goal: "play the topmost song".to_owned(),
            ..RunGoalRequest::default()
        },
    )
    .await;
    let failed: tinydesktop_bus::JevRunResult =
        serde_json::from_value(failed.data.expect("failed run data")).expect("result decodes");
    assert_eq!(failed.stop, JevStopReason::ActionUncertain);
    assert_eq!(failed.turns.len(), 1);
    assert!(!failed.turns[0].ok);

    let observation_runtime = runtime(vec![response("CLICK", 0.9, "1")]);
    let (backend, _) = backend(2);
    let lost_screen = run_goal_with(
        backend,
        observation_runtime,
        RunGoalRequest {
            app: "Spotify".to_owned(),
            goal: "play the topmost song".to_owned(),
            ..RunGoalRequest::default()
        },
    )
    .await;
    let lost_screen: tinydesktop_bus::JevRunResult =
        serde_json::from_value(lost_screen.data.expect("lost screen data"))
            .expect("result decodes");
    assert_eq!(lost_screen.stop, JevStopReason::ActionUncertain);
    assert_eq!(lost_screen.turns.len(), 1);
}

#[test]
fn screen_parsing_filters_disabled_nodes_and_builds_descriptions() {
    let reply = DesktopResponse::ok(
        "snapshot",
        json!({
            "app": "Spotify", "window": {"title": "Liked Songs"},
            "tree": {"role": "window", "children": [
                {"ref_id": "@s:e1", "role": "button", "name": "Play First Song by Artist", "available_actions": ["Click"], "children_count": 4},
                {"ref_id": "@s:e2", "role": "button", "name": "Disabled", "available_actions": ["Click"], "states": ["disabled"]}
            ]}
        }),
    );
    let screen = parse_reply(
        &crate::Desktop::new(),
        "Spotify",
        None,
        Some("@s:root"),
        reply,
    )
    .expect("synthetic snapshot parses");
    assert_eq!(screen.candidates.len(), 1);
    assert_eq!(
        describe(&screen.candidates[0], false)["untrusted_accessibility_data"]["contains"],
        json!(4)
    );
    let before = fingerprint(&screen);
    assert_eq!(before.len(), 16);
    let mut changed = screen.clone();
    changed
        .observed
        .iter_mut()
        .find(|node| node.name.as_deref() == Some("Play First Song by Artist"))
        .unwrap()
        .name = Some("Pause First Song by Artist".into());
    assert_ne!(before, fingerprint(&changed));
}

#[test]
fn textedit_native_identifier_survives_snapshot_and_is_offered_to_jev() {
    let reply = DesktopResponse::ok(
        "snapshot",
        json!({
            "app": "TextEdit", "window": {"title": "Untitled"},
            "tree": {"role": "window", "children": [{
                "ref_id": "@s:e1", "role": "text field", "name": null,
                "description": null,
                "native_id": {"kind": "ax_identifier", "value": "First Text View"},
                "available_actions": ["SetValue"], "value": ""
            }]}
        }),
    );
    let screen = parse_reply(&crate::Desktop::new(), "TextEdit", None, None, reply).unwrap();
    let field = &screen.candidates[0];
    assert_eq!(field.native_id.as_ref().unwrap().value, "First Text View");
    assert_eq!(
        target_payload(field).name.as_deref(),
        Some("First Text View")
    );
    assert!(
        describe(field, false)["untrusted_accessibility_data"]["what"]
            .as_str()
            .unwrap()
            .contains("First Text View")
    );
}

#[test]
fn changed_native_identifier_fails_fresh_target_validation() {
    let mut before = clickable_screen();
    before.candidates[0].name = None;
    before.candidates[0].native_id = Some(NativeId {
        kind: "ax_identifier".into(),
        value: "First Text View".into(),
    });
    let mut after = before.clone();
    after.candidates[0].native_id.as_mut().unwrap().value = "Second Text View".into();
    assert!(!same_target(
        &before,
        &after,
        &before.candidates[0],
        &after.candidates[0],
        JevOperation::Click
    ));
}

#[tokio::test]
async fn unlabeled_textedit_field_completes_scoped_text_task_in_one_call() {
    let text_choice = evaluation(json!({
        "model":"typesafe/jev-1.13-20260917",
        "answers":{
            "operation":{"type":"choice","choice":"TYPE_TEXT","confidence":0.9,
                "probabilities":{"TYPE_TEXT":0.95,"DONE":0.03,"BLOCKED":0.02}},
            "type_text_target":{"type":"choice","choice":"1","confidence":0.9,
                "probabilities":{"1":0.95,"none":0.05}},
            "destructive":{"type":"noul","noul":0.05}
        }, "usage":{"input_tokens":10,"output_tokens":2}
    }));
    let runtime = runtime(vec![text_choice]);
    let (inner, operations) = backend(3);
    {
        let mut screens = inner.screens.lock().unwrap();
        for screen in screens.iter_mut() {
            screen.app = "TextEdit".into();
            screen.window = Some("Untitled".into());
            screen.candidates[0].role = "text field".into();
            screen.candidates[0].name = None;
            screen.candidates[0].native_id = Some(NativeId {
                kind: "ax_identifier".into(),
                value: "First Text View".into(),
            });
            screen.candidates[0].available_actions = vec!["SetValue".into()];
            screen.candidates[0].value = Some(json!(""));
        }
        screens[2].candidates[0].value = Some(json!("marker"));
    }
    let values = Arc::new(Mutex::new(Vec::new()));
    let backend = RecordingTextBackend {
        inner,
        values: Arc::clone(&values),
    };
    let result = run_goal_with(
        backend,
        runtime,
        RunGoalRequest {
            app: "TextEdit".into(),
            goal: "place marker in First Text View".into(),
            window: Some("Untitled".into()),
            allowed_operations: vec![JevOperation::TypeText],
            allowed_targets: vec!["First Text View".into()],
            text_slots: BTreeMap::from([("First Text View".into(), "marker".into())]),
            success: vec![VisiblePredicate::ValueContains {
                name: "First Text View".into(),
                value: "marker".into(),
            }],
            require_confirmations: false,
            ..RunGoalRequest::default()
        },
    )
    .await;
    let result: tinydesktop_bus::JevRunResult =
        serde_json::from_value(result.data.unwrap()).unwrap();
    assert_eq!(result.stop, JevStopReason::Done);
    assert!(result.verified);
    assert_eq!(result.turns.len(), 1);
    assert_eq!(
        result.turns[0].target.as_ref().unwrap().name.as_deref(),
        Some("First Text View")
    );
    assert_eq!(*operations.lock().unwrap(), vec![JevOperation::TypeText]);
    assert_eq!(*values.lock().unwrap(), vec![Some("marker".into())]);
}

#[test]
fn action_space_and_requests_cover_every_supported_capability() {
    let screen = Screen {
        app: "App".to_owned(),
        window: Some("Window".to_owned()),
        window_id: None,
        surface: "window".to_owned(),
        root: None,
        candidates: vec![Candidate {
            ref_id: "@s:e1".to_owned(),
            role: "control".to_owned(),
            name: Some("Everything".to_owned()),
            value: Some(json!("held")),
            states: vec!["checked".to_owned()],
            available_actions: vec![
                "Click".to_owned(),
                "SetValue".to_owned(),
                "Toggle".to_owned(),
                "Expand".to_owned(),
                "Collapse".to_owned(),
                "Scroll".to_owned(),
            ],
            children_count: Some(3),
            ..Candidate::default()
        }],
        observed: Vec::new(),
    };
    let space = action_space(&screen, true);
    for operation in [
        "CLICK",
        "TYPE_TEXT",
        "CHECK",
        "UNCHECK",
        "EXPAND",
        "COLLAPSE",
        "SCROLL",
        "DRILL",
    ] {
        assert!(space.targets.contains_key(operation), "missing {operation}");
    }
    let evaluation = request("jev-latest", "change it", &screen, &space, &[], true);
    assert!(evaluation.questions.contains_key("type_text_target"));
    let rerank = rerank_request(
        "jev-latest",
        "change it",
        &screen,
        "CLICK",
        space.targets.get("CLICK").expect("click targets"),
        true,
    );
    assert_eq!(rerank.questions.len(), 1);
}

#[test]
fn answer_helpers_cover_terminal_missing_and_shortlist_paths() {
    let screen = clickable_screen();
    let space = action_space(&screen, false);
    let answer = Answer::Choice(ChoiceAnswer {
        choice: "1".to_owned(),
        probabilities: BTreeMap::from([("1".to_owned(), 0.8), ("none".to_owned(), 0.2)]),
        confidence: 0.2,
    });
    assert!(target(&space, "CLICK", Some(&answer)).is_some());
    assert_eq!(shortlist(&space, "CLICK", Some(&answer)).len(), 1);
    assert!(shortlist(&space, "MISSING", Some(&answer)).is_empty());
    assert!(choice(None).is_none());
    assert!((noul(None) - 1.0).abs() < f64::EPSILON);
    for (wire, operation) in [
        ("CLICK", JevOperation::Click),
        ("TYPE_TEXT", JevOperation::TypeText),
        ("CHECK", JevOperation::Check),
        ("UNCHECK", JevOperation::Uncheck),
        ("EXPAND", JevOperation::Expand),
        ("COLLAPSE", JevOperation::Collapse),
        ("SCROLL", JevOperation::Scroll),
        ("DRILL", JevOperation::Drill),
        ("WIDEN", JevOperation::Widen),
        ("WAIT", JevOperation::Wait),
        ("DONE", JevOperation::Done),
        ("BLOCKED", JevOperation::Blocked),
    ] {
        assert_eq!(parse_operation(wire), Some(operation));
    }
    assert_eq!(parse_operation("NOPE"), None);
}

#[test]
fn screen_helpers_cover_overlay_values_bounds_and_failed_observation() {
    let reply = DesktopResponse::ok(
        "snapshot",
        json!({
            "app": "App",
            "tree": {"role": "sheet", "children": [{
                "ref_id": "@s:e1", "role": "textfield", "value": "private",
                "available_actions": ["SetValue"], "states": ["focused"],
                "bounds": {"x": 1.0, "y": 2.0}
            }]}
        }),
    );
    let screen = parse_reply(&crate::Desktop::new(), "App", None, None, reply)
        .expect("original synthetic overlay remains usable");
    let with_values = describe(&screen.candidates[0], true);
    assert!(
        with_values["untrusted_accessibility_data"]
            .get("holds")
            .is_some()
    );
    assert!(
        with_values["untrusted_accessibility_data"]
            .get("state")
            .is_some()
    );
    let unnamed = Candidate {
        role: "button".to_owned(),
        bounds: Some(json!({"x": 1.0, "y": 2.0})),
        ..Candidate::default()
    };
    assert!(
        describe(&unnamed, false)["untrusted_accessibility_data"]
            .get("bounds")
            .is_some()
    );

    let failed = observe(
        &crate::Desktop::new(),
        "__tinydesktop_missing__",
        None,
        None,
    )
    .expect_err("missing app fails");
    assert!(!failed.ok);
    assert!(
        observe(
            &crate::Desktop::new(),
            "__tinydesktop_missing__",
            None,
            Some("@s:e1")
        )
        .is_err()
    );

    for role in ["alert", "menu", "popover"] {
        let screen = parse_reply(
            &crate::Desktop::new(),
            "__tinydesktop_missing__",
            None,
            None,
            DesktopResponse::ok(
                "snapshot",
                json!({"app": "App", "tree": {"role": role, "children": []}}),
            ),
        )
        .expect("synthetic overlay remains usable");
        assert_eq!(screen.surface, "window");
    }

    let failed_reply = DesktopResponse::err(
        "snapshot",
        tinydesktop_bus::DesktopError::new("FAIL", "failed"),
    );
    assert!(
        parse_reply(
            &crate::Desktop::new(),
            "App",
            None,
            Some("@s:root"),
            failed_reply
        )
        .is_err()
    );
    let no_data = DesktopResponse {
        version: tinydesktop_bus::ENVELOPE_VERSION.to_owned(),
        ok: true,
        command: "snapshot".to_owned(),
        data: None,
        error: None,
    };
    assert!(
        parse_reply(
            &crate::Desktop::new(),
            "App",
            None,
            Some("@s:root"),
            no_data
        )
        .is_err()
    );
}

#[test]
fn accessibility_tree_traversal_is_bounded() {
    let mut deep = json!({
        "ref_id": "@s:deep",
        "role": "button",
        "available_actions": ["Click"]
    });
    for _ in 0..66 {
        deep = json!({"role": "group", "children": [deep]});
    }
    let bounded = parse_reply(
        &crate::Desktop::new(),
        "App",
        None,
        Some("@s:root"),
        DesktopResponse::ok("snapshot", json!({"app": "App", "tree": deep})),
    )
    .expect("deep tree is bounded");
    assert!(bounded.candidates.is_empty());

    let many = (0..4_100)
        .map(|index| json!({"role": "group", "name": format!("node-{index}")}))
        .collect::<Vec<_>>();
    let bounded = parse_reply(
        &crate::Desktop::new(),
        "App",
        None,
        Some("@s:root"),
        DesktopResponse::ok(
            "snapshot",
            json!({"app": "App", "tree": {"role": "window", "children": many}}),
        ),
    )
    .expect("wide tree is bounded");
    assert!(bounded.candidates.is_empty());
}

#[test]
fn runtime_configuration_covers_all_providers_and_rejects_empty_keys() {
    for provider in [
        JevProvider::TypeSafe,
        JevProvider::OpenRouter,
        JevProvider::TinyHumansOpenRouter,
        JevProvider::OpenJev,
    ] {
        let mut request = JevConfig::new("key");
        request.provider = provider;
        request.model = Some("jev-test".to_owned());
        request.timeout_ms = Some(500);
        request.max_retries = Some(0);
        request.endpoint_url = Some("http://127.0.0.1:1/decisions".to_owned());
        let runtime = JevRuntime::configure(&request).expect("configuration is valid");
        assert_eq!(runtime.configuration.provider, provider);
    }
    assert!(JevRuntime::configure(&JevConfig::default()).is_err());
    let mut untrusted = JevConfig::new("key");
    untrusted.provider = JevProvider::OpenRouter;
    untrusted.endpoint_url = Some("https://attacker.example/decisions".to_owned());
    assert!(JevRuntime::configure(&untrusted).is_err());
}

#[test]
fn desktop_execution_dispatches_every_closed_operation_without_panicking() {
    let desktop = crate::Desktop::new();
    let candidate = Candidate {
        ref_id: String::new(),
        ..Candidate::default()
    };
    for operation in [
        JevOperation::Click,
        JevOperation::TypeText,
        JevOperation::Check,
        JevOperation::Uncheck,
        JevOperation::Expand,
        JevOperation::Collapse,
        JevOperation::Scroll,
        JevOperation::Wait,
        JevOperation::Drill,
        JevOperation::Widen,
        JevOperation::Done,
        JevOperation::Blocked,
    ] {
        let reply = execute_desktop(
            &desktop,
            operation,
            Some(&candidate),
            Some("text".to_owned()),
        );
        assert!(!reply.command.is_empty());
    }
}

#[tokio::test]
async fn one_step_resolution_and_public_wrappers_cover_success_and_observation_failure() {
    let runtime = runtime(vec![response("CLICK", 0.9, "1")]);
    assert!(format!("{runtime:?}").contains("JevRuntime"));
    let (backend, _) = backend(1);
    let reply = resolve_intent_with(
        backend,
        runtime.clone(),
        tinydesktop_bus::ResolveIntentRequest {
            app: "Spotify".to_owned(),
            intent: "play the topmost song".to_owned(),
            execute: false,
            ..tinydesktop_bus::ResolveIntentRequest::default()
        },
    )
    .await;
    assert!(reply.ok);

    let missing = tinydesktop_bus::ResolveIntentRequest {
        app: "__tinydesktop_missing__".to_owned(),
        intent: "click".to_owned(),
        ..tinydesktop_bus::ResolveIntentRequest::default()
    };
    assert!(
        !resolve_intent(crate::Desktop::new(), runtime.clone(), missing)
            .await
            .ok
    );
    assert!(
        !run_goal(
            crate::Desktop::new(),
            runtime,
            RunGoalRequest {
                app: "__tinydesktop_missing__".to_owned(),
                goal: "finish".to_owned(),
                ..RunGoalRequest::default()
            }
        )
        .await
        .ok
    );
}

#[tokio::test]
async fn one_step_resolution_reranks_a_close_target_shortlist() {
    let first = evaluation(json!({
        "model": "typesafe/jev-1.13-20260917",
        "answers": {
            "operation": {"type": "choice", "choice": "CLICK", "confidence": 0.4,
                "probabilities": {"CLICK": 0.9, "WAIT": 0.033_333_333_333, "DONE": 0.033_333_333_333, "BLOCKED": 0.033_333_333_334}},
            "click_target": {"type": "choice", "choice": "1", "confidence": 0.3,
                "probabilities": {"1": 0.5, "2": 0.4, "none": 0.1}},
            "destructive": {"type": "noul", "noul": 0.05}
        },
        "usage": {}
    }));
    let reranked = evaluation(json!({
        "model": "typesafe/jev-1.13-20260917",
        "answers": {
            "target": {"type": "choice", "choice": "2", "confidence": 0.6,
                "probabilities": {"1": 0.1, "2": 0.85, "none": 0.05}}
        },
        "usage": {}
    }));
    let runtime = runtime(vec![first, reranked]);
    let backend = FakeBackend {
        screens: Arc::new(Mutex::new(VecDeque::from([two_candidate_screen()]))),
        operations: Arc::new(Mutex::new(Vec::new())),
        fail_execute: false,
    };
    let reply = resolve_intent_with(
        backend,
        runtime,
        tinydesktop_bus::ResolveIntentRequest {
            app: "Spotify".to_owned(),
            intent: "activate the second song".to_owned(),
            ..tinydesktop_bus::ResolveIntentRequest::default()
        },
    )
    .await;
    assert!(reply.ok);
    let decision: tinydesktop_bus::JevDecision =
        serde_json::from_value(reply.data.expect("decision data")).expect("decision decodes");
    assert_eq!(decision.target.expect("target").ref_id, "@s1:e2");
}

#[test]
fn response_helpers_classify_provider_failures_and_policy_reasons() {
    for (error, code) in [
        (tinyjevclient::Error::Authentication, "JEV_AUTHENTICATION"),
        (tinyjevclient::Error::RateLimited, "JEV_RATE_LIMITED"),
        (tinyjevclient::Error::Timeout, "JEV_TIMEOUT"),
        (
            tinyjevclient::Error::InvalidResponse {
                reason: "bad".to_owned(),
            },
            "JEV_INVALID_RESPONSE",
        ),
        (
            tinyjevclient::Error::HttpStatus { status: 500 },
            "JEV_PROVIDER_FAILED",
        ),
    ] {
        let failure = tinyjevclient::EvaluationFailure {
            error,
            attempts: 1,
            latency: Duration::ZERO,
        };
        assert_eq!(
            provider_error(&failure)
                .error
                .as_ref()
                .expect("provider error payload")
                .code,
            code
        );
    }
    for decision in [
        JevDecisionKind::Act,
        JevDecisionKind::ConfirmationRequired,
        JevDecisionKind::Abstain,
        JevDecisionKind::NeedsText,
        JevDecisionKind::Done,
        JevDecisionKind::Blocked,
    ] {
        assert!(!reason(decision, 0.5, 0.6).is_empty());
    }
    let target = target_payload(&Candidate {
        ref_id: "@s:e1".to_owned(),
        role: "button".to_owned(),
        description: Some("described".to_owned()),
        ..Candidate::default()
    });
    assert_eq!(target.name.as_deref(), Some("described"));
    assert!(!internal_error("broken").ok);
    assert!(agent_response("test", &json!({"ok": true})).ok);
}
