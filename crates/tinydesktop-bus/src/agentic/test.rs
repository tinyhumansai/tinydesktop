//! Wire-form tests for Jev desktop-control payloads.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::{
    GoalContinuation, JevConfig, JevDecisionKind, JevOperation, JevProvider, JevStopReason,
    ResolveIntentRequest, RunGoalRequest, VisiblePredicate,
};
use serde_json::json;

#[test]
fn configuration_serializes_the_key_but_never_debug_prints_it() {
    let mut request = JevConfig::new("openrouter-secret");
    request.provider = JevProvider::OpenRouter;
    request.endpoint_url = Some("https://openrouter.ai/api/alpha/decisions".into());
    request.sdk_name = Some("openhuman".into());
    let value = serde_json::to_value(&request).expect("configuration serializes");

    assert_eq!(value["api_key"], json!("openrouter-secret"));
    assert!(!format!("{request:?}").contains("openrouter-secret"));
    assert_eq!(value["sdk_name"], json!("openhuman"));
}

#[test]
fn every_agentic_enum_pins_its_wire_spelling() {
    assert_eq!(
        serde_json::to_value(JevProvider::TypeSafe).unwrap(),
        json!("type_safe")
    );
    assert_eq!(
        serde_json::to_value(JevProvider::OpenRouter).unwrap(),
        json!("open_router")
    );
    assert_eq!(
        serde_json::to_value(JevProvider::TinyHumansOpenRouter).unwrap(),
        json!("tiny_humans_open_router")
    );
    assert_eq!(
        serde_json::to_value(JevProvider::OpenJev).unwrap(),
        json!("open_jev")
    );
    for (operation, wire) in [
        (JevOperation::Click, "CLICK"),
        (JevOperation::TypeText, "TYPE_TEXT"),
        (JevOperation::Check, "CHECK"),
        (JevOperation::Uncheck, "UNCHECK"),
        (JevOperation::Expand, "EXPAND"),
        (JevOperation::Collapse, "COLLAPSE"),
        (JevOperation::Scroll, "SCROLL"),
        (JevOperation::Drill, "DRILL"),
        (JevOperation::Widen, "WIDEN"),
        (JevOperation::Wait, "WAIT"),
        (JevOperation::Done, "DONE"),
        (JevOperation::Blocked, "BLOCKED"),
    ] {
        assert_eq!(serde_json::to_value(operation).unwrap(), json!(wire));
    }
    assert_eq!(
        serde_json::to_value(JevDecisionKind::ConfirmationRequired).unwrap(),
        json!("confirmation_required")
    );
    assert_eq!(
        serde_json::to_value(JevStopReason::ActionFailed).unwrap(),
        json!("action_failed")
    );
}

#[test]
fn agentic_requests_default_to_not_sharing_field_values() {
    let resolve: ResolveIntentRequest = serde_json::from_value(json!({
        "app": "Spotify",
        "intent": "open Search"
    }))
    .expect("resolve request decodes");
    let run: RunGoalRequest = serde_json::from_value(json!({
        "app": "Spotify",
        "goal": "open Search"
    }))
    .expect("run request decodes");

    assert!(!resolve.include_values);
    assert!(!run.include_values);
    assert!(run.window_id.is_none());
    assert_eq!((run.max_steps, run.max_model_calls), (40, 80));
    assert!(run.continuation.is_none());
}

#[test]
fn confirmation_payload_has_explicit_approval_and_one_use_handle() {
    let request: RunGoalRequest = serde_json::from_value(json!({
        "continuation": {"id": "opaque-handle", "approve": false}
    }))
    .expect("continuation decodes");
    assert_eq!(
        request.continuation,
        Some(GoalContinuation {
            id: "opaque-handle".to_owned(),
            approve: false,
        })
    );
    assert_eq!(
        serde_json::to_value(JevStopReason::Cancelled).unwrap(),
        json!("cancelled")
    );
    assert_eq!(
        serde_json::to_value(JevStopReason::StaleTarget).unwrap(),
        json!("stale_target")
    );
}

#[test]
fn scoped_goal_additions_are_backward_compatible_and_have_stable_wire_names() {
    let old: RunGoalRequest =
        serde_json::from_value(json!({"app":"TextEdit","goal":"type"})).unwrap();
    assert!(old.require_confirmations);
    assert!(old.success.is_empty());
    assert_eq!(old.max_elapsed_ms, 120_000);
    let scoped: RunGoalRequest = serde_json::from_value(json!({
        "app":"TextEdit", "goal":"type", "window":"Untitled",
        "window_id":"w-515619",
        "allowed_operations":["TYPE_TEXT"],
        "allowed_targets":["Document"],
        "text_slots":{"Document":"marker"},
        "success":[{"kind":"value_contains","name":"Document","value":"marker"}],
        "max_elapsed_ms":30000,
        "require_confirmations":false
    }))
    .unwrap();
    assert_eq!(scoped.allowed_operations, vec![JevOperation::TypeText]);
    assert_eq!(scoped.window_id.as_deref(), Some("w-515619"));
    assert_eq!(
        scoped.success,
        vec![VisiblePredicate::ValueContains {
            name: "Document".into(),
            value: "marker".into()
        }]
    );
    assert!(!scoped.require_confirmations);
    assert_eq!(
        serde_json::to_value(&scoped).unwrap()["success"][0]["kind"],
        json!("value_contains")
    );
}

#[test]
fn bounded_snapshot_cannot_claim_an_element_is_absent() {
    let unsupported = serde_json::from_value::<RunGoalRequest>(json!({
        "app": "TextEdit",
        "goal": "close dialog",
        "success": [{"kind": "name_absent", "name": "Dialog"}]
    }));
    assert!(unsupported.is_err());
}

#[test]
fn contained_name_fragment_has_a_stable_wire_shape() {
    let predicate = VisiblePredicate::NameContains {
        fragment: "Your message, Hello from OpenHuman".into(),
        within: "Messages in chat with Alex Rivera".into(),
    };
    let wire = json!({
        "kind": "name_contains",
        "fragment": "Your message, Hello from OpenHuman",
        "within": "Messages in chat with Alex Rivera"
    });
    assert_eq!(serde_json::to_value(&predicate).unwrap(), wire);
    assert_eq!(
        serde_json::from_value::<VisiblePredicate>(wire).unwrap(),
        predicate
    );
}
