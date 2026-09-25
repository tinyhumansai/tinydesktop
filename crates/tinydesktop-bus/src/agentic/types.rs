//! Wire types for native Jev-driven desktop control.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Jev-compatible decision service selected by the host.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JevProvider {
    /// `TypeSafe`'s first-party System One API.
    #[default]
    TypeSafe,
    /// `OpenRouter`'s Jev-compatible decisions API.
    OpenRouter,
    /// Tiny Humans' authenticated `OpenRouter` proxy.
    TinyHumansOpenRouter,
    /// OpenJEV's free public System One API (Jev-compatible).
    OpenJev,
}

/// Configures the Jev client retained by the loaded module.
///
/// This payload must be sent with `TinyBus` confidential delivery. Its custom
/// [`Debug`](std::fmt::Debug) implementation never prints the API key.
#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct JevConfig {
    api_key: String,
    /// Provider whose response contract should be validated.
    pub provider: JevProvider,
    /// Exact compatible endpoint, when the provider's conventional route is
    /// not desired.
    pub endpoint_url: Option<String>,
    /// Jev model or alias. Absent means `jev-latest`.
    pub model: Option<String>,
    /// Per-attempt HTTP timeout. Absent means the client default.
    pub timeout_ms: Option<u64>,
    /// Additional transient retries. Absent means the client default.
    pub max_retries: Option<u32>,
    /// Host product attribution for the `TinyHumans` proxy only.
    pub sdk_name: Option<String>,
}

impl JevConfig {
    /// Builds a configuration carrying `api_key` and provider defaults.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Self::default()
        }
    }

    /// Deliberately exposes the API key to the module constructing the client.
    #[must_use]
    pub fn api_key(&self) -> &str {
        &self.api_key
    }
}

impl std::fmt::Debug for JevConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("JevConfig")
            .field("api_key", &"[REDACTED]")
            .field("provider", &self.provider)
            .field("endpoint_url", &self.endpoint_url)
            .field("model", &self.model)
            .field("timeout_ms", &self.timeout_ms)
            .field("max_retries", &self.max_retries)
            .field("sdk_name", &self.sdk_name)
            .finish()
    }
}

/// Resolves one natural-language intent against the current application.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ResolveIntentRequest {
    /// Application whose current surface should be inspected.
    pub app: String,
    /// One action-oriented intent.
    pub intent: String,
    /// Caller-supplied text for a text-taking action.
    pub text: Option<String>,
    /// Optional container ref that narrows observation.
    pub root: Option<String>,
    /// Whether a safe resolved action should be executed.
    pub execute: bool,
    /// Whether ordinary field values may leave the machine for Jev.
    pub include_values: bool,
}

/// Runs a bounded observe-decide-act loop for one goal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RunGoalRequest {
    /// Application whose surfaces the loop controls.
    pub app: String,
    /// Visible end state the loop should reach.
    pub goal: String,
    /// Caller-supplied values consumed by text actions in order.
    pub text: Vec<String>,
    /// Optional container ref where observation starts.
    pub root: Option<String>,
    /// Exact window title to retain throughout the task.
    pub window: Option<String>,
    /// Exact window ID from `ListWindows`; binds every task observation.
    pub window_id: Option<String>,
    /// Allowed mutating operations. Empty keeps the legacy operation set.
    pub allowed_operations: Vec<JevOperation>,
    /// Exact accessible names or descriptions of permitted action targets.
    /// Empty keeps the legacy target set.
    pub allowed_targets: Vec<String>,
    /// Prepared text keyed by the accessible field name or description.
    pub text_slots: BTreeMap<String, String>,
    /// Accessibility-visible predicates that must all hold for verified completion.
    /// Empty preserves legacy Jev completion behavior.
    pub success: Vec<VisiblePredicate>,
    /// Whether ordinary field values may leave the machine for Jev.
    pub include_values: bool,
    /// Maximum executed actions, capped by the module at 40.
    pub max_steps: u32,
    /// Maximum Jev evaluations, capped by the module at 80.
    pub max_model_calls: u32,
    /// Whole-task wall-clock budget in milliseconds, capped at five minutes.
    pub max_elapsed_ms: u64,
    /// Whether consequential actions require a separate confirmation call.
    pub require_confirmations: bool,
    /// One-use handle from a previous confirmation stop. Other fields are ignored on continuation.
    pub continuation: Option<GoalContinuation>,
}

impl Default for RunGoalRequest {
    fn default() -> Self {
        Self {
            app: String::new(),
            goal: String::new(),
            text: Vec::new(),
            root: None,
            window: None,
            window_id: None,
            allowed_operations: Vec::new(),
            allowed_targets: Vec::new(),
            text_slots: BTreeMap::new(),
            success: Vec::new(),
            include_values: false,
            max_steps: 40,
            max_model_calls: 80,
            max_elapsed_ms: 120_000,
            require_confirmations: true,
            continuation: None,
        }
    }
}

/// A deterministic condition checked against a fresh accessibility snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VisiblePredicate {
    /// An element with this exact accessible name or description exists.
    NamePresent {
        /// Exact accessible name or description.
        name: String,
    },
    /// A descendant of an exactly named container has a name containing this text.
    NameContains {
        /// Required fragment of the descendant's accessible name.
        fragment: String,
        /// Exact accessible name of an ancestor container.
        within: String,
    },
    /// A named element holds this exact string value.
    ValueEquals {
        /// Exact accessible name or description.
        name: String,
        /// Expected complete string value.
        value: String,
    },
    /// A named element's string value contains this caller-supplied fragment.
    ValueContains {
        /// Exact accessible name or description.
        name: String,
        /// Expected string fragment.
        value: String,
    },
    /// A named element exposes this state token.
    StateContains {
        /// Exact accessible name or description.
        name: String,
        /// Expected accessibility state token.
        state: String,
    },
}

/// One predicate's compact, host-visible observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JevPredicateResult {
    /// Requested condition.
    pub predicate: VisiblePredicate,
    /// Whether the fresh observation satisfied it.
    pub matched: bool,
    /// Name of the observed element, when present.
    pub observed_name: Option<String>,
    /// Matched caller-supplied value or fragment; unrelated field content is omitted.
    pub observed_value: Option<String>,
    /// Observed state tokens only for a state predicate.
    pub observed_states: Vec<String>,
}

/// Last bounded accessibility evidence gathered by a goal run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JevObservation {
    /// Application reported by the snapshot.
    pub app: String,
    /// Window title reported by the snapshot.
    pub window: Option<String>,
    /// Surface type reported by the snapshot.
    pub surface: String,
    /// Independent predicate checks.
    pub predicates: Vec<JevPredicateResult>,
}

/// Host response to a pending consequential desktop action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoalContinuation {
    /// Opaque one-use handle returned by the module.
    pub id: String,
    /// Whether a person approved the exact pending operation and target.
    pub approve: bool,
}

/// A closed operation Jev may select.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum JevOperation {
    /// Activate one element.
    Click,
    /// Put caller-supplied text into one element.
    TypeText,
    /// Put a toggle into its checked state.
    Check,
    /// Put a toggle into its unchecked state.
    Uncheck,
    /// Expand a disclosure.
    Expand,
    /// Collapse a disclosure.
    Collapse,
    /// Scroll one container downward.
    Scroll,
    /// Inspect one truncated container.
    Drill,
    /// Return observation to the full surface.
    Widen,
    /// Wait for the application to settle.
    Wait,
    /// The visible goal is satisfied.
    Done,
    /// No offered operation can make progress.
    Blocked,
}

/// What the module decided about one proposed step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JevDecisionKind {
    /// The step may be executed.
    Act,
    /// The step is destructive and requires explicit confirmation.
    ConfirmationRequired,
    /// The evidence did not clear the execution threshold.
    Abstain,
    /// The selected operation needs caller-supplied text.
    NeedsText,
    /// The goal is visibly complete.
    Done,
    /// No offered operation can make progress.
    Blocked,
}

/// Element selected for an operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JevTarget {
    /// Snapshot-qualified element ref.
    pub ref_id: String,
    /// Accessibility role.
    pub role: String,
    /// Accessible name or description.
    pub name: Option<String>,
}

/// Result of resolving one intent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JevDecision {
    /// Policy outcome.
    pub decision: JevDecisionKind,
    /// Selected closed operation.
    pub operation: JevOperation,
    /// Selected element, if the operation needs one.
    pub target: Option<JevTarget>,
    /// Concentration reported for the selected target or terminal operation.
    pub confidence: f64,
    /// Probability that the step is hard to undo.
    pub destructive: f64,
    /// Human-readable, secret-free policy explanation.
    pub reason: String,
    /// Whether the safe step was executed.
    pub executed: bool,
}

/// One executed goal-loop turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JevTurn {
    /// One-based executed step number.
    pub step: u32,
    /// Operation that ran.
    pub operation: JevOperation,
    /// Target used by the operation.
    pub target: Option<JevTarget>,
    /// Target confidence.
    pub confidence: f64,
    /// Whether the desktop command succeeded.
    pub ok: bool,
    /// Whether the observed surface changed afterwards.
    pub changed: bool,
}

/// Why a goal loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JevStopReason {
    /// Jev reported visible completion.
    Done,
    /// No offered operation could advance the goal.
    Blocked,
    /// A destructive step requires the host's confirmation.
    ConfirmationRequired,
    /// The host declined a pending action.
    Cancelled,
    /// The approved target no longer matched the observed desktop.
    StaleTarget,
    /// Confidence was too low to act.
    LowConfidence,
    /// No caller-supplied value remained for a text action.
    NeedsText,
    /// The action budget was reached.
    ActionBudget,
    /// The model-call budget was reached.
    ModelBudget,
    /// Three consecutive turns changed nothing.
    Stalled,
    /// A desktop command failed or had uncertain delivery.
    ActionFailed,
    /// Jev ended before the visible conditions were satisfied.
    VerificationFailed,
    /// The wall-clock budget was exhausted.
    TimeBudget,
    /// The observed app/window or chosen action left the caller's scope.
    ScopeChanged,
    /// A mutation may have been delivered; it must not be replayed blindly.
    ActionUncertain,
}

/// Aggregate provider measurements for one result.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JevMetrics {
    /// Jev evaluations performed.
    pub calls: u32,
    /// HTTP attempts including retries.
    pub attempts: u32,
    /// Total provider latency in milliseconds.
    pub latency_ms: u64,
    /// Provider-reported input tokens.
    pub input_tokens: u64,
    /// Provider-reported output tokens.
    pub output_tokens: u64,
    /// Concrete model reported by the provider.
    pub model: Option<String>,
}

/// Result of a bounded goal loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JevRunResult {
    /// Structured stop reason.
    pub stop: JevStopReason,
    /// True only when every requested success predicate was observed.
    pub verified: bool,
    /// Last compact observation, if one was obtained.
    pub final_observation: Option<JevObservation>,
    /// Executed turns in order.
    pub turns: Vec<JevTurn>,
    /// Last decision when the loop stopped before executing it.
    pub pending: Option<JevDecision>,
    /// One-use handle to approve or decline `pending` through `RunGoal`.
    pub confirmation_id: Option<String>,
    /// Provider measurements.
    pub metrics: JevMetrics,
}

/// Non-secret summary of the retained Jev client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JevConfiguration {
    /// Configured provider.
    pub provider: JevProvider,
    /// Requested model or alias.
    pub model: String,
    /// Exact endpoint override, when set.
    pub endpoint_url: Option<String>,
}
