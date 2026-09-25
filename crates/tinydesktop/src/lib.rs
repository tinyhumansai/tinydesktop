//! Native desktop automation as an installable `TinyBus` module.
//!
//! tinydesktop wraps the [`agent-desktop`] engine — accessibility-tree
//! observation and interaction for macOS, Windows, and Linux — and serves it
//! over `TinyBus` as fifty-six typed members. A host loads the compiled
//! `cdylib`, and an agent behind that host gets structured access to any
//! running application: no screenshots to interpret, no pixel matching, no
//! browser.
//!
//! [`agent-desktop`]: https://github.com/lahfir/agent-desktop
//!
//! # Layout
//!
//! This is the implementation crate in a three-crate workspace:
//!
//! - [`tinydesktop_bus`] — the wire contract. Member names, request payloads,
//!   the response envelope, and the contract version, with no transport, no
//!   engine, and no behavior. A host that only makes calls depends on that
//!   crate alone.
//! - `tinydesktop` — this crate. The engine wrapper, the crate-wide error type,
//!   and the `TinyBus` adapter that serves them, built as both an `rlib` and
//!   the `cdylib` the loader consumes.
//!
//! Within this crate:
//!
//! - `src/error/` holds the crate-wide [`Error`] enum and the [`Result`] alias.
//! - `src/desktop/` holds [`Desktop`], with one method per member, split by
//!   family across sibling files.
//! - `src/tinybus_module/` adapts those methods to the bus and exports the
//!   module descriptor, embedded manifest, and initialization entrypoint.
//!
//! Every public item is re-exported from here — including all of
//! [`tinydesktop_bus`] — so downstream users have one predictable surface and
//! `tinydesktop::SnapshotRequest` is the *same type* as
//! `tinydesktop_bus::SnapshotRequest`, not a structural twin.
//!
//! # The model: observe, then act on what you observed
//!
//! A snapshot walks an application's accessibility tree and hands back a
//! compact description in which every element carries a *ref* — a qualified
//! handle like `@s8f3k2p9:e1`. Interaction members take those refs. They do not
//! take coordinates, and they do not take selectors evaluated fresh at click
//! time.
//!
//! That indirection is the whole design. A ref is bound to the snapshot it came
//! from, so acting on one either reaches the element that was described or
//! fails with `STALE_REF` and asks for a fresh snapshot. What it will not do is
//! click whatever has since moved into that position.
//!
//! ```no_run
//! use tinydesktop::{Desktop, FindRequest, RefRequest};
//!
//! let desktop = Desktop::new();
//!
//! let found = desktop.find(FindRequest {
//!     app: Some("Safari".to_owned()),
//!     role: Some("button".to_owned()),
//!     name: Some("Save".to_owned()),
//!     first: true,
//!     ..FindRequest::default()
//! });
//!
//! if found.ok {
//!     let reply = desktop.click(RefRequest::new("@s8f3k2p9:e1"));
//!     assert_eq!(reply.command, "click");
//! }
//! ```
//!
//! # Headless by default
//!
//! A ref action goes through the platform's accessibility API, not through
//! synthesized input, so it does not steal focus, move the cursor, or touch the
//! pasteboard as a side effect. A run can proceed while someone else is using
//! the machine. [`Desktop::with_headed`] relaxes that for the interactions that
//! genuinely need a real cursor, and the [`input`](tinydesktop_bus::input)
//! members bypass it entirely — both on purpose, and both the exception.
//!
//! # Errors are replies, not failures
//!
//! Every command method returns a [`DesktopResponse`] and never a `Result`. A
//! stale ref, a missing permission, an ambiguous application name — these carry
//! codes, suggestions, and recovery hints a caller can branch on, and flattening
//! them into an error string would throw that away. [`Error`] is reserved for
//! the module failing to start a command at all. See
//! [`tinydesktop_bus::envelope`] for the full reasoning.
//!
//! # Platform support
//!
//! macOS and Windows have full accessibility backends. Linux builds, loads, and
//! answers, but implements no surfaces yet: every observation there fails with
//! `PLATFORM_NOT_SUPPORTED` and lists the surfaces it does support, which is
//! none. That is inherited from the vendored engine and will follow it.

mod agentic;
mod desktop;
mod error;
mod tinybus_module;

/// Constructs this module for registration with an in-process TinyBus host.
#[cfg(feature = "static-link")]
pub use tinybus_module::linked_module;

pub use desktop::Desktop;
pub use error::{Error, Result};

// The wire contract, re-exported by module rather than by item so every path
// through this crate resolves to the same definitions the contract crate
// publishes. A host may depend on `tinydesktop-bus` directly and get exactly
// these types; nothing here redefines them.
pub use tinydesktop_bus;
pub use tinydesktop_bus::CloseAppRequest;
pub use tinydesktop_bus::{
    CONTRACT_VERSION, ClipboardFormat, ClipboardGetRequest, ClipboardSetRequest, Delivery,
    DeliveryDisposition, DesktopError, DesktopResponse, Direction, DismissAllNotificationsRequest,
    DismissNotificationRequest, DragEndpoint, DragRequest, ENVELOPE_VERSION, ElementProperty,
    ElementStateProperty, FindRequest, FocusWindowRequest, GetRequest, HoldKeyRequest,
    HoldMouseRequest, HoverRequest, INTERFACE, IsRequest, JevConfig, JevConfiguration, JevDecision,
    JevDecisionKind, JevMetrics, JevObservation, JevOperation, JevPredicateResult, JevProvider,
    JevRunResult, JevStopReason, JevTarget, JevTurn, LaunchRequest, ListAppsRequest,
    ListNotificationsRequest, ListSurfacesRequest, ListWindowsRequest, METHODS, Modifier,
    MouseButton, MouseClickRequest, MouseMoveRequest, MouseWheelRequest, MoveWindowRequest,
    NotificationActionRequest, OBJECT_PATH, PermissionsRequest, PressRequest, RecoveryHint,
    RefRequest, ResizeWindowRequest, ResolveIntentRequest, RetryDisposition, RunGoalRequest,
    ScreenshotRequest, ScrollRequest, SelectRequest, SetValueRequest, SnapshotRequest,
    StatePredicate, Surface, TypeRequest, VisiblePredicate, WaitRequest, WindowRequest,
    is_compatible, names, version,
};
