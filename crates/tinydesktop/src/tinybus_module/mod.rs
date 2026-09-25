//! `TinyBus` module entrypoint and bus-facing interface.
//!
//! This adapter keeps [`crate::Desktop`] independent from `TinyBus` while
//! exposing it as an installable, dynamically loaded integration. The names and
//! payload types it serves come from [`tinydesktop_bus`], so a host spells them
//! from the contract crate instead of repeating string literals.
//!
//! # Why every member hands its work to a blocking pool
//!
//! Accessibility APIs are synchronous, and some of these calls are slow on
//! purpose: a snapshot of a dense application walks thousands of elements, and
//! a wait blocks until something happens or thirty seconds pass. Running one on
//! the connection's dispatch task would stall every other caller behind it for
//! that whole time.
//!
//! So each member is a thin `async fn` that clones the configuration and hands
//! the real work to [`tokio::task::spawn_blocking`]. That is affordable because
//! [`crate::Desktop`] is four small fields and builds its platform adapter per
//! call — nothing platform-specific has to cross a thread boundary or survive
//! an `await`.

mod dispatch;

use serde_json::Value;
use tinybus::{Connection, Result as TinyBusResult};
use tinydesktop_bus::names;

pub(crate) use dispatch::DesktopService;

/// Serves the desktop interface and claims its well-known name.
///
/// `config` is whatever the host recorded for this module, already parsed. An
/// unreadable one fails the load rather than falling back to defaults: a module
/// silently ignoring the session it was told to join would allocate refs
/// nothing else can spend.
async fn setup(connection: Connection, config: Value) -> TinyBusResult<()> {
    let service = DesktopService::from_config(&config)
        .map_err(|error| tinybus::Error::failed(error.to_string()))?;

    connection
        .serve_at(names::OBJECT_PATH.try_into()?, service)
        .await?;
    connection.request_name(names::INTERFACE).await?;
    Ok(())
}

tinybus_module::module_export_optional_static! {
    setup = setup,
    config = serde_json::Value,
    // Two: one to run a blocking command on, and one to keep answering on
    // while it runs. A single thread would serialize the very calls
    // `spawn_blocking` exists to keep apart.
    worker_threads = 2,
    provides = ["ai.tinyhumans.tinydesktop.Desktop"],
    methods = [
        "ResolveIntent", "RunGoal",
        "Snapshot", "Find", "Get", "Is", "Screenshot",
        "Click", "DoubleClick", "TripleClick", "RightClick", "Type", "SetValue", "Clear",
        "Focus", "Select", "Toggle", "Check", "Uncheck", "Expand", "Collapse", "Scroll",
        "ScrollTo",
        "Press", "KeyDown", "KeyUp", "Hover", "Drag", "MouseMove", "MouseClick", "MouseDown",
        "MouseUp", "MouseWheel",
        "Launch", "CloseApp", "ListApps", "ListWindows", "ListDisplays", "ListSurfaces",
        "FocusWindow", "ResizeWindow", "MoveWindow", "Minimize", "Maximize", "Restore",
        "ClipboardGet", "ClipboardSet", "ClipboardClear",
        "ListNotifications", "NotificationAction", "DismissNotification",
        "DismissAllNotifications",
        "Wait",
        "Version", "Status", "Permissions",
    ],
    signals = [],
    requires = [],
    optional = [],
    lazy = false,
}

#[cfg(test)]
mod test;
