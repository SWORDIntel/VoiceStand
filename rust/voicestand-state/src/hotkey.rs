//! Linux-wide activation backed by X11/XWayland global hotkeys.

use anyhow::{anyhow, Context, Result};
use global_hotkey::{hotkey::HotKey, GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct HotkeyConfig {
    pub modifiers: Vec<String>,
    pub key: String,
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            modifiers: vec!["Ctrl".into(), "Alt".into()],
            key: "KeyV".into(),
        }
    }
}

impl HotkeyConfig {
    fn accelerator(&self) -> String {
        self.modifiers
            .iter()
            .chain(std::iter::once(&self.key))
            .cloned()
            .collect::<Vec<_>>()
            .join("+")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeyState {
    Pressed,
    Released,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HotkeyEvent {
    pub action: String,
    pub state: HotkeyState,
}

/// Owns registrations and bridges the synchronous desktop event source into Tokio.
pub struct HotkeyManager {
    registrations: HashMap<HotkeyConfig, String>,
    native: Option<GlobalHotKeyManager>,
    native_hotkeys: HashMap<u32, HotKey>,
    actions: HashMap<u32, String>,
    stop: Arc<AtomicBool>,
}

impl HotkeyManager {
    pub fn new() -> Self {
        Self {
            registrations: HashMap::new(),
            native: None,
            native_hotkeys: HashMap::new(),
            actions: HashMap::new(),
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    pub async fn initialize(&mut self) -> Result<()> {
        if std::env::var_os("DISPLAY").is_none() {
            return Err(anyhow!(
                "X11/XWayland activation unavailable: DISPLAY is not set"
            ));
        }
        self.native =
            Some(GlobalHotKeyManager::new().context("failed to connect to the X11 display")?);
        Ok(())
    }

    pub fn register_hotkey(&mut self, config: HotkeyConfig, action: String) -> Result<()> {
        let hotkey: HotKey = config
            .accelerator()
            .parse()
            .with_context(|| format!("invalid hotkey {}", config.accelerator()))?;
        let native = self
            .native
            .as_ref()
            .context("hotkey backend is not initialized")?;
        native
            .register(hotkey)
            .with_context(|| format!("binding conflict for {}", config.accelerator()))?;
        self.actions.insert(hotkey.id(), action.clone());
        self.native_hotkeys.insert(hotkey.id(), hotkey);
        self.registrations.insert(config, action);
        Ok(())
    }

    pub fn unregister_hotkey(&mut self, config: &HotkeyConfig) -> Result<()> {
        let hotkey: HotKey = config.accelerator().parse()?;
        if let Some(native) = &self.native {
            native.unregister(hotkey)?;
        }
        self.actions.remove(&hotkey.id());
        self.native_hotkeys.remove(&hotkey.id());
        self.registrations.remove(config);
        Ok(())
    }

    pub async fn start(&mut self) -> Result<mpsc::UnboundedReceiver<HotkeyEvent>> {
        if self.native.is_none() {
            return Err(anyhow!("hotkey backend is not initialized"));
        }
        let actions = self.actions.clone();
        let stop = Arc::clone(&self.stop);
        stop.store(false, Ordering::Release);
        let (tx, rx) = mpsc::unbounded_channel();
        std::thread::Builder::new()
            .name("voicestand-global-hotkey".into())
            .spawn(move || {
                let receiver = GlobalHotKeyEvent::receiver();
                while !stop.load(Ordering::Acquire) {
                    match receiver.recv_timeout(Duration::from_millis(100)) {
                        Ok(event) => {
                            if let Some(action) = actions.get(&event.id()) {
                                let state = match event.state() {
                                    HotKeyState::Pressed => HotkeyState::Pressed,
                                    HotKeyState::Released => HotkeyState::Released,
                                };
                                if tx
                                    .send(HotkeyEvent {
                                        action: action.clone(),
                                        state,
                                    })
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                    }
                }
            })
            .context("failed to start global hotkey event bridge")?;
        Ok(rx)
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        self.stop.store(true, Ordering::Release);
        if let Some(native) = &self.native {
            for hotkey in self.native_hotkeys.values() {
                native.unregister(*hotkey)?;
            }
        }
        self.registrations.clear();
        self.native_hotkeys.clear();
        self.actions.clear();
        self.native = None;
        Ok(())
    }
}

impl Default for HotkeyManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_binding_parses_for_native_backend() {
        let binding = HotkeyConfig::default();
        assert!(binding.accelerator().parse::<HotKey>().is_ok());
        assert_eq!(binding.accelerator(), "Ctrl+Alt+KeyV");
    }
    #[test]
    fn invalid_binding_is_rejected_before_registration() {
        let binding = HotkeyConfig {
            modifiers: vec!["Ctrl".into()],
            key: "DefinitelyNotAKey".into(),
        };
        assert!(binding.accelerator().parse::<HotKey>().is_err());
    }
}
