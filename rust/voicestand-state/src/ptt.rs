// Push-to-Talk functionality
use anyhow::Result;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub enum PttEvent {
    Pressed,
    Released,
}

pub struct PttManager {
    event_sender: mpsc::UnboundedSender<PttEvent>,
}

impl PttManager {
    pub fn new() -> (Self, mpsc::UnboundedReceiver<PttEvent>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        (Self { event_sender: sender }, receiver)
    }

    pub fn handle_key_press(&self) -> Result<()> {
        self.event_sender.send(PttEvent::Pressed)?;
        Ok(())
    }

    pub fn handle_key_release(&self) -> Result<()> {
        self.event_sender.send(PttEvent::Released)?;
        Ok(())
    }
}