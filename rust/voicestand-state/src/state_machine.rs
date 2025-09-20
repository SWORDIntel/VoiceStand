// State machine for VoiceStand coordination
use anyhow::Result;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
pub enum VoiceStandState {
    Idle,
    Listening,
    Processing,
    Speaking,
    Error,
}

#[derive(Debug, Clone)]
pub enum StateTransition {
    StartListening,
    StartProcessing,
    StartSpeaking,
    ReturnToIdle,
    ErrorOccurred,
}

pub struct StateMachine {
    current_state: VoiceStandState,
    state_entered: Instant,
    transition_count: usize,
}

impl StateMachine {
    pub fn new() -> Self {
        Self {
            current_state: VoiceStandState::Idle,
            state_entered: Instant::now(),
            transition_count: 0,
        }
    }

    pub fn current_state(&self) -> &VoiceStandState {
        &self.current_state
    }

    pub async fn initialize(&mut self) -> Result<()> {
        self.current_state = VoiceStandState::Idle;
        self.state_entered = Instant::now();
        self.transition_count = 0;
        Ok(())
    }

    pub fn transition(&mut self, transition: StateTransition) -> Result<()> {
        let new_state = match (&self.current_state, transition) {
            (VoiceStandState::Idle, StateTransition::StartListening) => VoiceStandState::Listening,
            (VoiceStandState::Listening, StateTransition::StartProcessing) => VoiceStandState::Processing,
            (VoiceStandState::Processing, StateTransition::StartSpeaking) => VoiceStandState::Speaking,
            (_, StateTransition::ReturnToIdle) => VoiceStandState::Idle,
            (_, StateTransition::ErrorOccurred) => VoiceStandState::Error,
            _ => return Ok(()), // Invalid transition, ignore
        };

        self.current_state = new_state;
        self.state_entered = Instant::now();
        self.transition_count += 1;
        Ok(())
    }
}