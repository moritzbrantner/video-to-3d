//! Per-invocation cooperative cancellation.

/// Per-invocation cooperative cancellation. Sticky state prevents an interrupted
/// internal analysis from becoming accepted if a later callback returns false.
pub(crate) struct ReconstructionControl<'a> {
    poll: &'a mut dyn FnMut() -> bool,
    canceled: bool,
}
impl<'a> ReconstructionControl<'a> {
    pub(crate) fn new(poll: &'a mut dyn FnMut() -> bool) -> Self {
        Self {
            poll,
            canceled: false,
        }
    }
    pub(crate) fn is_canceled(&mut self) -> bool {
        self.canceled = self.canceled || (self.poll)();
        self.canceled
    }
    pub(crate) fn check(&mut self) -> Result<(), String> {
        if self.is_canceled() {
            Err("reconstruction canceled".into())
        } else {
            Ok(())
        }
    }
}
