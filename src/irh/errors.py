class HarnessError(Exception):
    """A request or lifecycle failure safe to show to an operator."""


class ConflictError(HarnessError):
    """The requested transition conflicts with current state."""


class NotFoundError(HarnessError):
    """A requested object does not exist."""


class InterventionRequired(HarnessError):
    """External state requires the daemon to pause the project."""

    def __init__(self, source: str, detail: object):
        super().__init__(f"{source} requires operator intervention")
        self.source = source
        self.detail = detail
