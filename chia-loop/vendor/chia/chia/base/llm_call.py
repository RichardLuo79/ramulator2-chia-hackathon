
import warnings
from typing import List, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from chia.base.tools.ChiaTool import ChiaTool


# Sentinel for "argument not provided". Lets LLMCallBase tell an explicit value
# (which warrants a warning on a backend that ignores it) from the unset default.
UNSET = object()


@dataclass
class QueryResult:
    """
    Structured result from prompting an LLM or agent.
    
    :param result: The final response from the LLM or agent
    :param returncode: The returncode from running the prompt
    :param stderr: The stderr output from running the prompt (clis only)
    :param stream_result: The full transcript of all turns of the LLM or agent
    :param success: Whether the prompt completed successfully
    :type result: str
    :type returncode: int
    :type stderr: str
    :type stream_result: str
    :type success: bool
    """

    result: str
    returncode: int
    stderr: str
    stream_result: str
    success: bool = False
    # Unabridged UTF-8 CLI output is separate from the human-readable preview.
    # Keyword-only fields preserve positional construction by existing callers.
    raw_stdout: str = field(default="", kw_only=True)
    error: Optional[BaseException] = field(default=None, kw_only=True)


def single_attempt(run, restore, capture, result_type, classify=None) -> QueryResult:
    """Run once and return failure evidence with the newest native continuation.

    A workflow with its own durable attempt ledger must not invoke an invisible
    retry loop. Adapters still own native parsing/capture/restore and error
    types. In particular, capture happens before error classification, including
    timeout paths. Failure to restore/capture is an infrastructure exception,
    not permission to repeat a possibly charged request.
    """
    restore()
    try:
        result = run()
    except BaseException as exc:
        # subprocess.TimeoutExpired can carry acknowledged native output.
        def text(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")

        result = result_type(
            result="", returncode=-1, stderr=text(getattr(exc, "stderr", None)) or str(exc),
            stream_result="", raw_stdout=text(getattr(exc, "stdout", None)), error=exc,
        )
        if not isinstance(exc, Exception):
            capture(result)
            raise
    capture(result)
    if result.error is None and classify is not None:
        try:
            classify(result)
        except Exception as exc:
            result.error = exc
    result.success = result.error is None and result.returncode == 0
    return result

class LLMCallBase(ABC):
    """
    Polymorphic base container for generic LLM and agent
    configuration traits and behavior. Easy to switch between
    different backing providers, servers, and CLIs
    """

    # Capability flags — subclasses that honor these permission controls
    # override them to True. When False (the default), passing the corresponding
    # argument emits a warning that it will be ignored (see __init__).
    supports_dangerously_skip_permissions: bool = False
    supports_config: bool = False

    def __init__(
        self,
        system_message: str,
        dangerously_skip_permissions=UNSET,
        config=UNSET,
    ):
        self.system_message = system_message
        cls = type(self).__name__
        if (dangerously_skip_permissions is not UNSET
                and not self.supports_dangerously_skip_permissions):
            warnings.warn(
                f"{cls} does not support 'dangerously_skip_permissions'; the "
                f"argument is ignored (this backend has no permission gate).",
                stacklevel=2,
            )
        if config is not UNSET and not self.supports_config:
            warnings.warn(
                f"{cls} does not support a 'config' block; the "
                f"argument is ignored.",
                stacklevel=2,
            )
        # Maps ONLY to the backend's "dangerously skip permissions" CLI flag
        # (claude/opencode/antigravity --dangerously-skip-permissions, codex
        # --dangerously-bypass-approvals-and-sandbox, copilot --allow-all).
        # Honored only where supports_dangerously_skip_permissions is True.
        self.dangerously_skip_permissions = (
            True if dangerously_skip_permissions is UNSET else dangerously_skip_permissions
        )
        # The backend's config block (e.g. opencode's `permission`
        # object). ``None`` means "allow all". Honored only where
        # supports_config is True.
        self.config = None if config is UNSET else config

    @abstractmethod
    def prompt(self, user_message: str, tools: Optional[List[ChiaTool]] = []) -> QueryResult:
        """
        Send a prompt to this LLM

        :param user_message: Message used to prompt the LLM
        :param tools: Tools available to the LLM during the call
        :type user_message: str
        :type tools: Optional[List[ChiaTool]]
        """
        pass
