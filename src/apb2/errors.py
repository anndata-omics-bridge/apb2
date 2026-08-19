"""apb2's error vocabulary, shared by every layer.

Top level because both sides raise from it: the rule side rejects evidence that excludes a
level, the strategy side rejects a file that cannot satisfy one. A schema package must not
import ``parse_quant``, so an error class living there would force the rule side to hand
back a reason string for someone else to raise — which puts the message far from the facts.
"""

from __future__ import annotations


class RuleNotApplicable(ValueError):
    """This rule does not apply to what the caller has — try another level.

    The skip contract: ``configure_parse.make_parse_strategies`` catches this to move to
    the next quantification level, so anything meaning "not this level" must be this class
    or a subclass, and anything meaning "the caller is wrong" must not be.
    """


class IncompatibleSourceError(RuleNotApplicable):
    """The bound source cannot satisfy the rule's declared contract.

    A subclass, because a file that lacks the rule's columns is one way for the rule not to
    apply — and callers that want to know it was the *file* can still catch this alone.
    """


class RuleUnavailableError(ValueError):
    """No packaged rules.json satisfies the supplied evidence."""


class AmbiguousRuleError(ValueError):
    """Several packaged rules.json satisfy evidence that must identify exactly one."""


class NoCompatibleLevelError(ValueError):
    """None of the supplied rules can be constructed from the bound source."""


class AmbiguousDialectError(ValueError):
    """Several candidate dialects satisfy the rule; bind an explicit ``DelimitedFile``."""
