"""Every sentence the landing page prints.

The page is read by people who do not work on simulation, so its wording is the part of it
most likely to be rewritten. Holding all of it here means the writing can be edited and
reviewed as writing, without reading a builder, and it means no builder can quietly invent
a claim: a builder may only place a string from this module beside a number from the run
records.

Nothing here is derived and nothing here is a number that a record could supply. Where a
sentence needs a figure, it carries a named field that the builder fills from the model.

Two things in this module are markup rather than prose. The schematic is fixed artwork
that illustrates the premise rather than reporting a measurement, and it belongs beside
the sentences it illustrates. The inline emphasis in a paragraph is written as `**bold**`
and `` `code` ``, which the builder converts after escaping, so that curated text can be
emphasised without any string reaching the page unescaped.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ANSWER",
    "ASIDE",
    "COST",
    "DIAGNOSIS",
    "DIAGNOSIS_CLOSING",
    "DIAGNOSIS_TABLE",
    "DRIFT",
    "EYEBROW",
    "FINDINGS",
    "FINDING_LIST",
    "FOOTER",
    "HARNESS_ROLE",
    "HEADLINES",
    "NAV_LINKS",
    "PREMISE",
    "RUNS",
    "SCHEMATIC",
    "TITLE",
    "TRUST",
    "UNKNOWN_RANK",
    "WORDMARK",
    "Fault",
    "Figure",
    "Finding",
    "Headline",
    "Placeholder",
    "Section",
    "Table",
    "fault_note",
    "model_label",
    "system_label",
]


@dataclass(frozen=True, slots=True)
class Section:
    """The prose at the top of one section.

    Attributes:
        anchor: Fragment identifier, which the navigation links to.
        kicker: The line above the heading.
        heading: The heading.
        paragraphs: The introduction, one string per paragraph.
    """

    anchor: str
    kicker: str
    heading: str
    paragraphs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Figure:
    """The words around one figure.

    Attributes:
        title: What the figure is called.
        subtitle: How to read its axes.
        alt: What the figure shows, for a reader who cannot see it.
        caption: What to conclude from it.
    """

    title: str
    subtitle: str
    alt: str
    caption: str


@dataclass(frozen=True, slots=True)
class Placeholder:
    """What stands where a later phase draws something.

    Attributes:
        anchor: Identifier the builder marks the gap with, so a later phase can find it.
        text: What will go there.
    """

    anchor: str
    text: str


@dataclass(frozen=True, slots=True)
class Headline:
    """The words beside one hero figure.

    Attributes:
        lead: What the number is.
        note: Why it matters. Carries the named fields the builder fills.
    """

    lead: str
    note: str


@dataclass(frozen=True, slots=True)
class Finding:
    """One negative result.

    Attributes:
        what: The result in a phrase.
        why: The result in a paragraph.
    """

    what: str
    why: str


@dataclass(frozen=True, slots=True)
class Fault:
    """What one injected fault was, in the reader's terms.

    Attributes:
        name: What the fault is called, as a sentence rather than as an identifier.
        plain: What was broken, without the vocabulary of the codebase.
        hidden: Whether the fault leaves no trace in the configuration, so that the only
            way to find it is to read the numbers.
    """

    name: str
    plain: str
    hidden: bool


@dataclass(frozen=True, slots=True)
class Table:
    """The words around one table.

    Attributes:
        caption: What the table shows.
        headers: Column headings, in order.
    """

    caption: str
    headers: tuple[str, ...]


WORDMARK = "nnphysics"

TITLE = "Can a neural network replace a physics simulator?"

EYEBROW = "Neural surrogates and an evaluation harness"

ANSWER = (
    "On two systems, gravitational N-body and 2D fluid flow, the answer measured here is "
    "**not yet**. The interesting part is not that the networks failed. It is that the "
    "harness says precisely where, when and by how much, and it was built to catch itself "
    "being wrong."
)

NAV_LINKS = (
    ("drift", "The drift"),
    ("trust", "Trust"),
    ("cost", "Cost"),
    ("diagnosis", "Diagnosis"),
    ("runs", "Runs"),
)
"""Where the navigation points. The diagnosis link is dropped when a page is built without
the fault scores, because a link to a section that was not rendered is a broken link."""

HEADLINES = (
    Headline(
        lead="Steps the best surrogate stays usable",
        note="After that the error is larger than the signal it is predicting.",
    ),
    Headline(
        lead="Rollouts that survive an unseen setup",
        note="The {predictor} was shown a configuration no training trajectory came from.",
    ),
    Headline(
        lead="Than the simulator it was meant to replace",
        note="Compared at equal accuracy, which is the only comparison that means anything.",
    ),
)
"""In the order the hero prints them: how far the best surrogate got, how it did on a setup
it never trained on, and what it cost."""

ASIDE = (
    "The project also ships an agent that diagnoses failed training runs. Given {faults} "
    "deliberately sabotaged runs it named the true cause first in {agent} of them, against "
    "{baseline} for a conventional baseline."
)

ASIDE_LINK = "See the scoring"

PREMISE = Section(
    anchor="why",
    kicker="The idea",
    heading="Fast, or right. The question is how long you get both.",
    paragraphs=(
        "A physics simulator takes the current state of the world and steps it forward, "
        "over and over, following rules we know are correct. It is trustworthy and it is "
        "slow.",
        "A neural network can learn to take the same step in a fraction of the time. But "
        "it is an approximation, and each step is fed its own previous answer. Small "
        "errors compound. The prediction tracks reality, then drifts, then stops meaning "
        "anything.",
        '**So the useful question is never "is it accurate".** It is: how far can you run '
        "before it is not, does it warn you, and is it actually cheaper than running the "
        "simulator. This project measures all three.",
    ),
)

SCHEMATIC = Figure(
    title="How a surrogate fails",
    subtitle="Prediction error against simulated time, in schematic form.",
    alt=(
        "Schematic showing surrogate error growing away from the simulator's flat line, "
        "crossing a usable threshold early in the rollout."
    ),
    caption=(
        "**The danger is the right hand side.** A surrogate does not stop or throw an "
        "error when it stops being right. It keeps returning confident, plausible, wrong "
        "states. Every metric in this project exists to find the crossing point."
    ),
)

SCHEMATIC_SVG = """\
<svg viewBox="0 0 720 260" role="img" aria-label="{alt}">
  <defs>
    <linearGradient id="trustfade" x1="0" x2="1">
      <stop offset="0%" stop-color="var(--fluid)" stop-opacity="0.16"/>
      <stop offset="100%" stop-color="var(--fluid)" stop-opacity="0.02"/>
    </linearGradient>
  </defs>
  <rect x="60" y="30" width="105" height="170" fill="url(#trustfade)"/>
  <line x1="60" y1="200" x2="690" y2="200" stroke="var(--axis)" stroke-width="1"/>
  <line x1="60" y1="30" x2="60" y2="200" stroke="var(--axis)" stroke-width="1"/>
  <line x1="60" y1="150" x2="690" y2="150" stroke="var(--fail)" stroke-width="1.5" \
stroke-dasharray="5 4" opacity="0.75"/>
  <text x="686" y="142" text-anchor="end" font-size="12" fill="var(--fail)">\
error as large as the signal</text>
  <path d="M60 199 H690" stroke="var(--muted)" stroke-width="2" fill="none"/>
  <text x="200" y="192" font-size="12" fill="var(--muted)">\
simulator, correct by construction</text>
  <path d="M60 199 C 130 196, 168 172, 200 150 S 300 74, 420 48 S 600 38, 690 36" \
stroke="var(--fluid)" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <text x="300" y="105" font-size="12" fill="var(--fluid)" font-weight="600">surrogate</text>
  <circle cx="200" cy="150" r="5" fill="var(--surface)" stroke="var(--fail)" stroke-width="2.5"/>
  <text x="112" y="222" text-anchor="middle" font-size="12" fill="var(--ink-2)">usable</text>
  <text x="430" y="222" text-anchor="middle" font-size="12" fill="var(--muted)">\
not usable, but still producing numbers</text>
  <text x="375" y="248" text-anchor="middle" font-size="12" fill="var(--muted)">\
simulated time</text>
</svg>
"""
"""Fixed artwork, not a chart. It carries no measurement, so nothing in it is derived and
nothing in it may be read as a result. Every colour is a theme token, so it follows the
reader's theme like the rest of the page."""

DRIFT = Section(
    anchor="drift",
    kicker="The evidence",
    heading="Watch it drift",
    paragraphs=(
        "Both rows below start from the same state. The top row is the real simulation. "
        "The bottom row is the neural network's prediction, fed only its own previous "
        "output. Time runs left to right.",
    ),
)

DRIFT_PLACEHOLDER = Placeholder(
    anchor="drift-viewer",
    text="The drift viewer goes here.",
)

TRUST = Section(
    anchor="trust",
    kicker="How far you get",
    heading="Usable for a fraction of the rollout",
    paragraphs=(
        "A rollout is counted as usable while the prediction error stays below a tenth of "
        "the size of the state itself. Past that the mistake is comparable to the thing "
        "being predicted. Each bar is the length of that usable stretch, drawn against the "
        "full rollout the model was asked for.",
    ),
)

TRUST_PLACEHOLDER = Placeholder(
    anchor="usable-steps-chart",
    text="The chart of usable steps goes here.",
)

COST = Section(
    anchor="cost",
    kicker="The comparison that matters",
    heading="Is it even faster?",
    paragraphs=(
        "The point of a surrogate is speed, so it has to be timed against a simulator "
        "turned down to the same accuracy, not against the simulator at its most careful "
        "setting. The simulator has a quality dial. Turning it down is free and it is the "
        "real competition.",
    ),
)

COST_PLACEHOLDER = Placeholder(
    anchor="cost-chart",
    text="The chart of accuracy against time per step goes here.",
)

DIAGNOSIS = Section(
    anchor="diagnosis",
    kicker="The other half of the project",
    heading="An agent that works out why a training run went wrong",
    paragraphs=(
        "Reading two run records and working out what changed is slow, specialised work. "
        "`nnp diagnose` hands both records to a language model and asks for the cause. It "
        "changes nothing and runs nothing: it reads and explains.",
        "An agent that writes a fluent explanation of a run nobody broke cannot be told "
        "apart from one that writes a fluent explanation of anything. So the test is a "
        "fault injection. **{faults} known faults, written down in advance**, each "
        "introduced separately into a copy of the same healthy run. Some of them show up "
        "in a configuration file and could be spotted by reading two lines. The rest leave "
        "no trace anywhere except in the numbers.",
    ),
)

DIAGNOSIS_PLACEHOLDER = Placeholder(
    anchor="diagnosis-chart",
    text="The chart comparing the agent with the baseline goes here.",
)

DIAGNOSIS_TABLE = Table(
    caption="Every fault, and where the agent ranked the true cause.",
    headers=("What was broken", "In plain terms", "Findable in the config?", "Agent's rank"),
)

UNKNOWN_RANK = "not named"
"""What the table prints where the diagnoser never named the true cause at all."""

DIAGNOSIS_CLOSING = (
    "**The row it got wrong is the most interesting one.** On the lost optimiser state it "
    "ranked the true cause third, arguing that the two loss curves start at the same "
    "value, finish within half a per cent of each other and show no discontinuity where "
    "the resume happened. That reasoning is correct. On a model this small the fault "
    "barely does anything. A test suite where every answer is findable measures the suite, "
    "not the diagnoser.",
    "One caveat stated plainly: no API credential was available, so each context was "
    "answered by a separate model instance and scored by the same code rather than by the "
    "shipped command. The ranking is honest. The cost per diagnosis was never measured.",
)

FINDINGS = Section(
    anchor="findings",
    kicker="Negative results, kept",
    heading="What did not work",
    paragraphs=(
        "This section is worth more than the successes, so it is not compressed. Every "
        "claim here comes from a stored run record, and two of them correct numbers that "
        "an earlier version of this project reported wrongly.",
    ),
)

FINDING_LIST = (
    # docs/results.md, "What did not work": the neural operator lost to its own control.
    Finding(
        what="The fancy architecture lost to its own control",
        why=(
            "A neural operator was expected to beat a plain convolutional network on the "
            "fluid. At a parameter count matched to eight parts in ten thousand and "
            "identical training settings, it lost on every accuracy measure. Its single "
            "win, behaving consistently when the grid is refined, is real and large, and "
            "is the only thing the spectral structure bought."
        ),
    ),
    # docs/results.md, "What did not work": neither surrogate generalises to its held out
    # regime.
    Finding(
        what="Nothing generalises to an unseen setup",
        why=(
            "The N-body network diverges on all four unseen rollouts, between steps 153 "
            "and 159. The fluid operator diverges on all four by step 28. Only the "
            "convolutional network survives, and even it is beaten there by doing nothing "
            "at all."
        ),
    ),
    # docs/results.md, "What did not work": the ensemble did not rescue a model that had
    # failed.
    Finding(
        what="An ensemble did not rescue a failed model",
        why=(
            "Averaging four networks that have all left the physical manifold gives a path "
            "that is not on it either. It diverged a handful of steps later, which is not "
            "a rescue."
        ),
    ),
    # docs/results.md, "What did not work": calibration can be bought with vagueness.
    Finding(
        what="Confidence can be bought with vagueness, and was",
        why=(
            "The fluid ensemble's error bars cover the truth 96 per cent of the time, "
            "while being 159 times the size of the state being predicted. That is a model "
            "saying the answer might be anywhere. Every coverage figure in this project is "
            "printed beside its sharpness for exactly that reason."
        ),
    ),
    # docs/results.md, "What did not work": the uncertainty warns in time on one system and
    # not the other.
    Finding(
        what="Uncertainty warned too late on one of two systems",
        why=(
            "The fluid ensemble's spread crosses the trust threshold before its error "
            "does, which is a usable warning. The N-body ensemble's crosses after. A "
            "surrogate that notices it has gone wrong only once it has gone wrong cannot "
            "decide when to fall back."
        ),
    ),
    # docs/results.md, "What did not work": two numbers in the previous README were wrong.
    Finding(
        what="Two previously published numbers were wrong",
        why=(
            "An earlier README reported 3 of 4 unseen rollouts completing where the record "
            "shows 0 of 4, and reported the ensemble diverging sooner where it diverged "
            "later. Both are corrected, and the records are why they could be."
        ),
    ),
)

RUNS = Section(
    anchor="runs",
    kicker="The evidence, unabridged",
    heading="Every run behind these numbers",
    paragraphs=(
        "Each run identifier is a hash of the resolved configuration and the seed, so the "
        "same configuration reproduces the same identifier. Each record pins the commit, "
        "the dataset, the library versions and the machine.",
    ),
)

USABLE_CELL = "usable for"
SPEED_CELL = "speed"
VERDICT_CELL = "verdict"
NO_MODEL = "no model"
"""What the register prints where a run trained nothing to be usable for anything."""

HARNESS_ROLE = "harness check"
"""What a run that trained no model is called in the register."""

FOOTER = (
    "Reports are generated, not written. Rebuild everything linked here with "
    "`uv run nnp report render --run <run-id>`. Every metric in the harness has a sentinel "
    "test: a deliberately broken predictor it must catch. A metric that cannot fail is not "
    "a metric.",
    "Measured on one machine, eight CPU cores, no GPU. Absolute timings do not transfer. "
    "Full limitations in `docs/results.md`.",
)

THEME_BUTTON = ("dark", "light")
"""What the theme button says in the light theme and in the dark one. It names the theme it
switches to, not the one in use."""

_SYSTEMS = {"nbody": "N-body", "fluid": "Fluid"}

_MODELS = {
    "graph": "graph network",
    "operator": "neural operator",
    "convolution": "convolutional network",
    "mlp": "perceptron",
    "ensemble": "ensemble",
    "persistence": "repeat last state",
    "linear_extrapolation": "straight line",
    "noise": "random noise",
    "reference": "the simulator",
}

_FAULTS = {
    "wrong_normalisation": Fault(
        name="Wrong normalisation",
        plain="Training statistics that do not describe the data",
        hidden=True,
    ),
    "broken_symmetry": Fault(
        name="Broken symmetry",
        plain="A physical symmetry forced back on after every step",
        hidden=True,
    ),
    "no_optimiser_state": Fault(
        name="Lost optimiser state",
        plain="Resumed from a checkpoint with the optimiser's memory discarded",
        hidden=True,
    ),
    "high_learning_rate": Fault(
        name="Learning rate too high",
        plain="Two hundred times the configured rate",
        hidden=False,
    ),
    "wrong_regime": Fault(
        name="Wrong regime",
        plain="Training and held out setups swapped",
        hidden=False,
    ),
    "no_curriculum": Fault(
        name="No curriculum",
        plain="The multi step training schedule cut to a single step",
        hidden=False,
    ),
    "unstable_integrator": Fault(
        name="Unstable integrator",
        plain="The simulator run too coarsely to be stable",
        hidden=False,
    ),
}

HIDDEN_LABEL = "hidden"
VISIBLE_LABEL = "visible"


def system_label(system: str) -> str:
    """Name a system as the reader knows it.

    Args:
        system: The system as the record names it.

    Returns:
        The reader's name for it, or the record's name where there is no curated one. A
        system added later shows up under its own name rather than breaking the page.
    """
    return _SYSTEMS.get(system, system)


def model_label(model: str) -> str:
    """Name a model or a predictor as the reader knows it.

    Args:
        model: The model or predictor as the record names it.

    Returns:
        The reader's name for it, or the record's name where there is no curated one.
    """
    return _MODELS.get(model, model)


def fault_note(fault: str, true_cause: str) -> Fault:
    """Describe one injected fault in the reader's terms.

    Args:
        fault: The fault as the scores file names it.
        true_cause: What the scores file says was actually broken, used where the fault
            has no curated description.

    Returns:
        The description. A fault added to the suite is named and described by its own
        identifiers rather than left blank, and is reported as findable, which is the
        weaker claim.
    """
    return _FAULTS.get(
        fault,
        Fault(name=_sentence(fault), plain=_sentence(true_cause), hidden=False),
    )


def _sentence(identifier: str) -> str:
    """A recorded identifier as a sentence, so `wrong_regime` reads as `Wrong regime`."""
    spaced = identifier.replace("_", " ")
    return spaced[:1].upper() + spaced[1:]
