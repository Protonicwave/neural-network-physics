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
    "COST_ASIDE",
    "COST_CHART",
    "DIAGNOSIS",
    "DIAGNOSIS_CHART",
    "DIAGNOSIS_CLOSING",
    "DIAGNOSIS_TABLE",
    "DRIFT",
    "DRIFT_ALT",
    "DRIFT_CAPTION",
    "DRIFT_CONTROLS",
    "DRIFT_NOTES",
    "DRIFT_SPLITS",
    "EYEBROW",
    "FINDINGS",
    "FINDING_LIST",
    "FOOTER",
    "HARNESS_ROLE",
    "HEADLINES",
    "NAV_LABEL",
    "NAV_LINKS",
    "PREMISE",
    "RUNS",
    "SCHEMATIC",
    "SKIP_LINK",
    "THEME_LABEL",
    "TITLE",
    "TRUST",
    "TRUST_CHART",
    "UNKNOWN_RANK",
    "WORDMARK",
    "Fault",
    "Figure",
    "Finding",
    "Headline",
    "Section",
    "Table",
    "drift_looking",
    "drift_meaning",
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

SKIP_LINK = "Skip to content"

NAV_LABEL = "Sections of this page"
"""What the navigation is called for a reader who reaches it by structure rather than by
seeing it."""

TIMES = "\u00d7"
"""The multiplication sign, written once and as an escape, because the character itself is
easily confused with the letter x in source. A ratio against the solver is stated with it
rather than with the letter, which reads as a variable beside a number."""

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
        "A neural network can learn to take the same step in a fraction of the time. A "
        "network standing in for a simulator this way is called a surrogate. It is an "
        "approximation, and each step is fed its own previous answer, so small errors "
        "compound. The prediction tracks reality, then drifts, then stops meaning "
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
        "Every figure below starts from a state the simulator and the network agree on, "
        "and runs forward from it. Time runs left to right. The simulator is the truth. "
        "The network is fed only its own previous output, so its mistakes compound. Where "
        "a note says a prediction left the physical range, it means the numbers grew past "
        "anything the physics can produce and the run was stopped.",
    ),
)

DRIFT_CONTROLS = ("System", "Predictor", "Setup")
"""What the three groups of buttons are called, in the order the viewer shows them."""

DRIFT_SPLITS = (("test", "Trained on"), ("held_out", "Never seen"))
"""The two setups, and what a reader is told to call them. The record's own split names
mean nothing to somebody who has not read the harness."""

DRIFT_NOTES = ("What you are looking at", "What it means")
"""The headings of the two notes under the image."""

DRIFT_ALT = "The {predictor} prediction against the true {system} simulation, {setup} setup."

DRIFT_CAPTION = (
    '**Try "Never seen".** That setup is a configuration held back from training '
    "entirely: a shear layer for the fluid, a pair of tight binaries for N-body. It is "
    "the honest test, because in use a surrogate meets conditions nobody trained it on."
)

_DRIFT_LOOKING = {
    "fluid": (
        "Vorticity, which is how fast the fluid is turning at each point. Red turns one "
        "way, blue the other. The top row is the true flow and the bottom row is the "
        "prediction."
    ),
    "nbody": (
        "Point masses under gravity, seen from above. Filled dots are the true positions "
        "and open circles are the predicted ones. They start on top of each other."
    ),
}

_DRIFT_MEANING = {
    ("fluid", "convolution", "test"): (
        "The large structures survive, but by the end the prediction is smoother than the "
        "truth and a faint checkerboard has appeared. It is tracking the flow while "
        "inventing texture the physics never produced."
    ),
    ("fluid", "convolution", "held_out"): (
        "It stays stable on a flow it never trained on, which none of the other networks "
        "manage. It is still not accurate, and repeating the last state stays usable "
        "nearly three times as long."
    ),
    ("fluid", "operator", "test"): (
        "Two of the four rollouts leave the physical range altogether. That is not "
        "degradation, it is failure, and it happens on the setup the model trained on."
    ),
    ("fluid", "operator", "held_out"): (
        "All four rollouts diverge by step 28. The architecture expected to generalise "
        "best generalised worst."
    ),
    ("fluid", "persistence", "test"): (
        "The free baseline: return the state unchanged. It never diverges, and it stays "
        "usable slightly longer than the convolutional network trained to beat it."
    ),
    ("fluid", "persistence", "held_out"): (
        "On a flow that evolves slowly, returning the state unchanged is a genuinely good "
        "answer. It beats both trained networks here."
    ),
    ("fluid", "noise", "test"): (
        "A deliberately broken predictor, included so that the metrics can be checked "
        "against something that must score badly. A metric that cannot fail is not a "
        "metric."
    ),
    ("fluid", "noise", "held_out"): (
        "The same broken predictor on the unseen flow. Every metric has to catch it, and a "
        "test asserts that each one does."
    ),
    ("nbody", "graph", "test"): (
        "The cluster keeps roughly the right size and shape, but by the end individual "
        "bodies are visibly in the wrong places. The shape of the cluster outlasts the "
        "paths of the bodies in it."
    ),
    ("nbody", "graph", "held_out"): (
        "Two tight binaries, a configuration no training trajectory contained. All four "
        "rollouts leave the physical range, and nothing in the training numbers predicted "
        "it."
    ),
    ("nbody", "persistence", "test"): (
        "The free baseline: the bodies simply do not move. The graph network beats it "
        "here, which is the one clear win in the project."
    ),
    ("nbody", "persistence", "held_out"): (
        "Frozen bodies still score better than the graph network here, because the graph "
        "network's answer has left the physical range altogether."
    ),
    ("nbody", "noise", "test"): (
        "A deliberately broken predictor, included so that the metrics can be checked "
        "against something that must score badly."
    ),
    ("nbody", "noise", "held_out"): (
        "The same broken predictor on the unseen setup, as a control on the metrics rather "
        "than on the model."
    ),
}

TRUST = Section(
    anchor="trust",
    kicker="How far you get",
    heading="Usable for a fraction of the rollout",
    paragraphs=(
        "A rollout is one prediction run forward from a starting state, step after step, "
        "with nothing fed in but its own last answer. It counts as usable while the error "
        "stays below a tenth of the size of the state itself. Past that the mistake is "
        "comparable to the thing being predicted. Each bar is the length of that usable "
        "stretch, drawn against the full rollout the model was asked for.",
    ),
)

TRUST_CHART = Figure(
    title="Usable steps out of the full rollout",
    subtitle="Higher is better. The grey track is the length of the rollout requested.",
    alt=(
        "Bar chart of the stretch each model stays usable for, against the full rollout "
        "it was asked for. {summary}"
    ),
    caption=(
        "**Read the grey bars first.** Doing nothing at all, returning the previous state "
        "unchanged, matches or beats every network here on the setup it never trained on. "
        "A surrogate that cannot clear that line has not earned its training cost."
    ),
)

TRUST_LEGEND = ("{system}, trained on setup", "{system}, unseen setup")
"""What the two bars of a row mean, once per system so that the colours can be told
apart."""

TRUST_BASELINE = "Repeat last state, the free baseline"

TRUST_SUMMARY = "{system}: the best model stays usable for {steps} of {total} steps."
"""One clause of the chart's text alternative, per system. The system opens the clause for
the same reason it opens `COST_ASIDE`."""

TRUST_NOTE = (
    "Repeat last state is the free baseline: predict no change at all. Anything below it "
    "is not worth training. Where a row has no second bar, that model could not run on the "
    "unseen setup at all."
)

TRUST_ROLLOUT = "rollout of {steps} steps"

TRUST_NONE = "none"
"""What a row prints where a model could not run on a split at all, rather than a zero,
which would say it ran and was wrong immediately."""

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

COST_CHART = Figure(
    title="Accuracy against time per step, {system}",
    subtitle=(
        "Down and to the left is better: more accurate, less time. {threads}, {trials} "
        "timed trials."
    ),
    alt=(
        "Scatter of worst error against milliseconds per step. {summary} Both are drawn "
        "against the simulator's own quality dial, which is the real competition."
    ),
    caption=(
        "**The surrogates sit above and to the right of the simulator's curve**, which is "
        "the losing quadrant: slower and no more accurate. The {predictor} takes "
        "{surrogate} ms per step where the simulator reaches the same accuracy in "
        "{matched} ms. {payback}"
    ),
)

COST_LEGEND = ("Simulator, quality dial from low to high", "Neural surrogate")

COST_SUMMARY = "The {predictor} costs {surrogate} ms per step against the simulator's {matched} ms."
"""One clause of the chart's text alternative, for the comparison it draws in full."""

COST_AXES = ("worst error over the rollout", "milliseconds per step, log scale")
"""The two axis titles, up the side and along the bottom."""

COST_LADDER_ENDS = ("lowest quality", "exact, the reference")
"""What the two ends of the simulator's quality dial are called."""

COST_MATCHED = "same accuracy, {slowdown} the time"
"""The dashed connector between a surrogate and the solver setting as accurate as it."""

COST_PAYBACK = (
    "Counting the cost of training them, the cheapest repays itself after about {rollouts} "
    "rollouts."
)

COST_NEVER_PAYS = (
    "Counting the cost of training them, none of them ever pays back, at any number of rollouts."
)
"""The two ends of the same sentence. Which one the caption carries is read from the
benchmark, because a surrogate that never pays back and one that pays back after ten
thousand rollouts are different results and the page must not print the kinder one."""

COST_THREADS = ("One CPU thread", "{threads} CPU threads")

COST_ASIDE_TITLE = "The case the chart cannot settle"

COST_ASIDE = (
    "The {system} benchmark puts the {predictor} at {speed} the simulator's speed, at "
    "matched accuracy."
)
"""The system opens the sentence rather than sitting inside it, because a system is named
as `N-body` and `Fluid` and a capital letter mid sentence reads as a mistake."""

COST_ASIDE_BOUND = (
    "No solver setting ran both slower and less accurately than it, so that figure is a "
    "bound rather than a measurement, and a result that straddles one is not a speedup."
)

COST_ASIDE_PAYBACK = (
    "Even at that figure it would need about {rollouts} rollouts to repay the cost of training it."
)

COST_ASIDE_NEVER = "Counting the cost of training it, it never pays back at any number of rollouts."

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

DIAGNOSIS_CHART = Figure(
    title="Naming the true cause, over {faults} injected faults",
    subtitle=(
        "Each diagnoser returns the causes it thinks most likely, in order. Higher is better."
    ),
    alt=(
        "Bar chart comparing the agent with the rule based baseline. {summary} Both were "
        "given the same faults and scored by the same code."
    ),
    caption=(
        "**The baseline is not a straw man.** It names whichever metric regressed most and "
        "maps it to the cause that metric is usually about, using a table written with "
        "knowledge of which faults were coming. It still failed to name the cause at all "
        "in {missed} of {faults}."
    ),
)

DIAGNOSIS_LEGEND = ("Language model agent", "Conventional rule based baseline")

DIAGNOSIS_ROWS = ("Correct cause ranked first", "Correct cause in its top three")
"""The two measures the chart draws, in the order it draws them."""

DIAGNOSIS_SUMMARY = "{measure}: the agent {agent}, the baseline {baseline}."
"""One clause of the chart's text alternative, per measure."""

DIAGNOSIS_AXIS = "{faults} of {faults} faults"

DIAGNOSIS_TABLE = Table(
    caption="Every fault, and where the agent ranked the true cause.",
    headers=("What was broken", "In plain terms", "Findable in the config?", "Agent's rank"),
)

UNKNOWN_RANK = "not named"
"""What the table prints where the diagnoser never named the true cause at all."""

DIAGNOSIS_CLOSING = (
    "**The row it got wrong is the most interesting one.** On the lost optimiser state it "
    "ranked the true cause third, arguing that the two training error curves start at the "
    "same value, finish within half a per cent of each other and show no step where the "
    "run was resumed. That reasoning is correct. On a model this small the fault "
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
            "win, giving nearly the same answer when the grid is made finer, is real and "
            "large, and is the only thing its extra machinery bought."
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
            "Averaging four networks that have all left the physical range gives an answer "
            "outside it too. The average diverged a handful of steps later than the single "
            "network, which is not a rescue."
        ),
    ),
    # docs/results.md, "What did not work": calibration can be bought with vagueness.
    Finding(
        what="Confidence can be bought with vagueness, and was",
        why=(
            "The fluid ensemble's error bars cover the truth 96 per cent of the time, "
            "while being 159 times the size of the state being predicted. That is a model "
            "saying the answer might be anywhere. Every figure for how often the error "
            "bars cover the truth is printed beside how wide they had to be, for exactly "
            "that reason."
        ),
    ),
    # docs/results.md, "What did not work": the uncertainty warns in time on one system and
    # not the other.
    Finding(
        what="Uncertainty warned too late on one of two systems",
        why=(
            "The fluid ensemble's stated uncertainty grows past the usable limit before "
            "its error does, which is a warning in time to act on. The N-body ensemble's "
            "grows past it after. A surrogate that notices it has gone wrong only once it "
            "has gone wrong cannot decide when to hand back to the simulator."
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

THEME_LABEL = "Switch to the {theme} theme"
"""What the theme button is called. One word is enough to read beside the rest of the
navigation and is not enough to act on when it is heard on its own."""

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


def drift_looking(system: str) -> str | None:
    """Say what the reader is looking at in one system's state comparison.

    Args:
        system: The system as the record names it.

    Returns:
        The note, or `None` where the system has none. The caller decides what a missing
        note means, because an empty note under an image is worse than not offering the
        image at all.
    """
    return _DRIFT_LOOKING.get(system)


def drift_meaning(system: str, predictor: str, split: str) -> str | None:
    """Say what one combination shows.

    Args:
        system: The system as the record names it.
        predictor: Registered predictor name.
        split: Split name.

    Returns:
        The note, or `None` where the combination has none.
    """
    return _DRIFT_MEANING.get((system, predictor, split))


def _sentence(identifier: str) -> str:
    """A recorded identifier as a sentence, so `wrong_regime` reads as `Wrong regime`."""
    spaced = identifier.replace("_", " ")
    return spaced[:1].upper() + spaced[1:]
