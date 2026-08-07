# Landing page plan

Seven phases, numbered 12 to 18, continuing the plan in `docs/plan/README.md`.
Each phase is one session in a fresh chat. This file is the whole plan: there is
no separate brief per phase.

## What is being built

A generated landing page for the repository. It answers one question, can a
neural network replace a physics simulator, and it answers it for a reader who
knows none of the vocabulary. Every number on it comes from a run record. It
links to the per run reports, which already share its styling.

The visual reference is `docs/design/landing-page.html`, a hand written mockup
with the numbers typed in. The finished page must match it section for section
and pixel for pixel where the design is fixed, and must derive every number
rather than repeat it. Where a typed number in the mockup disagrees with a
record, the record wins and the phase notes the correction.

## How to run a phase

Open a new chat in this repository and say:

> Read CLAUDE.md and docs/plan/landing-page.md, then implement phase NN.

`CLAUDE.md` carries the writing style and the engineering standards.
`docs/plan/00-architecture.md` is worth reading for phases 12 and 13, and is not
needed for the rest. Nothing else needs to be in context.

Work on a branch named after the phase, for example `phase-12-theme`. Open a
pull request at the end of each phase. Do not start a phase before the ones it
depends on are merged.

## How these phases are sized

Each phase is scoped for one session at medium reasoning effort. That has
consequences the briefs rely on.

- Every phase names the files it touches. If a phase seems to need a file it
  does not name, that is a signal the phase has been misread, not a licence to
  widen it.
- Every phase states its entry check: one command whose output tells you the
  previous phase landed. Run it first.
- No phase requires reading the whole source tree. Discovery is bounded to the
  reporting package and the two data files named below.
- Refactoring outside the named files is out of scope in every phase. If
  something else is wrong, write it down in the pull request and leave it.
- If a phase finishes early, spend the time on tests, not on the next phase.

## The two sources of truth

**Run records.** `runs/<run-id>/record.json`, read through the existing record
reader. Everything about accuracy, cost, divergence and provenance comes from
here.

**The fault scores.** `docs/results/fault-scores.json`, the scored diagnosis
agent results. This is the only committed data file the page reads.

Nothing else is a source. Prose that is genuinely editorial, the premise, the
findings list, the drift annotations, is curated text held in the source, and
each phase says which file holds it.

## Where the page is written

The page is written to the runs root as `index.html`, replacing the listing that
`serve.bat` writes inline today. That keeps every generated artefact under
`runs/`, which is not committed, and makes the links short: a report is
`<run-id>/report.html` and a plot is `<run-id>/plots/<name>.png`.

Images are referenced, not embedded. A per run report embeds its images as data
URIs because a report is meant to be sent to one person as one file. The landing
page references fourteen plots and would reach several megabytes if it embedded
them, and it is always served from the directory that holds them. State this
reason in the module docstring so the difference from the report is not read as
an inconsistency.

---

# Phase 12: Shared theme

**Objective.** Move the styling shared by the reports and the landing page into
one module, so the two cannot drift.

**Depends on.** Nothing. **Branch.** `phase-12-theme`

**Entry check.** Render any run report and confirm it opens in the current
style.

## Why this phase exists

The report stylesheet is currently a string constant inside the document
serialiser. The landing page needs the same colour tokens, the same three type
families and the same table and figure treatment, and a second copy would be
wrong within a week.

## Build

1. A new module in the reporting package holding the design tokens: the light
   and dark colour values, the serif, sans and mono stacks, and the rules that
   both artefacts share, meaning body text, headings, tables, figures and the
   masthead.
2. The document serialiser takes its stylesheet from that module instead of its
   own constant. Its output changes only in that the stylesheet now arrives from
   elsewhere.
3. The tokens are exposed as data as well as text, because the chart phases need
   the colour names in order to reference them from generated markup.

## Definition of done

- Rendering a report twice produces byte identical HTML.
- A rendered report contains no reference to any host: no fonts, no scripts, no
  stylesheets.
- The reporting package has one stylesheet. A search for a second colour literal
  outside the theme module finds nothing.
- The existing reporting tests pass unchanged. If a test asserts on the old
  stylesheet text, update the assertion to the new source, not the other way
  round.

## Out of scope

No visual change to the reports. This phase moves code and changes nothing a
reader would see.

---

# Phase 13: The page model

**Objective.** Turn a runs root into the typed values the landing page shows,
with no HTML anywhere in the phase.

**Depends on.** Phase 12. **Branch.** `phase-13-page-model`

**Entry check.** The reporting tests pass and the theme module exists.

## Why this phase exists

The mockup has about forty numbers typed into it. Every one has to be derived,
because a hand typed number is a number that is wrong after the next training
run. Separating the derivation from the rendering also makes the page testable
without parsing HTML.

## Build

1. **A summary per run.** For each run under the runs root: the system, the
   learned model and its parameter count, the run identifier, when it ran, the
   commit, and the paths to its report and its plots.
2. **Usable steps.** The harness reports a horizon in simulated time. The page
   states it in stored steps, because steps are what a reader can picture.
   Divide the horizon at the ten per cent error threshold by the dataset's
   stored step. Do this for the trained on split and the held out split, for the
   learned model and for the persistence baseline.
3. **The verdict.** One short phrase per run, derived rather than written:
   - a run with no learned predictor in its results is a harness check,
   - a run whose held out rollouts all fail to start cannot run unseen,
   - a run with any diverged rollout diverges, and the split it diverged on is
     part of the phrase,
   - a run that completes every rollout is stable, qualified by whether it beats
     the persistence baseline,
   - a run carrying a benchmark and no learned evaluation is a cost benchmark,
     and its verdict is read from the speedup instead.
4. **The headline figures.** The three numbers in the hero, each selected by a
   stated rule over the summaries rather than named: the longest usable stretch
   any surrogate reaches on its trained on split, the worst held out completion
   count, and the largest matched accuracy slowdown in the benchmarks.
5. **The cost ladder.** For a benchmark run, the solver settings and the
   surrogates as points of error against seconds per step, plus the matched
   comparison and the break even count.
6. **The diagnosis scores.** Read from the committed fault scores file: the two
   diagnosers, their top one and top three rates, and the per fault rank.

## Definition of done

- Every value the mockup shows has a named field in the model, and the model has
  no field the page does not show.
- Given a fixture runs root, the model is exactly reproducible. Same inputs,
  same output, no wall clock and no ordering that depends on the filesystem.
- A run missing a benchmark, a run missing a held out split and a run whose
  model refused to build all produce a summary rather than an exception. Each
  case has a test.
- The verdict rules have one test each, including the two that the current runs
  do not exercise.
- Compare every derived number against the mockup. Where they differ, the record
  is right. List the differences in the pull request.

## Out of scope

No HTML, no SVG, no CSS. A reviewer should be able to read this phase without
opening a browser.

---

# Phase 14: Page skeleton and static sections

**Objective.** Build the page frame and every section that is prose rather than
a chart.

**Depends on.** Phase 13. **Branch.** `phase-14-skeleton`

**Entry check.** The page model builds from the real runs root and its tests
pass.

## Build

1. **The frame.** The sticky navigation with the wordmark, the section links and
   the theme button, the page wrapper, and the footer. The theme button switches
   between the light and dark token sets and starts on whichever the reader's
   system asks for.
2. **The hero.** The eyebrow, the question as the title, the answer paragraph,
   the three headline figures from the model, and the note pointing at the
   diagnosis section.
3. **The premise section.** The three explanatory paragraphs and the schematic
   figure. The schematic is fixed artwork, not derived, and belongs beside the
   curated prose.
4. **The findings section.** The six negative results, each a heading and a
   paragraph, held as curated text in one file with a comment saying which part
   of `docs/results.md` each is drawn from.
5. **The run register.** One row per run from the model, grouped by system,
   linking to the report, showing the usable stretch, the verdict and the
   identifier.
6. **The curated text file.** All editorial prose on the page lives in one
   module. No sentence is written inline in a builder.

## Definition of done

- The page renders from a fixture runs root and from the real one.
- Output is deterministic and every value that came from a record matches the
  model.
- Every string that reaches the page is escaped. A run name containing a less
  than sign is a test.
- The page references no host. Only relative paths to reports and plots.
- Open it beside the mockup and compare the sections built in this phase. They
  should be indistinguishable apart from the charts, which are not built yet.

## Out of scope

The charts and the drift viewer. Leave a placeholder where each will go.

---

# Phase 15: Charts

**Objective.** Draw the three data charts as generated markup, from the model.

**Depends on.** Phase 14. **Branch.** `phase-15-charts`

**Entry check.** The page renders with placeholders where the charts go.

## Why this phase exists

These charts are the argument. A reader who reads nothing else should be able to
see that the surrogate is usable for a small fraction of the rollout, that it
loses to the solver on cost, and that the agent beats its baseline.

## Build

1. **A small drawing helper.** Enough to place a horizontal bar, a point, a
   line, a tick and a label inside a fixed viewport, with one linear scale and
   one logarithmic scale. It is not a plotting library and must not grow into
   one.
2. **The usable steps chart.** Two panels, one per system, each with its own
   axis in steps because the two rollout lengths differ. Two bars per predictor,
   the trained on split and the held out split. The persistence baseline is
   drawn in grey in both panels, because the reader is meant to compare against
   it first.
3. **The cost chart.** Error against time per step on a logarithmic time axis,
   the solver settings joined as a line, the surrogates as points, and the
   matched accuracy pair joined by a dashed connector carrying the slowdown.
4. **The diagnosis chart.** Two pairs of bars, the agent against the rule based
   baseline, on top one and top three.
5. **Colour and labelling.** Colours come from the theme tokens by name, never
   as literals. Every series is directly labelled as well as coloured, and every
   chart carries a text alternative that states its conclusion.

## Definition of done

- Given known inputs, each chart's geometry is asserted at a few points: a bar
  whose value is a tenth of the axis is a tenth of the track, a point at the
  axis minimum sits on the axis.
- A value outside the axis range raises rather than drawing outside the frame.
- No colour literal appears outside the theme module.
- Both themes are legible. Check the dark one by eye as well as by test.
- Compare each chart against the mockup. Geometry may differ where the model's
  numbers correct the typed ones. Nothing else may differ.

## Out of scope

No tooltips, no animation, no interaction of any kind in the charts.

---

# Phase 16: The drift viewer

**Objective.** Build the section that lets a reader watch a prediction come
apart, using the plots the reports already generate.

**Depends on.** Phase 15. **Branch.** `phase-16-drift`

**Entry check.** All three charts render from the real runs root.

## Build

1. **The manifest.** For each system, predictor and split, the relative path to
   the state comparison plot, built by looking at which plots actually exist
   rather than by assuming. A predictor with no plot is left out of the
   controls, not linked to a missing file.
2. **The controls.** Three groups: system, predictor, split. Selecting any of
   them swaps the image and the two notes below it.
3. **The notes.** Two per combination: what the reader is looking at, which
   depends on the system, and what it means, which depends on the combination.
   Curated text, in the same file as the rest of the prose.
4. **The behaviour.** The smallest script that swaps an image source and two
   blocks of text from a table written into the page. No framework, no fetch, no
   build step.

## Definition of done

- Every path in the manifest exists on disk at render time. A test asserts this
  against a fixture runs root.
- Every combination the controls offer has both notes. A missing note is a
  failure, not an empty box.
- The section works without the script in the sense that the first combination
  is already rendered into the page and readable.
- The controls are reachable and operable from the keyboard, and the pressed
  state is exposed to assistive technology.
- Compare against the mockup, including the plate the image sits on, which keeps
  the light backgrounds of the plots legible in the dark theme.

## Out of scope

No new plots. This phase consumes what the reporting phase already produces.

---

# Phase 17: Command and serving

**Objective.** Make the page a command, and delete the inline generator.

**Depends on.** Phase 16. **Branch.** `phase-17-command`

**Entry check.** The full page renders and matches the mockup.

## Build

1. **The command.** A report subcommand that reads a runs root and writes
   `index.html` into it. It takes the runs root and an optional output path, and
   nothing else. It fails loudly if the root holds no readable record.
2. **The batch file.** `serve.bat` calls the command and then serves the
   directory. The inline Python one liner is removed entirely.
3. **The listing command.** The existing run listing and the page now derive the
   same summaries. Point them at the same code rather than leaving two
   definitions of what a run's key numbers are.
4. **Documentation.** One short section in the reproduction document saying how
   to build and serve the page, and one line in the README.

## Definition of done

- The command runs on a fixture runs root in a temporary directory and produces
  a page whose links all resolve.
- Running it twice produces byte identical output.
- A runs root with no records produces a clear error naming the directory, not
  an empty page.
- The batch file contains no Python.
- Nothing generated is committed. Confirm the runs root is still ignored.

## Out of scope

No server, no watch mode, no deployment.

---

# Phase 18: Editorial and accessibility pass

**Objective.** Make the finished page read well and work for everyone. This is
the phase that decides whether the project reads as careful or as unfinished.

**Depends on.** Phase 17. **Branch.** `phase-18-polish`

**Entry check.** The command produces the page and every earlier phase is
merged.

## The writing brief

Every sentence on the page is rewritten to this standard. It is stricter than
the repository style because this page is read by people who do not work on
simulation.

- **UK English.** Artefact, behaviour, normalisation, optimise, analyse,
  modelling.
- **No dashes as punctuation.** No em dashes and no en dashes. Use a comma, a
  colon or a full stop.
- **Succinct.** Cut any sentence that does not add information. Cut any clause
  that repeats the sentence before it.
- **Plain English.** No jargon without its meaning attached. A reader who does
  not know what a rollout, a regime, an invariant or a surrogate is must still
  follow every claim on the page. Where a technical word is unavoidable, define
  it in the same sentence and then use it consistently.
- **No hype.** No marketing tone, no exclamation marks, no emoji. The results
  are negative and stating them plainly is the point.
- **Every figure carries its meaning.** Each chart has a line saying what it
  shows and a line saying what to conclude. A chart without both is unfinished.

## The accessibility brief

- Every image has alternative text that says what it shows, not what it is
  called.
- Every chart has a text alternative stating its conclusion, and a reader who
  cannot see colour can still tell the series apart, because each is labelled.
- Colour contrast meets the standard in both themes, for text and for the
  boundaries of marks.
- Every control is reachable by keyboard, in a sensible order, with a visible
  focus ring and a correctly reported pressed state.
- The page respects a request for reduced motion.
- Headings descend in order and the sections are landmarked, so the page can be
  navigated by structure.

## Also in this phase

- Check the page at a narrow width, a tablet width and a wide one. Nothing
  overflows sideways. Tables and charts scroll inside their own boxes.
- Check both themes on a real screen, not only in tests.
- Delete the mockup from the repository root if it is still there, and keep the
  reference copy under `docs/design/`.
- Re-read the finished page against `docs/results.md` and confirm no claim on
  the page is stronger than the claim in the results document.

## Definition of done

- A read through finds no dash used as punctuation, no Americanism and no
  sentence that could be shorter.
- An automated accessibility check on the rendered page reports no violations,
  and the manual keyboard pass is done and noted in the pull request.
- The page and the reports look like one artefact.
- The pull request describes what changed in the wording, so the editing is
  reviewable rather than invisible.

## Out of scope

No new sections, no new data, no design changes. If something looks wrong at
this stage, note it and leave it.
