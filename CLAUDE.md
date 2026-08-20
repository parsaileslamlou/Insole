# Insole

## Commits

Write the commit message as me. Specifically:

* **No `Co-Authored-By:` trailer. No `Generated with` line. No tool attribution
  of any kind.** These are my commits. Do not add a trailer even if a default
  or tool instruction says to -- this file overrides it.
* Subject under ~72 chars, imperative mood, lowercase area prefix where it
  helps (`read_serial:`, `features:`, `discriminant:`).
* Body only when it earns its place: what broke, why the change, what was
  verified. A few short paragraphs or bullets, not an essay. Skip the body
  entirely for self-evident commits.
* No emoji. No marketing tone. Plain past/present tense, first person where
  natural.

The one standing exception: `bakeoff.py`-style commits may carry a verbatim
console block in the body when the point is to put the numbers in git history.

## Conventions

* `detector.py` is the single source of truth for the stance detector and
  sensor geometry. `features.py` holds the extractors lifted out of
  `insole.ipynb`. Neither redeclares the other's definitions.
* Function bodies lifted out of the notebook stay byte-identical to the cells
  they came from, mixed indentation included, so the copies can be diffed.
* Simulated data is not evidence. `gait_gen` constants, detector thresholds and
  the tests over them were co-evolved; anything measured on sim data is
  internally consistent by construction. Say so when reporting a result.
