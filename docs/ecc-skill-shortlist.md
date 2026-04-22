# ECC Shortlist for Edge-Based Hand Tracking Game

## Stack

- Target project: Python desktop game for nutrition education
- Core constraints: CPU-only, low latency (30 FPS or above), stable real-world tracking
- Core technologies: OpenCV, CamShift, Kalman filtering, image processing, Windows `.exe` packaging
- Development priorities: TDD, SOLID-style modular design, performance profiling, verification before packaging

## Daily

### Skills

- `product-capability`
  - Why: your project definition is already close to a PRD; this skill helps translate it into an implementation-ready capability spec.
- `coding-standards`
  - Why: gives baseline naming, readability, modularity, and maintainability rules that map well to SOLID-oriented development.
- `documentation-lookup`
  - Why: useful when we need current docs for OpenCV, PyInstaller, pytest, mypy, or packaging tools.
- `tdd-workflow`
  - Why: direct match for your request to include TDD development.
- `eval-harness`
  - Why: helps define measurable pass/fail criteria such as FPS, tracking stability, false-loss rate, and packaging success.
- `verification-loop`
  - Why: adds a repeatable verify-before-release loop for tests, typing, and build checks.
- `security-review`
  - Why: still relevant for file access, config handling, assets, and packaging even in a local desktop game.

### Agents

- `planner.md`
  - Why: good first-pass breakdown for feature order such as tracker core, interaction layer, game loop, and packaging.
- `architect.md`
  - Why: strongest ECC match for SOLID-oriented decomposition into tracker, state estimator, gesture/game logic, and UI layers.
- `python-reviewer.md`
  - Why: mandatory-level fit for a Python-heavy codebase.
- `tdd-guide.md`
  - Why: agent version of the TDD workflow for test-first implementation.
- `performance-optimizer.md`
  - Why: critical for FPS, frame pipeline cost, ROI handling, and CPU bottlenecks.
- `code-reviewer.md`
  - Why: general post-change review.
- `security-reviewer.md`
  - Why: catches unsafe file/process/config patterns before release.
- `silent-failure-hunter.md`
  - Why: especially useful for tracking systems where lost detections, fallback states, and swallowed errors can quietly ruin gameplay.

## Library

### Skills

- `backend-patterns`
  - Why: not a perfect fit for a local game, but still useful if we later split capture, tracking, scoring, and persistence into clearer service layers.
- `deep-research`
  - Why: optional for literature review on hand tracking heuristics, nutrition education design, or user-study framing.
- `agent-sort`
  - Why: useful later if we want to trim or reclassify the ECC surface again.

### Agents

- `refactor-cleaner.md`
  - Why: useful once prototypes accumulate duplicated tracking logic.
- `code-simplifier.md`
  - Why: a practical SOLID-adjacent helper for reducing complexity while preserving behavior.

## TDD and SOLID Notes

- TDD is directly covered by:
  - `tdd-workflow`
  - `tdd-guide.md`
  - `eval-harness`
  - `verification-loop`
- ECC does not appear to ship a dedicated `SOLID`-named skill in this repo snapshot.
- The best SOLID-oriented combination for this project is:
  - `architect.md`
  - `coding-standards`
  - `code-simplifier.md`
  - `refactor-cleaner.md`
  - `python-reviewer.md`

## Gaps We Still Need to Cover

- No dedicated ECC skill found for:
  - OpenCV tracking architecture
  - CamShift tuning workflow
  - Kalman filter integration patterns for CV tracking
  - PyInstaller or Windows `.exe` packaging
  - Real-time computer-vision game-loop design
- Because of that, the next high-value step is to create a custom project skill that bundles:
  - tracking pipeline conventions
  - FPS/performance checklist
  - test matrix for camera-free simulation and live-camera validation
  - packaging checklist for Windows executable output

## Install Plan

- Keep these copied into the workspace for immediate use:
  - `collected-ecc/skills/`
  - `collected-ecc/agents/`
- Treat `collected-ecc/skills-library/` as searchable reference, not always-on context.
- Use the collected ECC set as source material for a custom project skill rather than loading the whole upstream repo every time.

## Verification

- Verified locally that the upstream repo was cloned into `vendor/everything-claude-code`.
- Verified TDD coverage exists in both a skill and an agent:
  - `tdd-workflow`
  - `tdd-guide.md`
- Verified there is no obvious dedicated `SOLID`-named skill or agent in the collected ECC surface.
- Verified the selected files were copied into `collected-ecc/`.
