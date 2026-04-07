---
name: temporal
description: >-
  Orchestrates temporal CV sub-skills (LPC, feature-point tracking) for Viagent.
  A vision LLM uses preview frames plus this file’s viagent_routing to pick
  which sub-skills run.
license: Proprietary — Viagent
metadata:
  project: viagent
  analyst_key: temporal
---

# Temporal analyst — skill orchestration

Executable code: `src/skill/temporal/`. This file defines **when** each sub-skill is appropriate for LLM or explicit routing.

```yaml
viagent_routing:
  version: 1
  subskills:
    - id: lpc
      when: >-
        Local phase coherence around detected faces across frames. Use when faces
        or vlogs/talking-head are plausible. Omit for scenery-only, slides, or
        screen recordings with no people.
    - id: feature_stability
      when: >-
        Feature point tracking stability across frames. Useful when there is
        texture and motion; weak for static slides or blank backgrounds—may still
        run if motion exists.
```

## Runtime

Sub-skills are chosen by the **vision router** (preview frames + this file). Optional tuning: `llm.skill_router_temperature`, `llm.skill_router_max_preview_frames`.
