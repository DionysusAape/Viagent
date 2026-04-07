---
name: physics
description: >-
  Orchestrates physics/motion CV sub-skills (optical flow, geometry stability,
  NSG-lite) for Viagent. A vision LLM uses preview frames plus this file to pick
  sub-skills.
license: Proprietary — Viagent
metadata:
  project: viagent
  analyst_key: physics
---

# Physics analyst — skill orchestration

Executable code: `src/skill/physics/`.

```yaml
viagent_routing:
  version: 1
  subskills:
    - id: optical_flow
      when: >-
        Dense / batch optical flow between sampled frames. Core motion-coherence
        signal; keep for most moving-camera or moving-subject footage. Deprioritize
        for near-static slide decks.
    - id: geometry_stability
      when: >-
        Background straight-line stability vs camera motion. Best for
        live-action-like perspective; less informative for flat 2D animation or
        heavy transition effects (still optional).
    - id: nsg_lite
      when: >-
        Lightweight physical-consistency check across frames (CPU). Optional when
        you want extra signal; omit to save compute on very long or static clips.
```

## Runtime

Vision router + this file. Optional: `llm.skill_router_temperature`, `llm.skill_router_max_preview_frames`.
