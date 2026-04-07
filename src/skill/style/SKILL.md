---
name: style
description: >-
  Orchestrates style / spectral sub-skills (FFT) for Viagent. A vision LLM
  uses preview frames plus this file to decide whether to run FFT.
license: Proprietary - Viagent
metadata:
  project: viagent
  analyst_key: style
---

# Style analyst - skill orchestration

Executable code: `src/skill/style/`.

```yaml
viagent_routing:
  version: 1
  subskills:
    - id: fft
      when: >-
        Global 2D FFT / frequency-domain cues for synthetic vs traditional
        rendering. Default-on for style analysis; reserve skipping for strict
        compute/policy limits only.
```

## Runtime

Vision router + this file. Optional: `llm.skill_router_temperature`, `llm.skill_router_max_preview_frames`.
