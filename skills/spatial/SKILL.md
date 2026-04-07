---
name: spatial
description: >-
  Orchestrates spatial (frame-level) CV sub-skills for Viagent. When the spatial
  analyst runs, a vision LLM uses preview frames plus viagent_routing text here
  to choose which sub-skills to execute.
license: Proprietary — Viagent
metadata:
  project: viagent
  analyst_key: spatial
---

# Spatial analyst — skill orchestration

This package follows the [Agent Skills](https://agentskills.io/specification) layout. **Executable code** lives in `src/skill/spatial/`; this file defines **which Python sub-skills exist** and **when each is appropriate**, so a router can activate a subset per video.

## Machine-readable routing

Viagent parses the YAML block below (first fenced `yaml` block that contains `viagent_routing`). Each `id` must match a key in the spatial pipeline.

```yaml
viagent_routing:
  version: 1
  subskills:
    - id: ela
      when: >-
        Error Level Analysis along face/foreground boundaries. Use for almost all
        videos; best default for composite edges and halo artifacts. Safe general
        spatial check.
    - id: patch
      when: >-
        Local patch texture inconsistency inside facial regions. Prefer when faces
        or skin are visible; less informative for distant landscapes with no faces.
    - id: boundary
      when: >-
        Edge melting / abnormal face-to-background transition. Prefer when there is
        a clear subject and background; skip or deprioritize for flat abstract
        graphics or text-only slides.
    - id: edge
      when: >-
        Local edge coherence (soft/noisy boundaries, possible floating objects).
        General spatial cue; useful for natural scenes and talking-head videos.
    - id: multi_face
      when: >-
        Multi-face geometric collapse / inconsistency across frames. Run only when
        multiple faces or crowded people are plausible; omit for single-subject
        tutorials, screens, gameplay without faces, or nature-only footage.
    - id: blur
      when: >-
        Laplacian blur uniformity (face vs background sharpness). Requires face
        regions for full value; pair with multi_face when possible. Omit when no
        faces expected (reduces misleading ratios).
```

## Runtime (Viagent)

Each analyst calls a **vision sub-skill router** before CV: a multimodal LLM sees a few preview frames (chronological spread) plus this file’s `viagent_routing` text and JSON metadata, then returns which sub-skills to run.

Optional `llm` YAML keys: `skill_router_temperature`, `skill_router_max_preview_frames`. Code: `src/agents/routing/router_llm.py`.

## References

Add per-skill deep docs under `references/` as needed (one level from this file).
