# Agent Q: The "5-Toy" Curriculum & Logic Bundle

This workspace contains the starter kit for **Agent Q**, a pedagogical framework for understanding AI agency through deterministic, replayable **Toy Models**.

## 1. The 5 Toy Models

These definitions are located in `toys/starter_pack.json`.

| Toy ID | Name | Focus | Key Concepts |
| :--- | :--- | :--- | :--- |
| `toy.steering.v1` | **The Steering Problem** | Policy Learning | Credit assignment, reward shaping, failure analysis |
| `toy.rag_desk.v1` | **Memory & Retrieval Desk** | Retrieval (RAG) | Embedding distance, hallucination, grounding checks |
| `toy.swarm_turns.v1` | **Turn-Based Swarm** | Multi-Agent | Routing, conflict resolution, identity separation |
| `toy.boundary_pipe.v1` | **BoundaryPipe Runtime** | Validation | Invariants, audit logs, deterministic pipelines |
| `toy.bom_assembler.v1` | **BOM Assembler** | Composition | Assembly constraints, safe mutation (hinges) |

## 2. Directory Structure (Proposed)

```text
agentq/
  toys/
    starter_pack.json       <-- DEFINITIONS (Ready)
  runtime/                  <-- ENGINE
    engine.py               (Deterministic simulator)
    invariants.py           (Validation gates)
    logging.py              (Append-only event log)
    replay.py               (State reconstruction)
  agent/                    <-- WRAPPER
    wrapper.py              (LLM / Rule-based agent)
    prompts/
    policies/
  ui/
    tui.py                  (Terminal UI)
    web/
  runs/                     <-- LOGS
    2026-01-09_run_001/     (Example run)
```

## 3. Visual Identity (Blueprint + LEGO)

* **Colors**: Blueprint Navy background, Grid Cyan lines, Safety Yellow warnings.
* **Iconography**: LEGO "studs" for tokens, "Hinges" for safe mutation points.
* **Philosophy**: "Blueprint Realism" — clean engineering diagrams with playful discreteness.

## 4. Curriculum Roadmap

* **Agent Q 101**: Toy Worlds & State (Steering, BOM)
* **Agent Q 201**: Retrieval & Grounding (RAG Desk)
* **Agent Q 301**: Multi-Agent Systems (Swarm)
* **Agent Q 401**: Auditable Pipelines (BoundaryPipe)

## 5. Live Demo Script (Summary)

1. **"Understanding is a Debugger"**: Start with the `Steering Problem`. Show a failure (grip drop). Replay it. Fix it.
2. **"Hallucinations you can see"**: Use `RAG Desk`. Inject a bad card. Watch the retrieval fail. Fix it with a threshold.
3. **"Trust is a Log"**: Use `BoundaryPipe`. Break an invariant. Show the `ERROR` state. Prove the system refuses to lie.
