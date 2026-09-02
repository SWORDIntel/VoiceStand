# VoiceStand documentation

VoiceStand is currently a CPU-first, fully local Linux dictation project. The authoritative documents for the active implementation are:

1. [Current architecture](CURRENT_ARCHITECTURE.md)
2. [Project status and production gates](PROJECT_STATUS.md)
3. [Benchmark evidence](BENCHMARKS.md)
4. [CPU-first redesign specification](CPU_FIRST_PRODUCTION_REDESIGN.md)

## Current implementation documents

- `CURRENT_ARCHITECTURE.md` describes what is implemented, the measured ASR backend decision, data flow, safety boundaries, and the next migration.
- `PROJECT_STATUS.md` is the concise phase ledger. It does not claim production readiness before manual desktop and recovery gates pass.
- `BENCHMARKS.md` records reproducible corpus sources, commands, measurements, scope, and known limitations.
- The root [`README.md`](../README.md) contains setup, operation, and local CI instructions.

## Historical documents

The `architecture/`, `deployment/`, `implementation/`, `phases/`, `reports/`, and much of `technical/` preserve earlier Intel NPU/GNA design and completion reports. They are retained as project history and may describe prototypes, targets, or claims that are not part of the current supported product.

In particular, historical statements such as “production complete,” `<10 ms` end-to-end transcription, mandatory Meteor Lake hardware, or finished NPU/GNA inference must not be used as current status. When a historical document conflicts with the four authoritative documents above, the current documents win.

## Documentation policy

- Measured results must identify the backend, model, input, host scope, and metric boundary.
- A local green CI run is not described as a hosted GitHub Actions result.
- Implemented, benchmarked, integrated, manually accepted, and production-ready are distinct states.
- Raw test audio is committed only under `benchmarks/corpus/` with provenance and SHA-256 checksums.
