# cognee-litert-lm

Safe Rust bindings for the [LiteRT-LM](https://github.com/topoteretes/LiteRT-LM) on-device LLM inference engine, built on top of the `cognee-dev` branch of the Cognee fork.

LiteRT-LM is Google's on-device LLM runtime (successor to MediaPipe LLM Inference). This crate wraps the C API exposed by the fork's `liblitert_lm_c.so` shared library with idiomatic Rust types, RAII resource management, and an optional high-level `Conversation` API with structured/constrained decoding support.

## Architecture

```
cognee-litert-lm  (this crate)
    └── FFI bindings (bindgen)  →  liblitert_lm_c.so
                                       └── vendor/LiteRT-LM  (cognee-dev branch)
                                               ├── c/engine.cc          C API implementation
                                               ├── c/liblitert_lm_c.lds version script
                                               └── runtime/             LiteRT-LM C++ core
```

The `vendor/LiteRT-LM` submodule tracks the `cognee-dev` branch of `github.com/topoteretes/LiteRT-LM`, which adds:

- A stable C API (`c/engine.cc`) suitable for FFI from Rust and other languages
- A `cc_binary` Bazel target that produces a self-contained `liblitert_lm_c.so`
- Constrained decoding support via `SetConstraintProviderConfig` (llguidance-based JSON schema enforcement)
- A version script (`c/litert_lm_c.lds`) that exports `LiteRt*` and `litert_lm_*` symbols globally, fixing GPU plugin loading when the library is `dlopen`'d from a Rust process (see [prebuilt/android_arm64/README.md](vendor/LiteRT-LM/prebuilt/android_arm64/README.md))

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `async-constraint-masking` | ✅ | Overlap constraint mask computation with GPU decode on Android — eliminates the CPU stall between tokens for constrained output |
| `build-from-source` | ❌ | Build the C++ library via CMake instead of linking a prebuilt |
| `clap` | ❌ | Enable CLI arg parsing in examples |
| `serde_json` | ❌ | Enable JSON utilities in examples |

## Prerequisites

- Android NDK r27+ (for Android targets)
- Bazel 7+ (to build `liblitert_lm_c.so` from the submodule)
- A `.litertlm` model file (e.g. `gemma3-1b-it-int4.litertlm`)

Point Cargo at the Bazel-built shared library before building:

```bash
export LITERT_LM_LIB_DIR=/path/to/vendor/LiteRT-LM/bazel-bin/c
```

## Quick Start

### Low-level session API

```rust
use cognee_litert_lm::{Backend, EngineSettings, InputKind};

let settings = EngineSettings::new("model.litertlm", Backend::Gpu, None, None)?;
let engine = settings.build()?;
let session = engine.create_session(None)?;

let responses = session.generate_content(&[InputKind::Text("Hello!")])?;
println!("{}", responses.first_text()?);
```

### High-level Conversation API with constrained decoding

```rust
use cognee_litert_lm::{
    Backend, ConstraintType, Conversation, ConversationConfig,
    EngineSettings, OptionalArgs,
};

let settings = EngineSettings::new("model.litertlm", Backend::Gpu, None, None)?;
let engine = settings.build()?;

let config = ConversationConfig::new(
    "You are a helpful assistant.",
    None, None, None,
)?;
let conversation = Conversation::new(&engine, Some(config))?;

// Add a JSON schema constraint so the model output is valid JSON
let schema = r#"{"type":"object","properties":{"answer":{"type":"string"}}}"#;
let mut args = OptionalArgs::new()?;
args.set_constraint(ConstraintType::JsonSchema(schema.to_string()))?;

let response = conversation.send_message_with_args(
    r#"{"role":"user","content":"What is 2+2?"}"#,
    &args,
)?;
println!("{}", response.as_str().unwrap_or(""));
```

### Streaming

```rust
session.generate_content_stream(
    &[InputKind::Text("Tell me a story.")],
    |token, is_done, _| { print!("{token}"); },
)?;
```

### Benchmarking

```rust
let info = session.get_benchmark_info()?;
println!("Prefill: {:.1} tok/s", info.prefill_tokens_per_sec_at(0));
println!("Decode:  {:.1} tok/s", info.decode_tokens_per_sec_at(0));
```

## Android Benchmark

`benchmark_android.sh` cross-compiles the `simple_chat` example and runs a knowledge-graph extraction benchmark on a connected Android device across CPU, GPU sync, and GPU+spec (speculative constrained decoding) backends.

```bash
# Full build + push + run
ANDROID_NDK_HOME=~/Android/Sdk/ndk/29.0.13599879 ./benchmark_android.sh

# Skip rebuild/repush if binaries are already on device
./benchmark_android.sh --skip-build --skip-push
```

Results on Snapdragon 8 Gen 3 (GPU+spec, Gemma 3 1B INT4):

| Backend | Prefill | Decode |
|---------|---------|--------|
| GPU+spec (async constraint masking) | ~708 tok/s | ~23 tok/s |

## GPU Sampler Fix

When `liblitert_lm_c.so` is loaded by a Rust process, Android's linker namespace isolation prevents `libLiteRtTopKOpenClSampler.so` (the GPU TopK sampler plugin) from resolving `LiteRt*` symbols at `dlopen` time, causing a silent fallback to the CPU sampler and a ~40% decode regression.

The fix is documented in [vendor/LiteRT-LM/prebuilt/android_arm64/README.md](vendor/LiteRT-LM/prebuilt/android_arm64/README.md) and consists of:

1. Exporting `LiteRt*` symbols globally from `liblitert_lm_c.so` via the version script
2. An ELF binary patch on the prebuilt sampler SO that adds `liblitert_lm_c.so` as a `DT_NEEDED` dependency

The patch script lives at `tools/patch_sampler_so.py` and should be re-run if the upstream prebuilt is ever refreshed.

## License

Apache 2.0 — see [LICENSE](LICENSE).
