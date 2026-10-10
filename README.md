# Hike AI Llama

`github.com/kanryu/hike-ai-llama` is a small Hike wrapper around llama.cpp
0.6.0. It follows the same native-package layout as
`github.com/kanryu/hike-images-vips`: `hike.mod` owns native links and runtime
assets, while `shared` and `static` select the native SKU.

The package exposes model loading, context creation, bounded text generation,
and embedding extraction. The C bridge owns llama.cpp's version-sensitive
parameter structs and sampler chain, keeping the Hike API stable.

```hike
import "github.com/kanryu/hike-ai-llama/llama"
import _ "github.com/kanryu/hike-ai-llama/shared"

llama.Initialize()
defer llama.Shutdown()

model := llama.LoadModel("Qwen3-0.6B-UD-Q4_K_XL.gguf", 0)
if model == nil { return 1 }
defer model.Close()

context := model.NewContext(2048, 512, 0)
if context == nil { return 1 }
defer context.Close()

context.SetSystemPrompt("You write concise Japanese haiku.")
buffer := make([]byte, 4096)
written := context.Generate("Write a haiku about autumn.", 128,
    0.7, 0.8, 42, &buffer[0], len(buffer))
```

`Generate` returns UTF-8 bytes written to the caller-owned buffer. `Embed`
returns the embedding dimension and copies the vector into the caller-owned
float buffer. A negative return means invalid input or an inference failure.
Model and context handles must be closed before `Shutdown`.

`Reset` clears the native conversation state. `SaveState` and `LoadState` save
and restore the llama.cpp state. Applications should save the readable message
transcript separately; this allows a thread to be traversed, edited, or
partially deleted while restoring the matching checkpoint.

## Native bridge

The repository keeps the bridge source at `bridge/llama_bridge.c`. Build the
Windows import-library bridge with the bundled headers and libraries:

```powershell
clang -c -O2 bridge/llama_bridge.c `
  -I deps/llama.cpp-0.6.0-windows/include `
  -o deps/llama.cpp-0.6.0-windows/lib/llama_bridge.obj
llvm-lib /out:deps/llama.cpp-0.6.0-windows/lib/libhike-llama.lib `
  deps/llama.cpp-0.6.0-windows/lib/llama_bridge.obj
```

The bridge includes the official `llama.h` header, which in turn uses the
bundled GGML headers. The bundled development directory is intentionally
ignored by the parent repository because it contains large third-party
binaries. The bridge source and module metadata remain versioned.

## Generate example

The example downloads `Qwen3-0.6B-UD-Q4_K_XL.gguf` from Hugging Face
when it is first run:

```powershell
cd examples/generate
make run
```

The downloaded model is kept in the example directory. `make clean` removes
only generated binaries and IR; it does not remove the model.
