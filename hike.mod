module github.com/kanryu/hike-ai-llama

hike 1.0

// The native configuration is activated by importing shared or static from
// an application. The bridge keeps llama.cpp's C structs and sampler ABI out
// of Hike source; callers get a small, stable handle-based API instead.
package ./shared {
    target windows {
        link: "deps/llama.cpp-0.6.0-windows/lib/libhike-llama.lib"
        link: "deps/llama.cpp-0.6.0-windows/lib/llama.lib"
        link: "deps/llama.cpp-0.6.0-windows/lib/llama-common.lib"
        link: "deps/llama.cpp-0.6.0-windows/lib/ggml.lib"
        assets: "deps/llama.cpp-0.6.0-windows/bin/*.dll"
    }
}

package ./static {
    // A static llama.cpp bundle can provide the same bridge symbols. The
    // shipped Windows development bundle is shared, so shared is the ready
    // to use SKU and this package is reserved for a future static bundle.
    target windows {
        link: "deps/llama.cpp-0.6.0-windows/lib/libhike-llama.lib"
        link: "deps/llama.cpp-0.6.0-windows/lib/llama.lib"
        link: "deps/llama.cpp-0.6.0-windows/lib/llama-common.lib"
        link: "deps/llama.cpp-0.6.0-windows/lib/ggml.lib"
    }
}
