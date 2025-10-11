// Build script for voicestand-hardware
// Compiles stub implementations of NPU/GNA C libraries

fn main() {
    // Compile stub implementations for NPU and GNA
    cc::Build::new()
        .file("src/stubs.c")
        .compile("hardware_stubs");

    println!("cargo:rerun-if-changed=src/stubs.c");
}
