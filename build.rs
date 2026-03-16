use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor_dir = manifest_dir.join("vendor").join("LiteRT-LM");

    // --- Step 1: Build and link the C++ library ---
    #[cfg(feature = "build-from-source")]
    {
        let dst = cmake::Config::new(&vendor_dir)
            .define("CMAKE_BUILD_TYPE", "Release")
            .build();

        println!("cargo:rustc-link-search=native={}/lib", dst.display());
    }

    // Allow overriding the library search path via environment variable.
    // Usage: LITERT_LM_LIB_DIR=/path/to/lib cargo build
    if let Ok(lib_dir) = env::var("LITERT_LM_LIB_DIR") {
        println!("cargo:rustc-link-search=native={lib_dir}");
    }

    // Link the LiteRT-LM C library.
    // Use LITERT_LM_LINK_TYPE to choose static (default) or dylib.
    let link_type = env::var("LITERT_LM_LINK_TYPE").unwrap_or_else(|_| "static".to_string());
    println!("cargo:rustc-link-lib={link_type}=litert_lm");

    // Also link C++ standard library.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "linux" | "android" => println!("cargo:rustc-link-lib=dylib=stdc++"),
        "macos" | "ios" => println!("cargo:rustc-link-lib=dylib=c++"),
        _ => {}
    }

    // --- Step 2: Generate Rust FFI bindings from the C header ---
    let c_header = vendor_dir.join("c").join("engine.h");

    let bindings = bindgen::Builder::default()
        .header(c_header.to_str().unwrap())
        .allowlist_function("litert_lm_.*")
        .allowlist_type("LiteRtLm.*")
        .allowlist_type("InputData.*")
        .allowlist_type("Type")
        .allowlist_var("kType.*|kInput.*|kTopK|kTopP|kGreedy")
        .derive_debug(true)
        .derive_default(true)
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // Rebuild triggers
    println!("cargo:rerun-if-changed=vendor/LiteRT-LM/c/engine.h");
    println!("cargo:rerun-if-changed=vendor/LiteRT-LM/c/litert_lm_logging.h");
    println!("cargo:rerun-if-changed=vendor/LiteRT-LM/CMakeLists.txt");
    println!("cargo:rerun-if-env-changed=LITERT_LM_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LITERT_LM_LINK_TYPE");
}
