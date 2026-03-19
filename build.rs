use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor_dir = manifest_dir.join("vendor").join("LiteRT-LM");

    // --- Step 1: Build and link the C++ library ---
    #[cfg(feature = "build-from-source")]
    {
        let mut cmake_cfg = cmake::Config::new(&vendor_dir);
        cmake_cfg.define("CMAKE_BUILD_TYPE", "Release");

        if cfg!(feature = "async-constraint-masking") {
            cmake_cfg.cflag("-DLITERT_LM_ASYNC_CONSTRAINT_MASKING");
            cmake_cfg.cxxflag("-DLITERT_LM_ASYNC_CONSTRAINT_MASKING");
        }

        let dst = cmake_cfg.build();
        println!("cargo:rustc-link-search=native={}/lib", dst.display());
    }

    if cfg!(feature = "async-constraint-masking") {
        println!("cargo:rustc-cfg=async_constraint_masking");
    }

    // Allow overriding the library search path via environment variable.
    // Usage: LITERT_LM_LIB_DIR=/path/to/lib cargo build
    if let Ok(lib_dir) = env::var("LITERT_LM_LIB_DIR") {
        println!("cargo:rustc-link-search=native={lib_dir}");
    }

    // Link the LiteRT-LM C library.
    // Default to dylib (shared library built by Bazel via //c:liblitert_lm_c.so).
    // Override with LITERT_LM_LINK_TYPE=static for static linking.
    let link_type = env::var("LITERT_LM_LINK_TYPE").unwrap_or_else(|_| "dylib".to_string());
    println!("cargo:rustc-link-lib={link_type}=litert_lm_c");

    // When linking statically, we also need the C++ standard library and
    // platform-specific dependencies.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if link_type == "static" {
        match target_os.as_str() {
            "linux" => println!("cargo:rustc-link-lib=dylib=stdc++"),
            "android" => {
                println!("cargo:rustc-link-lib=static=c++_static");
                println!("cargo:rustc-link-lib=static=c++abi");
            }
            "macos" | "ios" => println!("cargo:rustc-link-lib=dylib=c++"),
            _ => {}
        }
    }

    // Android needs the log library and RPATH setup for shared libs.
    if target_os == "android" {
        println!("cargo:rustc-link-lib=dylib=log");
    }

    // --- Step 2: Generate Rust FFI bindings from the C header ---
    let c_header = vendor_dir.join("c").join("engine.h");

    let target = env::var("TARGET").unwrap_or_default();
    let mut builder = bindgen::Builder::default()
        .header(c_header.to_str().unwrap())
        .allowlist_function("litert_lm_.*")
        .allowlist_type("LiteRtLm.*")
        .allowlist_type("InputData.*")
        .allowlist_type("Type")
        .allowlist_var("kType.*|kInput.*|kTopK|kTopP|kGreedy")
        .derive_debug(true)
        .derive_default(true);

    // When cross-compiling for Android, point clang at the NDK sysroot.
    if target.contains("android") {
        let ndk_home = env::var("ANDROID_NDK_HOME")
            .or_else(|_| env::var("ANDROID_NDK"))
            .or_else(|_| env::var("NDK_HOME"))
            .expect(
                "Cross-compiling for Android requires ANDROID_NDK_HOME, \
                 ANDROID_NDK, or NDK_HOME to be set",
            );
        let sysroot = format!("{ndk_home}/toolchains/llvm/prebuilt/linux-x86_64/sysroot");
        builder = builder
            .clang_arg(format!("--sysroot={sysroot}"))
            .clang_arg(format!("--target={target}"));
    }

    let bindings = builder.generate().expect("Unable to generate bindings");

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
