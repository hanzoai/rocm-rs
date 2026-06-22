# Upstream PR notes — native Windows / MSVC support for rocm-rs

This fork (`hanzoai/rocm-rs`, tagged **v0.5.2**) makes rocm-rs build and run on
**native Windows (MSVC)** with ROCm 7.13.0 (TheRock / `repo.amd.com` gfx1151
wheels), in addition to Linux. Everything here is a candidate for a PR back to
`RustNSparks/rocm-rs`. Captured now so the PR is easy to assemble later.

## Why
Stock 0.5.1 builds on Linux but fails on native Windows/MSVC with **200+
compile errors**. All of them are platform-ABI differences, not logic bugs.

## Root causes & fixes (4 commits on top of 0.5.1)

### 1. C-enum signedness — ~50× `error[E0308] expected u32, found i32`
C enums with only non-negative values are `unsigned int`/`u32` under GCC/Clang
(Linux) but **signed `int`/`i32` under MSVC**. bindgen follows the ABI, so every
`<lib>_status` / enum alias (`rocblas_status`, `rocrand_status`,
`rocrand_rng_type`, `rocrand_ordering`, `rocfft_status_e`,
`rocfft_transform_type`, …) is `u32` on Linux and `i32` on Windows. The crate
hardcodes literal `u32` at FFI return types, `Error::new`, and enum-arg params.

**Fix:** reference the bindgen-generated type instead of a literal `u32`
(`ffi::rocblas_status`, `bindings::rocrand_rng_type`, …). **No-op on Linux**
(the alias *is* `u32` there); correct on MSVC. Files: `rocblas/macros.rs`,
`rocrand/{error,generator,mod}.rs`, `rocfft/{error,plan,description}.rs`,
`rocarray/random.rs`. → clean, low-risk, clearly upstreamable.

### 2. Edition-2024 `unsafe_op_in_unsafe_fn` — 17× `error[E0133]`
The crate is `edition = "2024"`, where `unsafe_op_in_unsafe_fn` is deny-by-
default. This Linux toolchain emitted it as a *warning* while MSVC enforced it
as a hard *error* (toolchain lint-config difference). Thin `pub unsafe fn`
rocBLAS forwarders call their unsafe trait method with no `unsafe {}` block.

**Fix:** wrap the calls in `unsafe { }`. Identical on both platforms. Files:
`rocblas/level{1,2,3}.rs`. → clean, clearly upstreamable (and arguably a
correctness fix the Linux build should also enforce).

### 3. amdgcn GPU-sort kernel — `error[E0463]: can't find crate for core`
`hip/memory_ext/sorting.rs` embeds an amdgcn kernel via the external
`rocm_kernel_macros::amdgpu_kernel_finalize!` proc-macro, which shells out a
nested `cargo build` for `amdgcn-amd-amdhsa` (nightly + `-Zbuild-std`,
rust-src). That nested build **inherits the parent rustc environment** and
drops off the nightly toolchain on non-Linux hosts → core can't be built.

**Fix (fork-local, pragmatic):** gate the embedded kernel behind a **default-on
`gpu-sort` feature** with an empty-bytes fallback. The crate then builds
everywhere; GPU sort becomes a runtime no-op when the feature is off (LLM
inference never calls it). Files: `hip/memory_ext/sorting.rs`, `Cargo.toml`.

**Real upstream fix to discuss:** the bug is in `rocm_kernel_macros`'s nested
build invocation — it only does `.env_remove("RUSTUP_TOOLCHAIN")`; it should
also remove `RUSTC`, `CARGO`, `CARGO_BUILD_TARGET`, `RUSTFLAGS`,
`CARGO_ENCODED_RUSTFLAGS` so the nested `cargo` resolves the kernel's own
`rust-toolchain.toml` (nightly) + `.cargo/config.toml` (`build-std`) cleanly on
every host. The feature gate is the safe interim; the macro fix is the proper
cure. Suggest opening that as a separate issue/PR on `rocm_kernel_macros`.

## Verification
- **Linux:** `cargo check --no-default-features --features macros` *and* default
  features both reach `Finished` — proving every type change is a no-op there.
- **Windows (MSVC, ROCm 7.13.0, gfx1151 / Radeon 8060S):** builds clean; the
  downstream LLM engine builds and runs inference at **158.7 tok/s** decode
  (Qwen3-0.6B-Q8), ~36% faster than the same engine under WSL2 (no `/dev/dxg`
  per-dispatch overhead).

## PR packaging
1. Commits #1 (enum types) and #2 (unsafe blocks) → one PR, "native Windows /
   MSVC build support". Low risk, no Linux behavior change.
2. Commit #3 (gpu-sort gate) → either fold in with a note, or hold pending the
   `rocm_kernel_macros` env fix. Maintainer's call.
3. Do **not** include regenerated `bindings.rs` (per-toolchain build artifacts;
   bindgen rewrites them against the locally-installed ROCm headers).
