// swift-tools-version: 6.2
import PackageDescription

// mlx-gemma-llm-swift — a Gemma-3 `llm` surface backed by the MLX-Swift LM runtime, conforming
// to the MLXEngine contract (MLXToolKit). Dedicated per-family package (the Qwen convention:
// qwen-llm / qwen3-tts / qwen25vl are separate packages over one family).
//
// **DESIGN CONSTRAINT — MLXLLM only, NEVER MLXVLM.** mlx-swift-lm's process-global
// `ModelFactoryRegistry` probes the VLM factory first, and `gemma3` is registered in BOTH
// factories — linking MLXVLM anywhere in a host process shadows `gemma3` to the multimodal
// `Gemma3` and breaks text-only consumers (the LTX BRIDGE-LTX-003 defect). This package exists
// so hosts get governed Gemma text generation WITHOUT that hazard: it links the text factory
// alone, and `load()` verifies the resolved model type so a host that re-introduces MLXVLM gets
// a diagnosable error instead of a silently fatter (vision-tower-resident) model.
//
// First consumer: the LTX-2.3 app's prompt enhancer (PromptEnhanceKit) — previously an
// out-of-engine `GemmaTextGenerator` load; this package moves it under MLXEngine governance
// (admission, footprint charge, prewarm, license gate, store integration).
//
// NB: depends on the LOCAL engine (path dep) until the engine release that carries
// `SPDXLicense.gemmaTerms` is tagged; switch to the URL dep at promotion.
let package = Package(
    name: "mlx-gemma-llm-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXGemmaLLM", targets: ["MLXGemmaLLM"]),
    ],
    dependencies: [
        // Local engine until the .gemmaTerms release is tagged (then: url + from:).
        .package(path: "../../../MLXEngine/mlx-engine-swift"),
        // MLX-Swift LM runtime — 3.31.4 is the fleet-validated floor (LTX gates green on it).
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.4")),
        // mlx-swift, for MLX.Memory.clearCache() in unload(). Matches mlx-swift-lm's resolution.
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.5"),
        // mlx-swift-lm 3.x decoupled the HF stack — download macro/tokenizer need these.
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.2.1"),
    ],
    targets: [
        .target(
            name: "MLXGemmaLLM",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                // Text factory ONLY — see the design constraint above. Do NOT add MLXVLM.
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                // HF download + tokenizer for the #huggingFaceLoadModelContainer macro.
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "RunGemmaLLM",
            dependencies: [
                "MLXGemmaLLM",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MLXGemmaLLMTests",
            dependencies: [
                "MLXGemmaLLM",
                // Test-only: run the catalog through the engine's admissibility check.
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
