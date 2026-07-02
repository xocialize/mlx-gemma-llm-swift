import Foundation
import MLXToolKit

/// Init-time configuration for `GemmaLLMPackage` (C9). Carries the chosen checkpoint
/// (size × quant), an optional pinned revision, and WHERE the weights live; everything that
/// changes call-to-call (prompt, sampling, mode) rides the `LLMRequest`, never here.
public struct GemmaLLMConfiguration: PackageConfiguration, ModelStorable, QuantConfigured,
    FootprintConfigured
{
    /// Which Gemma-3 checkpoint to materialize and load.
    public var model: GemmaModel
    /// Pinned weights revision (commit/tag). `nil` resolves to the repo default.
    public var revision: String?
    /// **Direct local weights directory** (config.json + safetensors), bypassing the Hub cache.
    /// The LTX-app case: pass the SAME `gemmaDirectory` the LTX package uses as its text encoder
    /// — one copy of weights on disk, zero extra download, and the engine prewarms it (see
    /// `WeightPrewarming` below). Excluded from `Codable` (environment-specific).
    public var modelDirectory: URL?
    /// Engine-chosen models folder for Hub materialization on machines that DON'T already carry
    /// the weights (caller holds security-scoped access). Ignored when `modelDirectory` is set.
    /// Excluded from `Codable`.
    public var modelsRootDirectory: URL?

    public init(model: GemmaModel = .default,
                revision: String? = nil,
                modelDirectory: URL? = nil,
                modelsRootDirectory: URL? = nil) {
        self.model = model
        self.revision = revision
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    /// The HF `mlx-community` repo id for the chosen checkpoint, if published.
    public var weightsRepo: String? { model.weightsRepo }

    // MARK: QuantConfigured — the registered variant's quant, so the governor charges the
    // selected checkpoint's QuantFootprint instead of the manifest default's.
    public var quant: Quant { model.quant }

    // MARK: FootprintConfigured — precise per-config split (size × quant), overriding the
    // static manifest figures (the BiRefNet/Qwen per-config-hint pattern).
    public var residentBytesHint: UInt64? { model.residentBytes }
    public var peakActivationBytesHint: UInt64? { model.peakActivationBytes }

    private enum CodingKeys: String, CodingKey {
        case model, revision
    }
}

/// Cold-start weight prewarm (engine ≥0.7.0): page the weight files into the OS file cache
/// before `load()`'s first GPU evals, so a cold load off a slow/external volume never faults
/// weights inside a live Metal command buffer (the I5 `kIOGPUCommandBufferCallbackErrorTimeout`).
/// This replaces the hand-rolled prewarm the pre-governance path (LTX2's `GemmaTextGenerator`)
/// had to carry. Only the direct-directory case prewarms — Hub materialization downloads to a
/// local (fast) cache where the fault cost is negligible.
extension GemmaLLMConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        guard let dir = modelDirectory else { return [] }
        return [dir]
    }
}
