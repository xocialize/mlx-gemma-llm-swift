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
    /// Engine-stamped models folder (`ModelStorable`). On machines that DON'T already carry the
    /// weights, `load()` auto-materializes the declared `weightSources` into this store's
    /// `<root>/<org>/<name>` layout (caller holds security-scoped access). Ignored when
    /// `modelDirectory` is set. Excluded from `Codable`.
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

// MARK: - Weight sources (engine ≥0.19.0 auto-materialization / MAT gate)

extension GemmaLLMConfiguration: WeightSourcing {
    /// One source per configuration: the checkpoint repo. The quant axis is tiered by REPO
    /// (`gemma-<gen>-12b-it-<quant>` is a dedicated repo per quant), so `matching` stays nil
    /// (whole snapshot) — there are no sibling quant files to exclude, unlike the LTX2
    /// shared-components case. Unpublished quants (`weightsRepo == nil`) declare nothing;
    /// they are loadable only via an explicit `modelDirectory` (and `load()` diagnoses the
    /// dir-less case with `configurationMismatch`).
    public var weightSources: [WeightSource] {
        guard let repo = weightsRepo else { return [] }
        return [WeightSource(role: "checkpoint", repo: repo, revision: revision)]
    }

    /// Explicit `modelDirectory` first (the LTX shared-weights case — probe `config.json`),
    /// then the ModelStore layout (`<root>/<org>/<name>`). Nil store + no explicit directory
    /// ⇒ everything missing (the honest fresh-machine answer, MAT-4).
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        weightSources.filter { source in
            !Self.hasCheckpoint(at: modelDirectory)
                && !Self.hasCheckpoint(at: ModelStore(root: storeRoot).directory(for: source.repo))
        }
    }

    /// Where `load()` reads the checkpoint from: the explicit `modelDirectory` always wins;
    /// a nil directory resolves to the store layout (post-materialization home).
    public func resolvedModelDirectory(storeRoot: URL?) -> URL? {
        if let modelDirectory { return modelDirectory }
        guard let repo = weightsRepo else { return nil }
        return ModelStore(root: storeRoot).directory(for: repo)
    }

    /// A directory counts as carrying the checkpoint when `config.json` is present — the
    /// same cheap probe the LTX2 reference uses (safetensors land in the same snapshot).
    private static func hasCheckpoint(at directory: URL?) -> Bool {
        guard let directory else { return false }
        return FileManager.default.fileExists(
            atPath: directory.appending(path: "config.json").path)
    }
}

/// Cold-start weight prewarm (engine ≥0.7.0): page the weight files into the OS file cache
/// before `load()`'s first GPU evals, so a cold load off a slow/external volume never faults
/// weights inside a live Metal command buffer (the I5 `kIOGPUCommandBufferCallbackErrorTimeout`).
/// This replaces the hand-rolled prewarm the pre-governance path (LTX2's `GemmaTextGenerator`)
/// had to carry. Resolves against the store layout (engine ≥0.19.0), so a dir-less config
/// prewarms its DOWNLOADED weights from the second cold launch on — the models folder may sit
/// on a slow/external volume. First launch is a no-op (nothing on disk yet; the prewarmer
/// skips missing paths, best-effort).
extension GemmaLLMConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        guard let dir = resolvedModelDirectory(storeRoot: modelsRootDirectory) else { return [] }
        return [dir]
    }
}
