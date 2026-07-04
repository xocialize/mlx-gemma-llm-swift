import Foundation
import MLXToolKit

/// The Gemma model generation. Repo prefix, the MLXLLM text-model type the factory must
/// resolve, and the WEIGHT license all differ per family; the 12B decoder geometry does not
/// (verified against both published configs, 2026-07-04).
///
/// - `gemma3` — `model_type = gemma3` (multimodal checkpoint; SigLIP vision tower stays on
///   disk under the text factory). Weights: Gemma Terms of Use.
/// - `gemma4` — the 12B declares `model_type = gemma4_unified` (encoder-free multimodal);
///   mlx-swift-lm ≥ 3.31.4 registers it in the TEXT factory (`Gemma4Model`), whose `sanitize`
///   strips the `vision_embedder`/audio weights. Weights: Apache-2.0 (upstream
///   `google/gemma-4-*`; the mlx-community cards omit the tag but derive from Apache-2.0).
public enum GemmaFamily: String, Sendable, Codable, CaseIterable {
    case gemma3
    case gemma4

    /// The mlx-community repo prefix: `gemma-3-…` / `gemma-4-…`.
    var repoPrefix: String {
        switch self {
        case .gemma3: return "gemma-3"
        case .gemma4: return "gemma-4"
        }
    }

    /// The MLXLLM model type the shared architecture key must resolve to — the
    /// MLXVLM-shadowing guard (`WrongModelTypeError`). Both `gemma3` and `gemma4_unified`
    /// are registered in BOTH factories, so the hazard is identical across families.
    var expectedTextModelType: String {
        switch self {
        case .gemma3: return "Gemma3TextModel"
        case .gemma4: return "Gemma4Model"
        }
    }

    /// Per-family WEIGHT license. The static manifest declares the STRICTER of the two
    /// (`.gemmaTerms`) so the registration-time gate is conservative-correct for every
    /// variant; this property is the accurate per-variant record (both are on the
    /// permissive allowlist, so gating is unaffected).
    public var weightLicense: SPDXLicense {
        switch self {
        case .gemma3: return .gemmaTerms
        case .gemma4: return .apache2
        }
    }
}

/// A Gemma instruction-tuned model size. Repo ids follow the mlx-community convention
/// `gemma-<gen>-<size>-it-<quant>`.
///
/// v0.1 cataloged Gemma-3 **12B only** — the size the LTX-2.3 stack already ships on disk (its
/// text encoder is the same checkpoint, so the governed `llm` surface adds zero download for
/// that host). v0.2 adds the Gemma-4 12B on the same size case: the decoder geometry below is
/// IDENTICAL across the two generations (48 layers, 8 global + 40 sliding, 8 KV heads,
/// head_dim 256, hidden 3840, window 1024 — Gemma 4 expresses the hybrid split via
/// `layer_types` instead of `sliding_window_pattern`, same 8/40 counts). Additional sizes
/// (1B/4B/27B gemma-3; E2B/E4B gemma-4, which have DIFFERENT geometry) get cases when a
/// consumer wants them; the geometry table below is per-size so growth is additive.
public enum GemmaSize: String, Sendable, Codable, CaseIterable {
    case b12 = "12b"

    /// Approximate parameter count in billions — used for footprint estimates.
    public var paramsBillions: Double {
        switch self {
        case .b12: return 12.2
        }
    }

    // MARK: KV-cache geometry (from the published config.json + Gemma3Text loader defaults)
    //
    // Gemma is a HYBRID sliding/global attention transformer: some layers are GLOBAL
    // attention with a context-growing KV cache; the rest are SLIDING-window layers whose
    // cache is capped at `slidingWindow` tokens and stops growing past it. So the
    // context-scaling transient is driven by the global layers alone; the sliding layers
    // contribute a fixed ceiling. Values verified against
    // mlx-community/gemma-3-12b-it-4bit `text_config` (2026-07-02) AND
    // mlx-community/gemma-4-12b-it-4bit `text_config` (2026-07-04 — identical; the Gemma 4
    // config lists `layer_types` with 8 full_attention + 40 sliding_attention). head_dim
    // (256) and vocab (262144) are explicit in the gemma-4 config and the mlx-swift-lm
    // loader defaults for gemma-3.

    /// Total decoder layers (`num_hidden_layers`).
    var numLayers: Int {
        switch self {
        case .b12: return 48
        }
    }

    /// KV heads (`num_key_value_heads`, GQA).
    var numKVHeads: Int {
        switch self {
        case .b12: return 8
        }
    }

    /// Per-head dimension (`head_dim`; 256 across both 12B generations).
    var headDim: Int { 256 }

    /// Model width (`hidden_size`) — the scale factor for prefill-scratch estimates.
    var hiddenSize: Int {
        switch self {
        case .b12: return 3840
        }
    }

    /// Sliding-window span for the local-attention layers (`sliding_window`).
    var slidingWindow: Int { 1024 }

    /// 1-in-`slidingWindowPattern` layers is global attention; the rest are sliding.
    /// (Gemma 4 lists `layer_types` explicitly instead — same 8-global/40-sliding split.)
    var slidingWindowPattern: Int { 6 }

    /// Count of GLOBAL (context-growing) attention layers: 48/6 = 8 on 12B.
    var numGlobalLayers: Int { numLayers / slidingWindowPattern }

    /// Count of sliding-window layers (cache capped at `slidingWindow`).
    var numSlidingLayers: Int { numLayers - numGlobalLayers }
}

/// A concrete Gemma checkpoint = family × size × quantization. Confirmed-published
/// combinations on mlx-community: gemma-3 12B-it in {4bit, 8bit, bf16}; gemma-4 12B-it in
/// {4bit, 8bit, bf16} (5bit/6bit/qat-4bit also exist upstream but are out-of-grammar for
/// `Quant` → uncataloged).
public struct GemmaModel: Sendable, Codable, Equatable, Hashable {
    public var family: GemmaFamily
    public var size: GemmaSize
    public var quant: Quant

    public init(family: GemmaFamily = .gemma3, size: GemmaSize, quant: Quant) {
        self.family = family
        self.size = size
        self.quant = quant
    }

    /// First-bring-up default: **Gemma-3 12B-it 4-bit** — the exact checkpoint the LTX-2.3
    /// stack already materializes as its text encoder (7.5 GB on disk), so the governed `llm`
    /// surface is free for that host. (Gemma-4 does NOT displace the default: LTX hosts keep
    /// the zero-extra-download property.)
    public static let `default` = GemmaModel(size: .b12, quant: .int4)

    public var displayName: String {
        let gen = family == .gemma3 ? "Gemma 3" : "Gemma 4"
        return "\(gen) · \(size.rawValue)-it (\(quant.rawValue))"
    }

    /// HF `mlx-community` repo id, e.g. `mlx-community/gemma-3-12b-it-4bit` /
    /// `mlx-community/gemma-4-12b-it-4bit`. `nil` when the quant has no published
    /// mlx-community suffix.
    public var weightsRepo: String? {
        guard let suffix = quant.mlxCommunitySuffix else { return nil }
        return "mlx-community/\(family.repoPrefix)-\(size.rawValue)-it-\(suffix)"
    }

    /// Approximate on-disk size of the materialized weights, in bytes.
    ///
    /// NB both checkpoints carry never-loaded multimodal weights on disk — gemma-3 a SigLIP
    /// vision tower (~0.4B params), gemma-4 the `gemma4_unified` `vision_embedder` — that the
    /// text factory strips at load, so true residency lands UNDER this plus overhead (the
    /// mem-bench floor measurement is authoritative).
    /// Measured on disk (safetensors sum from the HF headers):
    ///   gemma-3 4bit: 7.5 GB (2026-07-02) · gemma-4 4bit: 6.74 / 8bit: 12.72 / bf16: 23.92 GB
    ///   (2026-07-04).
    public var onDiskBytes: UInt64 {
        switch (family, quant) {
        case (.gemma3, .int4): return 7_500_000_000  // measured (incl. quant scales + vision tower)
        case (.gemma4, .int4): return 6_741_039_511  // measured (HF safetensors sum)
        case (.gemma4, .int8): return 12_716_202_713 // measured (HF safetensors sum)
        case (.gemma4, .bf16): return 23_919_548_177 // measured (HF safetensors sum)
        default:
            return UInt64(paramsIncludingVisionBillions * 1_000_000_000 * quant.bytesPerWeightEstimate)
        }
    }

    private var paramsIncludingVisionBillions: Double { size.paramsBillions + 0.4 }

    /// The persistent floor — what stays resident while loaded. MEASURED via
    /// `RunGemmaLLM --mem-bench` (post-warmup + clearCache phys_footprint), NOT the on-disk
    /// bytes: the floor lands above disk (text weights materialized by the first forward +
    /// tokenizer/runtime overhead ≈ +1.8 GB on gemma-3 4-bit) even though the multimodal
    /// weights never load.
    ///   gemma-3 int4 measured 9.32 GB (2026-07-02) → declared 10 GB.
    ///   gemma-4 int4 measured 7.11 GB (2026-07-04, RunGemmaLLM --mem-bench --gemma4) →
    ///   declared 8 GB. The lower floor vs gemma-3 tracks the smaller checkpoint (6.74 vs
    ///   7.5 GB on disk; no SigLIP tower to skip).
    /// Other quants are the measured overhead added to their weight estimate — re-measure
    /// when a consumer validates them.
    public var residentBytes: UInt64 {
        switch (family, quant) {
        case (.gemma3, .int4): return 10_000_000_000  // measured floor 9.32 GB + headroom
        case (.gemma4, .int4): return 8_000_000_000   // measured floor 7.11 GB + headroom
        case (_, .int8): return 16_000_000_000        // est: ~13 GB weights + measured ~2.4 GB overhead
        default: return 27_000_000_000                // bf16 est: ~24.4 GB weights + overhead
        }
    }

    /// The documented context envelope (prompt + generated tokens) the declared footprint is
    /// sized for. Prompt-enhancement calls (the first consumer) run ~1–2k total tokens; 2048
    /// matches the Qwen package's envelope rationale (a realistic working window, far below the
    /// 128k/256k `max_position_embeddings`). Requests past it still run; they just exceed the
    /// declared transient (the reactive phys-footprint governor still catches a true OOM).
    public static let contextEnvelopeTokens = 2048

    /// The persisted KV-cache size at a given total context. Gemma hybrid attention:
    /// global layers grow with context; sliding layers cap at `slidingWindow`.
    /// `2 (K+V) × [globalLayers × T + slidingLayers × min(T, window)] × kvHeads × headDim × 2 B (bf16)`.
    /// Same geometry for both 12B generations (see `GemmaSize`).
    public func kvCacheBytes(maxTokens: Int) -> UInt64 {
        let cacheDtypeBytes = 2  // bf16 K/V cache regardless of weight quant
        let kPlusV = 2
        let t = max(0, maxTokens)
        let globalElems = size.numGlobalLayers * t
        let slidingElems = size.numSlidingLayers * min(t, size.slidingWindow)
        let perLayerHead = kPlusV * size.numKVHeads * size.headDim
        return UInt64((globalElems + slidingElems) * perLayerHead * cacheDtypeBytes)
    }

    /// The transient activation peak at the documented envelope. MEASURED via
    /// `RunGemmaLLM --mem-bench` at ~2k total tokens (≈1.2k prefill + 800 generated):
    ///   gemma-3 int4 **0.37 GB** over the floor (2026-07-02) — tracking the analytic hybrid
    ///   KV closely (dense softmax Gemma has no Qwen GatedDeltaNet prefill-scratch blowup);
    ///   declared as analytic KV + 0.5 GB headroom (≈ 0.97 GB at the envelope).
    ///   gemma-4 int4 **2.16 GB** over the floor (2026-07-04) — SAME KV geometry but well
    ///   above the analytic KV (+0.47 GB): `gemma4_unified` carries extra prefill-transient
    ///   scratch the analytic model doesn't capture, so the family gets its own MEASURED
    ///   declaration (+0.5 GB headroom) instead of the analytic one.
    public var peakActivationBytes: UInt64 {
        switch family {
        case .gemma3:
            return kvCacheBytes(maxTokens: Self.contextEnvelopeTokens) + 500_000_000
        case .gemma4:
            return 2_700_000_000  // measured 2.16 GB + headroom
        }
    }

    /// Cost-to-run footprint for the Model Manager (C10): weights floor + split-out transient.
    public var footprint: QuantFootprint {
        QuantFootprint(quant: quant,
                       residentBytes: residentBytes,
                       peakActivationBytes: peakActivationBytes)
    }

    /// The C10 requirements for this exact checkpoint.
    public var requirements: RequirementsManifest {
        RequirementsManifest(
            footprints: [footprint],
            requiredBackends: [.metalGPU],
            os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
            chipFloor: nil
        )
    }

    /// Every checkpoint published on mlx-community (the catalog / admissibility sanity markers).
    public static let allPublished: [GemmaModel] = [
        GemmaModel(family: .gemma3, size: .b12, quant: .int4),
        GemmaModel(family: .gemma3, size: .b12, quant: .int8),
        GemmaModel(family: .gemma3, size: .b12, quant: .bf16),
        GemmaModel(family: .gemma4, size: .b12, quant: .int4),
        GemmaModel(family: .gemma4, size: .b12, quant: .int8),
        GemmaModel(family: .gemma4, size: .b12, quant: .bf16),
    ]

    // MARK: Codable back-compat — v0.1 payloads have no `family` key; default to gemma3.

    private enum CodingKeys: String, CodingKey {
        case family, size, quant
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.family = try c.decodeIfPresent(GemmaFamily.self, forKey: .family) ?? .gemma3
        self.size = try c.decode(GemmaSize.self, forKey: .size)
        self.quant = try c.decode(Quant.self, forKey: .quant)
    }
}

extension Quant {
    /// The mlx-community repo suffix for this quant when one is published under the
    /// `gemma-<gen>-<size>-it-<suffix>` scheme.
    var mlxCommunitySuffix: String? {
        switch self {
        case .int4: return "4bit"
        case .int8: return "8bit"
        case .bf16: return "bf16"
        default: return nil
        }
    }

    /// Approximate bytes-per-weight for on-disk estimation (unmeasured quants only).
    var bytesPerWeightEstimate: Double {
        switch self {
        case .int4: return 0.56  // 4-bit + group-64 scales/biases overhead
        case .int8: return 1.06
        case .bf16: return 2.0
        default: return 2.0
        }
    }
}
