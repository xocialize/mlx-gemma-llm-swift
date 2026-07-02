import Foundation
import MLXToolKit

/// A Gemma-3 instruction-tuned model size. Repo ids follow the mlx-community convention
/// `gemma-3-<size>-it-<quant>`.
///
/// v0.1 catalogs **12B only** — the size the LTX-2.3 stack already ships on disk (its text
/// encoder is the same checkpoint, so the governed `llm` surface adds zero download for that
/// host). Additional sizes (1B/4B/27B are published on mlx-community) get cases when a consumer
/// wants them; the geometry table below is per-size so growth is additive.
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
    // Gemma 3 is a HYBRID sliding/global attention transformer: every `slidingWindowPattern`-th
    // layer is GLOBAL attention with a context-growing KV cache; the rest are SLIDING-window
    // layers whose cache is capped at `slidingWindow` tokens and stops growing past it. So the
    // context-scaling transient is driven by the global layers alone; the sliding layers
    // contribute a fixed ceiling. Values verified against
    // mlx-community/gemma-3-12b-it-4bit `text_config` (2026-07-02); head_dim (256), vocab
    // (262144) and sliding_window_pattern (6) are the mlx-swift-lm Gemma3Text loader defaults
    // for keys the published config omits.

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

    /// Per-head dimension (`head_dim`; loader default 256 across the Gemma-3 family).
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
    var slidingWindowPattern: Int { 6 }

    /// Count of GLOBAL (context-growing) attention layers: 48/6 = 8 on 12B.
    var numGlobalLayers: Int { numLayers / slidingWindowPattern }

    /// Count of sliding-window layers (cache capped at `slidingWindow`).
    var numSlidingLayers: Int { numLayers - numGlobalLayers }
}

/// A concrete Gemma-3 checkpoint = size × quantization. Confirmed-published combinations on
/// mlx-community: 12B-it in {4bit, 8bit, bf16}.
public struct GemmaModel: Sendable, Codable, Equatable, Hashable {
    public var size: GemmaSize
    public var quant: Quant

    public init(size: GemmaSize, quant: Quant) {
        self.size = size
        self.quant = quant
    }

    /// First-bring-up default: **12B-it 4-bit** — the exact checkpoint the LTX-2.3 stack already
    /// materializes as its text encoder (7.5 GB on disk), so the governed `llm` surface is free
    /// for that host.
    public static let `default` = GemmaModel(size: .b12, quant: .int4)

    public var displayName: String { "Gemma 3 · \(size.rawValue)-it (\(quant.rawValue))" }

    /// HF `mlx-community` repo id, e.g. `mlx-community/gemma-3-12b-it-4bit`. `nil` when the
    /// quant has no published mlx-community suffix.
    public var weightsRepo: String? {
        guard let suffix = quant.mlxCommunitySuffix else { return nil }
        return "mlx-community/gemma-3-\(size.rawValue)-it-\(suffix)"
    }

    /// Approximate on-disk size of the materialized weights, in bytes.
    ///
    /// NB the checkpoint is the MULTIMODAL `Gemma3ForConditionalGeneration` layout — it carries a
    /// SigLIP vision tower (~0.4B params) on disk that `Gemma3TextModel` never loads, so true
    /// residency lands slightly UNDER this (the P2 floor measurement is authoritative).
    /// 4-bit measured on disk: 7.5 GB (2026-07-02).
    public var onDiskBytes: UInt64 {
        switch quant {
        case .int4: return 7_500_000_000  // measured (incl. quant scales + vision tower)
        default: return UInt64(paramsIncludingVisionBillions * 1_000_000_000 * quant.bytesPerWeightEstimate)
        }
    }

    private var paramsIncludingVisionBillions: Double { size.paramsBillions + 0.4 }

    /// The persistent weights floor — what stays resident while loaded. On-disk bytes of the
    /// selected checkpoint (mmap'd, paged on demand); the KV/prefill transient is split out into
    /// `peakActivationBytes` so the engine reserves ONE shared activation across co-residents.
    public var residentBytes: UInt64 { onDiskBytes }

    /// The documented context envelope (prompt + generated tokens) the declared footprint is
    /// sized for. Prompt-enhancement calls (the first consumer) run ~1–2k total tokens; 2048
    /// matches the Qwen package's envelope rationale (a realistic working window, far below the
    /// 128k `max_position_embeddings`). Requests past it still run; they just exceed the declared
    /// transient (the reactive phys-footprint governor still catches a true OOM).
    public static let contextEnvelopeTokens = 2048

    /// The persisted KV-cache size at a given total context. Gemma-3 hybrid attention:
    /// global layers grow with context; sliding layers cap at `slidingWindow`.
    /// `2 (K+V) × [globalLayers × T + slidingLayers × min(T, window)] × kvHeads × headDim × 2 B (bf16)`.
    public func kvCacheBytes(maxTokens: Int) -> UInt64 {
        let cacheDtypeBytes = 2  // bf16 K/V cache regardless of weight quant
        let kPlusV = 2
        let t = max(0, maxTokens)
        let globalElems = size.numGlobalLayers * t
        let slidingElems = size.numSlidingLayers * min(t, size.slidingWindow)
        let perLayerHead = kPlusV * size.numKVHeads * size.headDim
        return UInt64((globalElems + slidingElems) * perLayerHead * cacheDtypeBytes)
    }

    /// The transient activation peak at the documented envelope.
    ///
    /// Analytic KV at 2048 tokens ≈ 0.47 GB (134 MB global + 336 MB sliding-capped). The Qwen
    /// package's measurement lesson says prefill COMPUTE SCRATCH dominates the persisted cache —
    /// Gemma-3 is dense softmax (no GDN chunked-scan), so the Qwen per-token figure doesn't
    /// transfer; this initial declaration is KV + a width-scaled scratch allowance.
    /// **ESTIMATE — P2 re-measures this empirically (mem-bench at the 2048 envelope) and the
    /// measured value replaces it before promotion.**
    public var peakActivationBytes: UInt64 {
        kvCacheBytes(maxTokens: Self.contextEnvelopeTokens) + 2_500_000_000
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
        GemmaModel(size: .b12, quant: .int4),
        GemmaModel(size: .b12, quant: .int8),
        GemmaModel(size: .b12, quant: .bf16),
    ]
}

extension Quant {
    /// The mlx-community repo suffix for this quant when one is published under the
    /// `gemma-3-<size>-it-<suffix>` scheme.
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
