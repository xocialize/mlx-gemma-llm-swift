import Foundation
import MLX
import MLXToolKit
import MLXConstrainedDecoding
import MLXLMCommon
import MLXLLM
import MLXHuggingFace
import HuggingFace
import Tokenizers

/// A Gemma instruction-tuned model (family axis: Gemma-3 / Gemma-4) exposing the canonical
/// `llm` surface — the governed counterpart of running Gemma out-of-engine.
///
/// One `ModelPackage`, one surface. The engine owns the lifecycle (inversion of control): it
/// constructs this from a `GemmaLLMConfiguration`, prewarms + pages weights in with `load()`,
/// drives `run(_:)`, and reclaims with `unload()`. Lifecycle methods are isolated to
/// `InferenceActor` (C13).
///
/// **Text factory only.** This package links `MLXLLM` and never `MLXVLM`, so the shared
/// architecture keys resolve to the TEXT models — `gemma3` → `Gemma3TextModel` (the SigLIP
/// vision tower stays on disk, unloaded) and `gemma4_unified` → `Gemma4Model` (mlx-swift-lm
/// ≥ 3.31.4; its `sanitize` strips the `vision_embedder`/audio weights). Both keys are
/// registered in BOTH factories upstream, so the shadowing hazard is identical across
/// families: `load()` verifies the resolution and throws a diagnosable error if a host
/// process re-introduced MLXVLM (see `WrongModelTypeError`).
@InferenceActor
public final class GemmaLLMPackage: ModelPackage {
    public typealias Configuration = GemmaLLMConfiguration

    /// Static, registrable blueprint — read at registration/eligibility time (nonisolated).
    /// Describes the default variant (12B-it 4-bit); per-variant footprints ride the
    /// configuration's `FootprintConfigured` hints.
    public nonisolated static var manifest: PackageManifest {
        let target = GemmaModel.default
        return PackageManifest(
            // Weights: the manifest is static (one declaration for the package), so it
            // carries the STRICTER of the two family licenses — Gemma Terms of Use
            // (LicenseRef-Gemma-Terms), reviewed 2026-07-02, permissive-with-AUP, on the
            // engine's permissiveAllowlist. Obligations when SHIPPING gemma-3 weights:
            // terms passthrough + the §3.1 notice. Gemma-4 weights are Apache-2.0
            // (strictly more permissive — see `GemmaFamily.weightLicense`), so the gate
            // stays conservative-correct for every variant. Port code: Apache-2.0.
            license: LicenseDeclaration(weightLicense: .gemmaTerms, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: target.weightsRepo ?? "mlx-community/gemma-3-12b-it-4bit",
                revision: "main",
                tier: 1
            ),
            requirements: target.requirements,
            specialties: [
                SpecialtyWeight(.general, strength: 0.7)
            ],
            surfaces: [
                // The surface name stays "gemma-3-llm" for PackageID stability (existing
                // registrations key on it). A Gemma-4 configuration is registered from the
                // same package type — pass an explicit `id:` (e.g. "gemma-4-llm") to
                // co-register both families side by side.
                LLMContract.descriptor(
                    name: "gemma-3-llm",
                    summary: "Gemma instruction-tuned text generation (MLX, text factory "
                        + "only; family axis gemma-3/gemma-4 via the configuration). The "
                        + "gemma-3 default is the same checkpoint the LTX-2.3 stack uses as "
                        + "its text encoder, so hosts carrying LTX get a governed "
                        + "prompt-enhancement LLM at zero extra download.",
                    // Honest C11 advertisement: grammar-masked decode via
                    // MLXConstrainedDecoding (contract 1.16.0).
                    supportsStructuredOutput: true
                )
            ]
        )
    }

    /// Thrown when the shared architecture key resolves to something other than the family's
    /// TEXT model (`Gemma3TextModel` / `Gemma4Model`) — in practice the HOST process linked
    /// MLXVLM (directly or via another package), whose factory is probed first in
    /// mlx-swift-lm's process-global registry and shadows the text architecture with the
    /// multimodal variant (`Gemma3` / `Gemma4Unified`). Generation would still work, but the
    /// vision weights load too (footprint drift) and text-only consumers in the same process
    /// break (the LTX BRIDGE-LTX-003 defect) — fail loud instead.
    public struct WrongModelTypeError: Error, CustomStringConvertible {
        public let expected: String
        public let actual: String
        public var description: String {
            "GemmaLLMPackage: expected \(expected), got \(actual). A dependency in the host "
                + "app links MLXVLM, which shadows the architecture key in mlx-swift-lm's "
                + "process-global factory registry. Remove the MLXVLM-linking dependency."
        }
    }

    private let configuration: Configuration
    /// The resident model + tokenizer, paged in by `load()`. `nil` until loaded.
    private var container: ModelContainer?

    /// Vocab classification for constrained decoding (contract 1.16.0 `responseFormat`) —
    /// built once per residency on the first structured request (a full-vocab pass over the
    /// ~262k-entry SentencePiece tokenizer; not paid by freeform-only consumers), dropped
    /// with the weights on `unload()`.
    private var constrainedVocabulary: TokenVocabulary?

    /// Masking telemetry of the most recent structured run — surfaced for the
    /// `RunGemmaLLM --structured` gate's latency report.
    public private(set) var lastStructuredStats: JSONConstraintEngine.Stats?

    /// Cheap construction — no compute, no weight paging (C13). Residency is `load()`'s job.
    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    /// Page the working set in. Idempotent when already resident.
    ///
    /// Dir-less configurations auto-materialize (engine ≥0.19.0): missing declared
    /// `weightSources` download into the store's `<root>/<org>/<name>` layout with per-file
    /// progress via `WeightDownloadProgress` (the engine binds the sink around this call and
    /// surfaces `.downloading`), then loading proceeds from the store-resolved directory.
    /// Explicit-directory configs never touch the network (the DEV_ARCHIVE / LTX-app case).
    public func load() async throws {
        guard container == nil else { return }
        let loaded: ModelContainer
        if let dir = configuration.modelDirectory {
            // Direct local directory (the LTX-app case; engine prewarm already paged it).
            let modelConfig = ModelConfiguration(directory: dir)
            loaded = try await #huggingFaceLoadModelContainer(configuration: modelConfig)
        } else {
            guard let repo = configuration.weightsRepo else {
                throw PackageError.configurationMismatch(
                    expected: "a published mlx-community quant or an explicit modelDirectory",
                    got: configuration.model.displayName)
            }
            if let root = configuration.modelsRootDirectory,
               let dir = ModelStore(root: root).directory(for: repo)
            {
                // Auto-materialize into the engine-chosen models folder (caller holds
                // security-scoped access), then load from the store layout.
                let missing = configuration.missingWeightSources(storeRoot: root)
                if !missing.isEmpty {
                    try await WeightMaterializer.materialize(missing, into: root)
                }
                loaded = try await #huggingFaceLoadModelContainer(
                    configuration: ModelConfiguration(directory: dir))
            } else {
                // No store stamped: fall back to the downloader's default Hub cache.
                let modelConfig = ModelConfiguration(
                    id: repo, revision: configuration.revision ?? "main")
                loaded = try await #huggingFaceLoadModelContainer(configuration: modelConfig)
            }
        }
        // Verify the text factory won the registry (see WrongModelTypeError).
        let expected = configuration.model.family.expectedTextModelType
        let typeName = await loaded.perform { String(describing: type(of: $0.model)) }
        guard typeName.hasSuffix(expected) else {
            throw WrongModelTypeError(expected: expected, actual: typeName)
        }
        container = loaded
    }

    /// Release the working set; the instance survives for a later `load()`.
    public func unload() async {
        constrainedVocabulary = nil
        container = nil
        MLX.Memory.clearCache()  // release the retained MLX pool so eviction frees RSS
    }

    /// Run one `llm` call: decode the canonical request, generate on the resident model,
    /// return canonical text. Honors cancellation so the MemoryGovernor can preempt + requeue.
    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let container else { throw PackageError.notLoaded }
        guard request.capability == .llm, let llm = request as? LLMRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        // Map canonical sampling controls onto MLX GenerateParameters.
        var parameters = GenerateParameters()
        if let temperature = llm.parameters.temperature { parameters.temperature = Float(temperature) }
        if let topP = llm.parameters.topP { parameters.topP = Float(topP) }
        parameters.maxTokens = llm.parameters.maxTokens

        // Structured output (contract 1.16.0): bypass ChatSession — mlx-swift-lm 3.31.x has
        // no processor-injection seam through GenerateParameters/ChatSession — and drive
        // TokenIterator directly with the grammar-masking LogitProcessor (same wiring as
        // mlx-qwen-llm-swift; the kit is shared).
        if llm.responseFormat != nil {
            return try await runStructured(container: container, request: llm,
                                           parameters: parameters)
        }

        // System turns become session instructions.
        let instructions = llm.messages
            .filter { $0.role == .system }
            .map(\.content)
            .joined(separator: "\n\n")

        // Multi-turn: prior non-system turns seed history; respond to the final user turn.
        let conversational = llm.messages.filter { $0.role != .system }
        let prompt = conversational.last?.content ?? ""
        let priorTurns = conversational.isEmpty ? [] : Array(conversational.dropLast())

        // No mode → template-kwarg mapping: Gemma (3 and 4 alike) has no thinking toggle
        // (unlike Qwen3.5), so `Mode` intentionally injects nothing here.
        let text = try await Self.generate(
            container: container,
            instructions: instructions.isEmpty ? nil : instructions,
            history: priorTurns,
            prompt: prompt,
            parameters: parameters
        )
        return LLMResponse(text: text, finishReason: .stop)
    }

    // MARK: - Structured output (contract 1.16.0, ENGINE-NEEDS N6)

    /// Grammar-constrained generation for `responseFormat` requests: template the full
    /// message list via `UserInput`/`UserInputProcessor.prepare`, then drive `TokenIterator`
    /// with `JSONConstrainedLogitProcessor` masking every token that can't extend a valid
    /// JSON prefix. Generation stops by construction once the top-level value completes.
    private func runStructured(container: ModelContainer,
                               request llm: LLMRequest,
                               parameters: GenerateParameters) async throws -> LLMResponse {
        guard let format = llm.responseFormat else {
            throw PackageError.configurationMismatch(expected: "responseFormat", got: "nil")
        }

        // Contract format → grammar container. C12: defaults on both additive enums.
        let grammarContainer: JSONStateMachine.Container
        switch format {
        case .json(let container):
            switch container {
            case .object: grammarContainer = .object
            case .array:  grammarContainer = .array
            case .any:    grammarContainer = .any
            @unknown default: grammarContainer = .any
            }
        case .jsonSchema(let schema):
            // V1 best-effort lane: valid-JSON syntax + container inferred from the schema
            // root; field shape stays prompt-steered (documented in the contract).
            grammarContainer = JSONSchemaHint.container(fromSchema: schema)
        @unknown default:
            throw PackageError.unsupportedRequestFeature(
                "responseFormat: unrecognized case (package built against an older contract)")
        }

        // Once-per-residency vocab classification (SentencePiece pieces → bytes).
        if constrainedVocabulary == nil {
            constrainedVocabulary = await container.perform { context in
                var eosIDs = context.configuration.eosTokenIds
                if let id = context.tokenizer.eosTokenId { eosIDs.insert(id) }
                for token in context.configuration.extraEOSTokens {
                    if let id = context.tokenizer.convertTokenToId(token) { eosIDs.insert(id) }
                }
                return TokenVocabulary(
                    pieceForID: { context.tokenizer.convertIdToToken($0) },
                    eosTokenIDs: eosIDs)
            }
        }
        guard let vocabulary = constrainedVocabulary else { throw PackageError.notLoaded }

        // Same message mapping as the freeform path (system → instructions up front).
        // `Chat.Message` is not Sendable, so the Sendable pieces cross into `perform` and
        // the chat history materializes inside it.
        let instructions = llm.messages
            .filter { $0.role == .system }
            .map(\.content)
            .joined(separator: "\n\n")
        let turns = llm.messages.filter { $0.role != .system }

        // Backstop cap: constrained decode force-stops at the complete value (or a grammar
        // dead end), but cap pathological in-string rambles when the caller didn't.
        let maxTokens = parameters.maxTokens ?? 2048

        let (text, complete, stats): (String, Bool, JSONConstraintEngine.Stats) =
            try await container.perform { context in
                var history: [Chat.Message] = []
                if !instructions.isEmpty { history.append(.system(instructions)) }
                for turn in turns {
                    switch turn.role {
                    case .assistant: history.append(.assistant(turn.content))
                    case .user, .system: history.append(.user(turn.content))
                    }
                }
                let input = UserInput(chat: history)
                let lmInput = try await context.processor.prepare(input: input)
                let engine = JSONConstraintEngine(vocabulary: vocabulary,
                                                  container: grammarContainer)
                let processor = JSONConstrainedLogitProcessor(engine: engine)
                var iterator = try TokenIterator(
                    input: lmInput, model: context.model, cache: nil,
                    processor: processor, sampler: parameters.sampler(),
                    maxTokens: maxTokens)
                var tokens: [Int] = []
                while let token = iterator.next() {
                    if vocabulary.eosTokenIDs.contains(token) { break }
                    tokens.append(token)
                    try Task.checkCancellation()   // C13: cooperatively evictable
                }
                return (context.tokenizer.decode(tokenIds: tokens),
                        engine.isComplete, engine.stats)
            }
        lastStructuredStats = stats

        // Honest finish reason: `.stop` only when the grammar completed a top-level value;
        // a maxTokens truncation mid-value reports `.length`.
        return LLMResponse(text: text, finishReason: complete ? .stop : .length)
    }

    /// One generation on a fresh `ChatSession`. `ChatSession` is not `Sendable`, so it is
    /// created and consumed entirely inside this `nonisolated` helper from `Sendable` inputs —
    /// it never crosses the `InferenceActor` boundary. `ModelContainer` self-isolates.
    private nonisolated static func generate(
        container: ModelContainer,
        instructions: String?,
        history: [ChatMessage],
        prompt: String,
        parameters: GenerateParameters
    ) async throws -> String {
        let chatHistory: [Chat.Message] = history.map { message in
            switch message.role {
            case .assistant: return .assistant(message.content)
            case .user, .system: return .user(message.content)
            }
        }
        let session = ChatSession(
            container,
            instructions: instructions,
            history: chatHistory,
            generateParameters: parameters
        )
        return try await session.respond(to: prompt)
    }
}

extension GemmaLLMPackage {
    /// The author one-liner the engine registers: manifest + license-gated factory.
    public nonisolated static var registration: PackageRegistration {
        .of(GemmaLLMPackage.self)
    }
}
