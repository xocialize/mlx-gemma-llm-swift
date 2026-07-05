// RunGemmaLLM — live gates for the Gemma `llm` package (GPU inference runs here, not in the
// SPM test product, whose metallib is unreliable — the fleet CLI-gate convention).
//
//   swift run -c release RunGemmaLLM --smoke [modelDir]       one governed-shape generate
//   swift run -c release RunGemmaLLM --mem-bench [modelDir]   split-footprint measurement:
//       resident floor (post-load) + activation peak at the ~2k-token documented envelope,
//       sampled via phys_footprint (the governor's basis) — feeds GemmaModel.peakActivationBytes.
//   Add --gemma4 to run the Gemma-4 12B variant (gemma4_unified via the text factory) — the
//   configuration then carries the gemma-4 catalog entry so labels/declarations match.

import Foundation
import MLX
import MLXToolKit
import MLXGemmaLLM

let defaultModelDir = "/Volumes/DEV_ARCHIVE/models/mlx-community/gemma-3-12b-it-4bit"

func gbOf(_ b: UInt64) -> Double { Double(b) / 1_000_000_000.0 }

/// OS `phys_footprint` via `task_info(TASK_VM_INFO)` — the MemoryGovernor's basis.
func physFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
}

/// Background phys high-water sampler (the peak is a transient inside prefill/decode).
final class PhysSampler: @unchecked Sendable {
    private let lock = NSLock()
    private var _max: UInt64 = 0
    private var _running = false
    func start() {
        lock.lock(); _running = true; lock.unlock()
        let t = Thread { [weak self] in
            while self?.running == true {
                self?.observe(physFootprintBytes())
                Thread.sleep(forTimeInterval: 0.025)
            }
        }
        t.stackSize = 1 << 20
        t.start()
    }
    var running: Bool { lock.lock(); defer { lock.unlock() }; return _running }
    func observe(_ p: UInt64) { lock.lock(); if p > _max { _max = p }; lock.unlock() }
    func resetMax() { lock.lock(); _max = physFootprintBytes(); lock.unlock() }
    func maxBytes() -> UInt64 { lock.lock(); defer { lock.unlock() }; return _max }
    func stop() { lock.lock(); _running = false; lock.unlock() }
}

/// Page weight files into the OS cache first — outside the engine there's no WeightPrewarmer,
/// and cold faults off the archive volume inside a live command buffer trip the GPU watchdog.
/// Per-iteration autoreleasepool is load-bearing: without it every "discarded" chunk stays
/// autoreleased-alive (~the file size in dead Data) and the residency itself triggers the crash.
func prewarmFiles(in directory: URL) {
    let files = ((try? FileManager.default.contentsOfDirectory(
        at: directory, includingPropertiesForKeys: nil)) ?? [])
        .filter { $0.pathExtension == "safetensors" }
    for p in files {
        guard let fh = try? FileHandle(forReadingFrom: p) else { continue }
        defer { try? fh.close() }
        var done = false
        while !done {
            autoreleasepool {
                guard let chunk = try? fh.read(upToCount: 64 << 20), !chunk.isEmpty else {
                    done = true
                    return
                }
                _ = chunk.count
            }
        }
    }
}

/// An enhancement-shaped request: a long system template + a brief, ~the PE working set.
func enhancementRequest(maxTokens: Int) -> LLMRequest {
    let system = String(
        repeating: "You are a cinematic prompt writer for a text-to-video model. "
            + "Write one flowing present-tense paragraph with explicit camera movement and a "
            + "synchronized audio description. Scale detail to the clip duration. ",
        count: 24)  // ~1.2k tokens of instructions — the envelope-representative prefill
    return LLMRequest(
        messages: [
            .init(role: .system, content: system),
            .init(role: .user, content: "a street musician playing violin in the rain"),
        ],
        parameters: .init(temperature: 0.7, topP: 0.95, maxTokens: maxTokens))
}

@InferenceActor
func smoke(modelDir: String, model: GemmaModel) async throws {
    let cfg = GemmaLLMConfiguration(model: model, modelDirectory: URL(fileURLWithPath: modelDir))
    prewarmFiles(in: URL(fileURLWithPath: modelDir))
    let pkg = GemmaLLMPackage(configuration: cfg)
    let t0 = Date()
    try await pkg.load()
    print(String(format: "[smoke] load %.1fs", Date().timeIntervalSince(t0)))
    let r0 = Date()
    let resp = try await pkg.run(enhancementRequest(maxTokens: 256)) as! LLMResponse
    print(String(format: "[smoke] run %.1fs · %d chars", Date().timeIntervalSince(r0), resp.text.count))
    print("[smoke] ---\n\(resp.text.prefix(400))\n[smoke] ---")
    await pkg.unload()
    print(resp.text.isEmpty ? "[smoke] FAIL ❌ (empty)" : "[smoke] PASS ✅")
}

@InferenceActor
func memBench(modelDir: String, model: GemmaModel) async throws {
    let cfg = GemmaLLMConfiguration(model: model, modelDirectory: URL(fileURLWithPath: modelDir))
    print("[mem-bench] \(cfg.model.displayName) · envelope \(GemmaModel.contextEnvelopeTokens) tokens")
    let p0 = Date()
    prewarmFiles(in: URL(fileURLWithPath: modelDir))
    print(String(format: "[mem-bench] prewarm %.1fs", Date().timeIntervalSince(p0)))

    let sampler = PhysSampler(); sampler.start()
    let pkg = GemmaLLMPackage(configuration: cfg)
    let t0 = Date()
    try await pkg.load()
    print(String(format: "[mem-bench] load %.1fs  phys-after-load=%.2f GB",
                 Date().timeIntervalSince(t0), gbOf(physFootprintBytes())))

    // Warmup (kernel compile; excluded from the peak), then the floor.
    _ = try await pkg.run(enhancementRequest(maxTokens: 16))
    Memory.clearCache()
    let floor = physFootprintBytes()
    print(String(format: "[mem-bench] resident floor (post-warmup + clearCache): %.2f GB", gbOf(floor)))

    // Measured run at the envelope: ~1.2k-token prefill + generation to ~2k total.
    sampler.resetMax()
    let r0 = Date()
    let resp = try await pkg.run(enhancementRequest(maxTokens: 800)) as! LLMResponse
    let peak = sampler.maxBytes(); sampler.stop()
    let activation = peak > floor ? peak - floor : 0
    print(String(format: "[mem-bench] run %.1fs · %d chars", Date().timeIntervalSince(r0), resp.text.count))
    print(String(format: "[mem-bench] SPLIT floor=%.2f GB  peak=%.2f GB  act=%.2f GB",
                 gbOf(floor), gbOf(peak), gbOf(activation)))
    print(String(format: "[mem-bench] DECLARE → residentBytes ≈ %llu  peakActivationBytes ≈ %llu (+headroom)",
                 floor, activation))
    await pkg.unload()
}

/// Structured-output gate (N6, shared kit with mlx-qwen-llm-swift — the FULL 3-case × N=20
/// statistical gate runs there on 0.8B-8bit; this one proves the SAME wiring holds on the
/// SentencePiece/Gemma tokenizer + text factory).
@InferenceActor
func structuredGate(modelDir: String, model: GemmaModel, runs: Int) async throws {
    let cfg = GemmaLLMConfiguration(model: model, modelDirectory: URL(fileURLWithPath: modelDir))
    prewarmFiles(in: URL(fileURLWithPath: modelDir))
    let pkg = GemmaLLMPackage(configuration: cfg)
    try await pkg.load()
    _ = try await pkg.run(LLMRequest(prompt: "Hi",
                                     parameters: .init(temperature: 0, maxTokens: 8)))

    struct Case { let name: String; let system: String; let user: String
                  let format: ResponseFormat; let expectsArray: Bool }
    let cases = [
        Case(name: "parseFacts",
             system: "You extract stable facts about the user. Respond with ONLY a JSON "
                 + "array of short fact strings. No prose, no code fences.",
             user: "User said: \"I'm Marisol, I live in Reykjavík with my two cats, and I "
                 + "teach piano on weekends.\" Extract the facts.",
             format: .json(container: .array), expectsArray: true),
        Case(name: "parseAffect",
             system: "You read the user's emotional state. Respond with ONLY a JSON object "
                 + "{\"mood\": string, \"energy\": number 0..1, \"valence\": number -1..1}. No prose.",
             user: "User said: \"today was a lot. the recital went fine but I'm wiped.\"",
             format: .json(container: .object), expectsArray: false),
        Case(name: "decideSearchQuery",
             system: "Decide whether answering needs a web search. Respond with ONLY "
                 + "{\"action\": \"search\", \"query\": \"<terms>\"} or {\"action\": \"none\"}.",
             user: "User asked: \"what's the weather in Reykjavík this weekend?\"",
             format: .json(container: .object), expectsArray: false),
    ]

    var failures: [String] = []
    var maskSeconds = 0.0
    var maskSteps = 0
    for c in cases {
        var ok = 0
        var secs = 0.0
        for _ in 0..<runs {
            let t0 = Date()
            let resp = try await pkg.run(LLMRequest(
                messages: [.init(role: .system, content: c.system),
                           .init(role: .user, content: c.user)],
                parameters: .init(temperature: 0.7, topP: 0.95, maxTokens: 256),
                responseFormat: c.format)) as! LLMResponse
            secs += Date().timeIntervalSince(t0)
            let trimmed = resp.text.trimmingCharacters(in: .whitespacesAndNewlines)
            let parsed = trimmed.data(using: .utf8)
                .flatMap { try? JSONSerialization.jsonObject(with: $0) }
            let containerOK = c.expectsArray ? parsed is [Any] : parsed is [String: Any]
            if containerOK, resp.finishReason == .stop {
                ok += 1
            } else {
                print("[structured]   ✗ \(c.name): \(trimmed.prefix(160))")
            }
            if let stats = pkg.lastStructuredStats {
                maskSeconds += stats.maskSeconds
                maskSteps += stats.steps
            }
        }
        print(String(format: "[structured] %-18s strict-parse %2d/%d · avg %.2fs",
                     (c.name as NSString).utf8String!, ok, runs, secs / Double(runs)))
        if ok != runs { failures.append("\(c.name): \(ok)/\(runs) (must be 100%)") }
    }
    if maskSteps > 0 {
        print(String(format: "[structured] masking overhead: %.3f ms/step over %d steps",
                     maskSeconds / Double(maskSteps) * 1000, maskSteps))
    }
    await pkg.unload()
    if failures.isEmpty {
        print("[structured] PASS ✅")
    } else {
        print("[structured] FAIL ❌")
        for f in failures { print("[structured]   - \(f)") }
        exit(1)
    }
}

let defaultGemma4ModelDir = "/Volumes/DEV_ARCHIVE/models/mlx-community/gemma-4-12b-it-4bit"

let args = CommandLine.arguments
// Positional = everything that isn't a flag or a flag's value (--runs N).
var nonFlagArgs = Array(args.dropFirst())
if let i = nonFlagArgs.firstIndex(of: "--runs"), i + 1 < nonFlagArgs.count {
    nonFlagArgs.remove(at: i + 1)
}
let positional = nonFlagArgs.filter { !$0.hasPrefix("--") }
let gemma4 = args.contains("--gemma4")
let model = gemma4
    ? GemmaModel(family: .gemma4, size: .b12, quant: .int4)
    : GemmaModel.default
let modelDir = positional.first ?? (gemma4 ? defaultGemma4ModelDir : defaultModelDir)

if args.contains("--mem-bench") {
    try await memBench(modelDir: modelDir, model: model)
} else if args.contains("--structured") {
    var runs = 5
    if let i = args.firstIndex(of: "--runs"), i + 1 < args.count, let n = Int(args[i + 1]) {
        runs = n
    }
    try await structuredGate(modelDir: modelDir, model: model, runs: runs)
} else if args.contains("--smoke") {
    try await smoke(modelDir: modelDir, model: model)
} else {
    print("usage: RunGemmaLLM --smoke | --mem-bench | --structured [--runs N]  [--gemma4]  [modelDir]")
    print("  default modelDir: \(defaultModelDir)")
    print("  default --gemma4 modelDir: \(defaultGemma4ModelDir)")
}
