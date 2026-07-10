// CancellationTests.swift — Gemma LLM through the engine's CAN gate (offline, no MLX kernels).
// CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before notLoaded
// validation or weights). CAN-3 is the document of record for the checkpoint cadence: BOTH
// generation paths bail per generated token — the freeform (fresh-ChatSession) path via
// mlx-swift-lm's own per-token `Task.isCancelled` check plus the wrapper's post-respond
// `try Task.checkCancellation()` (which converts the silent partial return into the canonical
// CancellationError), and the structured path via the package-owned TokenIterator drive.

import Foundation
import MLXServeConformance
import MLXToolKit
import Testing

@testable import MLXGemmaLLM

@Suite struct GemmaCancellationTests {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    @Test func canGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = GemmaLLMPackage(configuration: GemmaLLMConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: LLMRequest(prompt: "probe"))
        #expect(report.passed, "\(report.summary)")
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    @Test func canCadenceDeclaration() {
        // The STATIC manifest describes the default variant (gemma-3 12B int4, ~0.97 GB
        // activation), which stays under the 2 GB long-run threshold and `llm` is not a
        // long-run capability — so the static manifest does NOT imply long runs...
        #expect(!CancellationConformance.longRunImplied(by: GemmaLLMPackage.manifest))
        // ...but the gemma-4 configuration of this same package measures 2.16 GB (declared
        // 2.7 GB) at the 2k envelope, which crosses the threshold — so the package declares
        // a real cadence rather than leaning on a sub-second exemption.
        #expect(GemmaModel(family: .gemma4, size: .b12, quant: .int4).peakActivationBytes
            >= CancellationConformance.longRunActivationBytes)

        let report = CancellationConformance.checkCadence(
            manifest: GemmaLLMPackage.manifest,
            posture: .cadence([
                // Per generated token, on both paths:
                // — Freeform path (fresh ChatSession per run): mlx-swift-lm 3.31.4's
                //   generation loop checks `Task.isCancelled` once per token
                //   (MLXLMCommon/Evaluate.swift, `tokenLoop`) and stops; the wrapper's
                //   post-respond `try Task.checkCancellation()` (GemmaLLMPackage.generate)
                //   converts the silent partial return into the canonical CancellationError.
                // — Structured path: the package-owned TokenIterator drive checks
                //   `try Task.checkCancellation()` once per generated token
                //   (GemmaLLMPackage.runStructured, the `while let token = iterator.next()`
                //   loop).
                .init(phase: .generate, unit: .token),
            ]))
        #expect(report.passed, "\(report.summary)")
    }
}
