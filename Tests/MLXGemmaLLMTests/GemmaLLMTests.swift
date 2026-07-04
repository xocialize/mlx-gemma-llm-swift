import Foundation
import Testing
import MLXToolKit
import MLXServeCore

@testable import MLXGemmaLLM

// Offline conformance: manifest, license, footprints, config Codable, and the engine's
// admissibility over the catalog. Pure logic — no MLX kernels, no downloads.

private func engine(budgetBytes: UInt64) -> MLXServeEngine {
    let device = DeviceProfile(
        chipTier: .max,
        macOS: SemanticVersion(major: 26, minor: 0, patch: 0),
        backends: [.metalGPU],
        totalMemoryBytes: 128_000_000_000
    )
    return MLXServeEngine(device: device, governor: MemoryGovernor(budgetBytes: budgetBytes))
}

@Suite struct GemmaManifestTests {

    @Test func manifestDeclaresGemmaTermsAndLLMSurface() {
        let m = GemmaLLMPackage.manifest
        #expect(m.license.weightLicense == .gemmaTerms)
        #expect(m.license.portCodeLicense == .apache2)
        #expect(m.surfaces.contains { $0.name == "gemma-3-llm" })
    }

    @Test func gemmaTermsIsPermissiveAllowlisted() {
        // The whole point of P0: the default `.permissiveOnly` policy admits Gemma weights.
        #expect(SPDXLicense.gemmaTerms.isPermissive)
        #expect(LicensePolicy.permissiveOnly.evaluate(GemmaLLMPackage.manifest.license).isAdmitted)
    }

    @Test func catalogIsTheKnownPublishedSet() {
        #expect(GemmaModel.allPublished.count == 6)  // {gemma3, gemma4} × 12B-it × {4bit, 8bit, bf16}
        #expect(GemmaModel.allPublished.contains(.default))  // gemma-3 12B-it 4-bit
        #expect(GemmaModel.allPublished.allSatisfy { $0.weightsRepo != nil })
        #expect(GemmaModel.default.weightsRepo == "mlx-community/gemma-3-12b-it-4bit")
        // The default stays gemma-3 (the LTX zero-extra-download property).
        #expect(GemmaModel.default.family == .gemma3)
    }

    @Test func gemma4CatalogAndRepoNaming() {
        let g4 = GemmaModel(family: .gemma4, size: .b12, quant: .int4)
        #expect(g4.weightsRepo == "mlx-community/gemma-4-12b-it-4bit")
        #expect(g4.displayName == "Gemma 4 · 12b-it (int4)")
        #expect(GemmaModel.allPublished.filter { $0.family == .gemma4 }.count == 3)
    }

    @Test func gemma4WeightLicenseIsApacheAndPermissive() {
        // Per-family license record: gemma-3 = Gemma Terms; gemma-4 upstream = Apache-2.0.
        #expect(GemmaFamily.gemma3.weightLicense == .gemmaTerms)
        #expect(GemmaFamily.gemma4.weightLicense == .apache2)
        // Both permissive → the static manifest's stricter gemmaTerms declaration is
        // conservative-correct for every variant (gate outcome identical).
        #expect(GemmaFamily.allCases.allSatisfy { $0.weightLicense.isPermissive })
    }

    @Test func gemma4ExpectedTextModelTypeGuardsShadowing() {
        // The MLXVLM-shadowing guard is family-aware: gemma4_unified must resolve to the
        // TEXT factory's Gemma4Model, never MLXVLM's Gemma4Unified.
        #expect(GemmaFamily.gemma3.expectedTextModelType == "Gemma3TextModel")
        #expect(GemmaFamily.gemma4.expectedTextModelType == "Gemma4Model")
    }

    @Test func gemma4SharesThe12BHybridGeometry() {
        // Verified against both published configs: identical decoder geometry, so the
        // analytic KV is family-invariant at 12B.
        let g3 = GemmaModel(family: .gemma3, size: .b12, quant: .int4)
        let g4 = GemmaModel(family: .gemma4, size: .b12, quant: .int4)
        #expect(g3.kvCacheBytes(maxTokens: 2048) == g4.kvCacheBytes(maxTokens: 2048))
    }

    @Test func gemma4FootprintIsMeasuredNotAnalytic() {
        // mem-bench 2026-07-04: floor 7.11 GB, activation 2.16 GB at the 2k envelope —
        // the activation is ~4.6× the analytic KV (gemma4_unified prefill scratch), so
        // the declaration is the MEASURED number + headroom, not the gemma-3 analytic.
        let g4 = GemmaModel(family: .gemma4, size: .b12, quant: .int4)
        #expect(g4.residentBytes == 8_000_000_000)         // measured 7.11 GB + headroom
        #expect(g4.peakActivationBytes == 2_700_000_000)   // measured 2.16 GB + headroom
        #expect(g4.peakActivationBytes > g4.kvCacheBytes(maxTokens: GemmaModel.contextEnvelopeTokens))
        #expect(g4.residentBytes > g4.onDiskBytes)         // floor > disk: materialized + overhead
    }

    @Test func legacyPayloadWithoutFamilyDecodesAsGemma3() throws {
        // v0.1 GemmaModel payloads carry only {size, quant} — they must keep decoding
        // (as gemma-3) so persisted configurations survive the family-axis addition.
        let legacy = Data(#"{"size":"12b","quant":"int8"}"#.utf8)
        let decoded = try JSONDecoder().decode(GemmaModel.self, from: legacy)
        #expect(decoded.family == .gemma3)
        #expect(decoded == GemmaModel(family: .gemma3, size: .b12, quant: .int8))
    }

    @Test func footprintSplitsResidentFromActivation() {
        let model = GemmaModel.default
        #expect(model.residentBytes == 10_000_000_000)  // mem-bench floor 9.32 GB (2026-07-02)
        #expect(model.residentBytes > model.onDiskBytes)  // floor > disk: materialized + overhead
        #expect(model.peakActivationBytes > model.kvCacheBytes(maxTokens: GemmaModel.contextEnvelopeTokens))
        // Measured activation at the envelope was 0.37 GB; the declaration must cover it.
        #expect(model.peakActivationBytes > 370_000_000)
        #expect(model.requirements.footprints.first?.residentBytes == model.residentBytes)
    }

    @Test func kvCacheHybridGeometry() {
        let model = GemmaModel.default
        // At the 2048 envelope: 8 global layers grow to 2048; 40 sliding layers cap at 1024.
        // 2(K+V) × [8×2048 + 40×1024] × 8 heads × 256 dim × 2 B = 469,762,048.
        #expect(model.kvCacheBytes(maxTokens: 2048) == 469_762_048)
        // Below the window, sliding layers scale too: all 48 layers at T=512.
        #expect(model.kvCacheBytes(maxTokens: 512) == UInt64(2 * 48 * 512 * 8 * 256 * 2))
        // Past the envelope only the 8 global layers keep growing.
        let at4k = model.kvCacheBytes(maxTokens: 4096)
        let at2k = model.kvCacheBytes(maxTokens: 2048)
        #expect(at4k - at2k == UInt64(2 * 8 * 2048 * 8 * 256 * 2))
    }

    @Test func configurationCodableRoundTripsPortableFieldsOnly() throws {
        var config = GemmaLLMConfiguration(
            model: GemmaModel(size: .b12, quant: .int8),
            revision: "abc123")
        config.modelDirectory = URL(fileURLWithPath: "/Volumes/somewhere")
        config.modelsRootDirectory = URL(fileURLWithPath: "/Volumes/elsewhere")
        let decoded = try JSONDecoder().decode(
            GemmaLLMConfiguration.self,
            from: JSONEncoder().encode(config))
        #expect(decoded.model == config.model)
        #expect(decoded.revision == "abc123")
        // Environment-specific URLs are intentionally NOT portable.
        #expect(decoded.modelDirectory == nil)
        #expect(decoded.modelsRootDirectory == nil)
    }

    @Test func configurationHintsMatchSelectedVariant() {
        let config = GemmaLLMConfiguration(model: GemmaModel(size: .b12, quant: .bf16))
        #expect(config.quant == .bf16)
        #expect(config.residentBytesHint == config.model.residentBytes)
        #expect(config.peakActivationBytesHint == config.model.peakActivationBytes)
    }

    @Test func prewarmOnlyForDirectDirectory() {
        var config = GemmaLLMConfiguration()
        #expect(config.prewarmPaths.isEmpty)
        config.modelDirectory = URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/models/gemma")
        #expect(config.prewarmPaths == [URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/models/gemma")])
    }
}

@Suite struct GemmaAdmissibilityTests {

    @Test func allVariantsFitOnLargeBudget() async {
        let e = engine(budgetBytes: 64_000_000_000)
        for model in GemmaModel.allPublished {
            let verdict = await e.admissibility(for: model.requirements)
            #expect(verdict.admissible, "\(model.displayName) should be admissible at 64 GB")
        }
    }

    @Test func smallBudgetAdmitsOnly4bit() async {
        // 12 GB budget: the 4-bit floors (+ ~1 GB activation) fit; 8-bit/bf16 don't.
        let e = engine(budgetBytes: 12_000_000_000)
        var admissible: Set<GemmaModel> = []
        for model in GemmaModel.allPublished
        where await e.admissibility(for: model.requirements).admissible {
            admissible.insert(model)
        }
        #expect(admissible == [
            GemmaModel(family: .gemma3, size: .b12, quant: .int4),
            GemmaModel(family: .gemma4, size: .b12, quant: .int4),
        ])
    }
}
