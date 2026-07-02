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
        #expect(GemmaModel.allPublished.count == 3)  // 12B-it × {4bit, 8bit, bf16}
        #expect(GemmaModel.allPublished.contains(.default))  // 12B-it 4-bit
        #expect(GemmaModel.allPublished.allSatisfy { $0.weightsRepo != nil })
        #expect(GemmaModel.default.weightsRepo == "mlx-community/gemma-3-12b-it-4bit")
    }

    @Test func footprintSplitsResidentFromActivation() {
        let model = GemmaModel.default
        #expect(model.residentBytes == 7_500_000_000)  // measured on disk 2026-07-02
        #expect(model.peakActivationBytes > model.kvCacheBytes(maxTokens: GemmaModel.contextEnvelopeTokens))
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
        // 12 GB budget: 4-bit charges 7.5 + ~3 GB ≈ 10.5 (fits); 8-bit ≈ 16.4 (doesn't).
        let e = engine(budgetBytes: 12_000_000_000)
        var admissible: Set<GemmaModel> = []
        for model in GemmaModel.allPublished
        where await e.admissibility(for: model.requirements).admissible {
            admissible.insert(model)
        }
        #expect(admissible == [GemmaModel(size: .b12, quant: .int4)])
    }
}
