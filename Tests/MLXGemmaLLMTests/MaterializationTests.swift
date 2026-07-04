import Foundation
import Testing
import MLXServeConformance
import MLXToolKit

@testable import MLXGemmaLLM

// The engine's offline MAT-1..5 auto-materialization gate (engine ≥0.19.0) plus the
// package-specific declaration/probe/resolution behavior. No network, no weights — the
// satisfied configurations use tiny probe files in a temp dir.

/// A temp dir holding the `config.json` probe that makes an explicit-dir config read as
/// satisfied (the same cheap probe `missingWeightSources` uses).
private func satisfiedModelDir() throws -> (dir: URL, cleanup: () -> Void) {
    let base = FileManager.default.temporaryDirectory
        .appending(path: "gemma-mat-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
    FileManager.default.createFile(
        atPath: base.appending(path: "config.json").path, contents: Data([0]))
    return (base, { try? FileManager.default.removeItem(at: base) })
}

@Suite struct GemmaMaterializationTests {

    // MARK: - Engine MAT gate, per selectable tier (the declaration changes with the repo)

    @Test(arguments: GemmaModel.allPublished)
    func matGatePassesEveryPublishedCheckpoint(model: GemmaModel) throws {
        let (dir, cleanup) = try satisfiedModelDir()
        defer { cleanup() }
        let report = MaterializationConformance.check(
            freshConfiguration: GemmaLLMConfiguration(model: model),
            satisfiedConfiguration: GemmaLLMConfiguration(model: model, modelDirectory: dir))
        #expect(report.passed, "\(model.displayName):\n\(report.summary)")
    }

    // MARK: - Source declaration shape

    @Test func declaresOneCheckpointSourcePerConfig() {
        let sources = GemmaLLMConfiguration().weightSources
        #expect(sources.map(\.role) == ["checkpoint"])
        #expect(sources[0].repo == "mlx-community/gemma-3-12b-it-4bit")
        // Quant is tiered by REPO, so the whole snapshot is fetched (no globs to exclude).
        #expect(sources[0].matching == nil)
        // The family/quant axes select the repo.
        let g4int8 = GemmaLLMConfiguration(
            model: GemmaModel(family: .gemma4, size: .b12, quant: .int8))
        #expect(g4int8.weightSources[0].repo == "mlx-community/gemma-4-12b-it-8bit")
    }

    @Test func pinnedRevisionRidesTheSource() {
        let cfg = GemmaLLMConfiguration(revision: "abc123")
        #expect(cfg.weightSources[0].revision == "abc123")
        #expect(GemmaLLMConfiguration().weightSources[0].revision == nil)
    }

    @Test func unpublishedQuantDeclaresNothing() {
        // No mlx-community suffix ⇒ no repo ⇒ no declarable source; such a config is
        // explicit-directory-only (load() diagnoses the dir-less case).
        let cfg = GemmaLLMConfiguration(model: GemmaModel(size: .b12, quant: .int5))
        #expect(cfg.weightsRepo == nil)
        #expect(cfg.weightSources.isEmpty)
        #expect(cfg.missingWeightSources(storeRoot: nil).isEmpty)
    }

    // MARK: - Store-layout probe + resolution

    @Test func storeLayoutSatisfiesAndResolves() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "gemma-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = GemmaLLMConfiguration()
        // Empty store: the checkpoint is missing.
        #expect(cfg.missingWeightSources(storeRoot: root).count == 1)
        // Populate the expected <root>/<org>/<name> layout.
        let repoDir = root.appending(path: "mlx-community/gemma-3-12b-it-4bit")
        try FileManager.default.createDirectory(at: repoDir, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: repoDir.appending(path: "config.json").path, contents: Data([0]))
        #expect(cfg.missingWeightSources(storeRoot: root).isEmpty)
        // Resolution lands on the store layout.
        #expect(cfg.resolvedModelDirectory(storeRoot: root)?.path == repoDir.path)
        // A sibling quant is a DIFFERENT repo dir — still missing.
        let int8 = GemmaLLMConfiguration(model: GemmaModel(size: .b12, quant: .int8))
        #expect(int8.missingWeightSources(storeRoot: root).count == 1)
    }

    @Test func explicitDirectoryWinsOverStore() throws {
        let (dir, cleanup) = try satisfiedModelDir()
        defer { cleanup() }
        var cfg = GemmaLLMConfiguration()
        cfg.modelDirectory = dir
        // Satisfied by the explicit path even with no store at all (MAT-5 semantics).
        #expect(cfg.missingWeightSources(storeRoot: nil).isEmpty)
        // And resolution prefers it over any store layout.
        let root = URL(fileURLWithPath: "/tmp/some-store")
        #expect(cfg.resolvedModelDirectory(storeRoot: root) == dir)
    }

    @Test func prewarmResolvesTheStoreLayout() {
        // Nil dir + stamped store ⇒ prewarm targets the resolved store layout, so downloaded
        // weights prewarm from the second cold launch on (paths may not exist on a true first
        // run — the prewarmer is best-effort and skips them).
        let root = URL(fileURLWithPath: "/tmp/some-store")
        var cfg = GemmaLLMConfiguration()
        cfg.modelsRootDirectory = root
        #expect(cfg.prewarmPaths.map(\.path)
            == [root.appending(path: "mlx-community/gemma-3-12b-it-4bit").path])
        // Explicit directory still wins.
        cfg.modelDirectory = URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/models/gemma")
        #expect(cfg.prewarmPaths == [URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/models/gemma")])
        // Nothing to prewarm without a directory or a store.
        #expect(GemmaLLMConfiguration().prewarmPaths.isEmpty)
    }
}
