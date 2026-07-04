// WeightMaterializer.swift — first-run download of the package's declared weight sources
// (engine ≥0.19.0 auto-materialization: the PACKAGE materializes; the app only picks the
// models folder).
//
// Executes the `WeightSourcing` declaration on GemmaLLMConfiguration: each missing source's
// repo is snapshot-downloaded into the engine ModelStore layout (`<root>/<org>/<name>/…`) via
// swift-huggingface's HubClient, with per-file progress forwarded to `WeightDownloadProgress`
// so the engine's PreparationMonitor surfaces a real `.downloading(fraction:)` phase — a
// download the monitor can't see is a conformance smell.

import Foundation
import HuggingFace
import MLXToolKit

enum WeightMaterializer {

    enum MaterializeError: Error, LocalizedError {
        case badRepoId(String)
        var errorDescription: String? {
            switch self {
            case .badRepoId(let id): return "Malformed weight-source repo id '\(id)' (want org/name)."
            }
        }
    }

    /// Download every `source` into `root` (ModelStore layout). Sources download sequentially so
    /// the reported fraction is monotonic across the whole materialization: source i of n spans
    /// [i/n, (i+1)/n). (Gemma declares a single source; the span math keeps this general.)
    static func materialize(_ sources: [WeightSource], into root: URL) async throws {
        let client = HubClient()   // env-detected endpoint + token; gated repos honor HF_TOKEN
        let store = ModelStore(root: root)
        for (index, source) in sources.enumerated() {
            guard let repoId = Repo.ID(rawValue: source.repo),
                  let destination = store.directory(for: source.repo) else {
                throw MaterializeError.badRepoId(source.repo)
            }
            try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
            let base = Double(index) / Double(sources.count)
            let span = 1.0 / Double(sources.count)
            _ = try await client.downloadSnapshot(
                of: repoId,
                to: destination,
                revision: source.revision ?? "main",
                matching: source.matching ?? [],
                progressHandler: { progress in
                    WeightDownloadProgress.report(
                        fraction: base + span * progress.fractionCompleted,
                        bytesPerSecond: nil)
                })
        }
    }
}
