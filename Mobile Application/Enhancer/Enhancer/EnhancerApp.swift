//
//  EnhancerApp.swift
//  Enhancer
//
//  Created by Turca Vasile  on 23.05.2024.
//

import SwiftUI
import SwiftData

@main
struct EnhancerApp: App {
    let container: ModelContainer
    
    var body: some Scene {
        WindowGroup {
            TabController(viewModel: TabController.ViewModel(modelContext: container.mainContext))
        }
        .modelContainer(container)
    }
    
    init() {
        do {
            container = try ModelContainer(for: ProcessedImage.self, Album.self)
        } catch {
            fatalError("Failed to create ModelContainer for Movie.")
        }
    }
}
