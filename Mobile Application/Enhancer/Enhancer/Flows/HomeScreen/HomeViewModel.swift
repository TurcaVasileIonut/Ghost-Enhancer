//
//  HomeViewModel.swift
//  Enhancer
//
//  Created by Turca Vasile  on 25.05.2024.
//

import Foundation
import Combine
import SwiftData

extension HomeView {
    
    @Observable
    @MainActor
    class ViewModel {
        
        let secondaryAlbumsColumsCount = 2
        
        private var modelContext: ModelContext
        var albums = [Album]()
        
        init(modelContext: ModelContext) {
            self.modelContext = modelContext
            fetchAlbums()
        }
        
        var primaryCardViewModel: HomeCard.ViewModel {
            guard let album = albums.first else {
                return HomeCard.ViewModel(modelContext: modelContext, album: Album())
            }
            return HomeCard.ViewModel(modelContext: modelContext, album: album)
        }
        
        var secondaryAlbumsRowsCount: Int {
            guard albums.count > 0 else {
                return 0
            }
            return (albums.count + secondaryAlbumsColumsCount - 2) / secondaryAlbumsColumsCount
        }
        
        func viewModelFor(albumAt index: GridIndex) -> HomeCard.ViewModel? {
            guard 1 + index.row * secondaryAlbumsColumsCount + index.column < albums.count else {
                return nil
            }
            return HomeCard.ViewModel(modelContext: modelContext, album: albums[1 + index.row * secondaryAlbumsColumsCount + index.column])
        }
        
        func refresh() {
            fetchAlbums()
        }
        
        private func fetchAlbums() {
            do {
                let descriptor: FetchDescriptor<Album> = FetchDescriptor<Album>()
                albums = try modelContext.fetch(descriptor)
            } catch {
                print("Fetch failed")
            }
        }
    }
}
