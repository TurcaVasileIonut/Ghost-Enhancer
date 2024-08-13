//
//  HomeCardViewModel.swift
//  Enhancer
//
//  Created by Turca Vasile  on 28.05.2024.
//

import SwiftData
import SwiftUI

extension HomeCard {
    
    @Observable
    @MainActor
    class ViewModel {
        private var modelContext: ModelContext
        private var album: Album
        
        init(modelContext: ModelContext, album: Album) {
            self.modelContext = modelContext
            self.album = album
        }
        
        var lastEnhancedImage: Image {
            album.lastEnhancedImage?.enhancedImage ?? Image(systemName: "photo.circle.fill")
        }
        
        var cameraViewModel: CameraView.ViewModel {
            CameraView.ViewModel(modelContext: modelContext, album: album)
        }
        
    }
    
}
