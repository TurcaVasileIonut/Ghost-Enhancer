//
//  HomeScreenViewModel.swift
//  Enhancer
//
//  Created by Turca Vasile  on 24.05.2024.
//

import SwiftUI
import SwiftData

extension TabController {
    
    @Observable
    @MainActor
    class ViewModel {
        var galleryViewModel: GalleryView.ViewModel
        var cameraViewModel: CameraView.ViewModel
        var homeViewModel: HomeView.ViewModel
        
        init(modelContext: ModelContext) {
            self.galleryViewModel = GalleryView.ViewModel(modelContext: modelContext)
            self.cameraViewModel = CameraView.ViewModel(modelContext: modelContext)
            self.homeViewModel = HomeView.ViewModel(modelContext: modelContext)
        }
        
        func tabChanged(tab: Tab) {
            guard tab == .camera else {
                homeViewModel.refresh()
                galleryViewModel.refresh()
                return
            }
            cameraViewModel.album = Album()
        }
    }
}
