//
//  CameraViewModel.swift
//  Enhancer
//
//  Created by Turca Vasile  on 24.05.2024.
//

import SwiftUI
import PhotosUI
import SwiftData

extension CameraView {
    
    @Observable
    @MainActor
    class ViewModel {
        
        private var superResolutionService = SuperResolutionService()
        
        private var modelContext: ModelContext
        var images = [ProcessedImage]()
        var album: Album {
            didSet {
                fetchData()
            }
        }
        
        init(modelContext: ModelContext, album: Album? = nil) {
            self.modelContext = modelContext
            self.album = album ?? Album()
        }
        
        private func fetchData() {
            do {
                let descriptor = FetchDescriptor<ProcessedImage>(
                    sortBy: [SortDescriptor(\.date, order: .reverse)]
                )
                images = try modelContext.fetch(descriptor)
                images = images.filter { image in
                    image.album.id == album.id
                }
            } catch {
                print("Fetch failed")
            }
        }
        
        func refreshImages() {
            fetchData()
        }
        
        func photoViewerViewModel(image: ProcessedImage, deleteImage: @escaping () -> ()) -> PhotoViewerViewModel {
            PhotoViewerViewModel(items: [TabItem(label: "Original", image: image.originalImage),
                                         TabItem(label: "Enhanced", image: image.enhancedImage)],
                                 deleteImage: deleteImage)
        }
        
        func save(image: UIImage?) {
            Task {
                guard let image = image,
                      let originalImageData = image.pngData() else {
                    return
                }
                
                let enhancedImage = await superResolutionService.analyzeImage(image: image)
                let enhancedImageData = enhancedImage?.pngData() ?? originalImageData
                let processedImage = ProcessedImage(originalImage: originalImageData, enhancedImage: enhancedImageData, album: album)
                
                modelContext.insert(processedImage)
                images.append(processedImage)
                print("images.append")
            }
        }
        
        func delete(image: ProcessedImage) {
            modelContext.delete(image)
            fetchData()
        }
    }
}
