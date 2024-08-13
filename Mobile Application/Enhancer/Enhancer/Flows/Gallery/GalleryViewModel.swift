//
//  GalleryViewModel.swift
//  Enhancer
//
//  Created by Turca Vasile  on 24.05.2024.
//

import SwiftUI
import SwiftData

extension GalleryView {
    
    @Observable
    @MainActor
    class ViewModel  {
        private let imagesInRow = 3
        
        private var modelContext: ModelContext
        var images = [ProcessedImage]()
        
        private var isLastRowPartialFilled: Bool {
            images.count % imagesInRow != 0
        }
        
        var rowsCount: Int {
            images.count / imagesInRow + (isLastRowPartialFilled ? 1 : 0)
        }
        
        var columnsCount: Int {
            imagesInRow
        }

        init(modelContext: ModelContext) {
            self.modelContext = modelContext
            fetchData()
        }
        
        func imageAt(index: GridIndex) -> ProcessedImage? {
            guard index.row * imagesInRow + index.column < images.count else {
                return nil
            }
            return images[index.row * imagesInRow + index.column]
        }

        private func fetchData() {
            do {
                let descriptor = FetchDescriptor<ProcessedImage>(sortBy: [SortDescriptor(\.date, order: .reverse),])
                images = try modelContext.fetch(descriptor)
            } catch {
                print("Fetch failed")
            }
        }
        
        func refresh() {
            fetchData()
        }
        
        func delete(image: ProcessedImage) {
            modelContext.delete(image)
            fetchData()
        }
        
        func photoViewerViewModel(for image: ProcessedImage, deleteImage: @escaping () -> ()) -> PhotoViewerViewModel {
            PhotoViewerViewModel(items: [TabItem(label: "Original", image: image.originalImage),
                                         TabItem(label: "Enhanced", image: image.enhancedImage)],
                                 deleteImage: deleteImage)
        }
    }
}
