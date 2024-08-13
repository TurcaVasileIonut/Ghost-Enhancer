//
//  Galley.swift
//  Enhancer
//
//  Created by Turca Vasile  on 24.05.2024.
//

import SwiftUI
import SwiftData

@MainActor
struct GalleryView: View {
    var viewModel: ViewModel
    
    private let gridSpacing: CGFloat = 8.0
    private var imageWidth: CGFloat {
        UIScreen.main.bounds.size.width / CGFloat(viewModel.columnsCount) - gridSpacing
    }
    
    private func photoViewer(for image: ProcessedImage) -> PhotoViewer {
        PhotoViewer(viewModel: viewModel.photoViewerViewModel(for: image) {
            viewModel.delete(image: image)
        })
    }
    
    var body: some View {
        VStack(alignment: .leading) {
            header
            photos
        }
        .padding(16)
    }
    
    @ViewBuilder
    var header: some View {
        Text("Ghost Enhancer")
            .font(.title)
    }
    
    @ViewBuilder
    var photos: some View {
        ScrollView(.vertical, showsIndicators: false) {
            GridView(rows: viewModel.rowsCount, columns: viewModel.columnsCount, spacing: gridSpacing) { index in
                if let image = viewModel.imageAt(index: index) {
                    NavigationLink(destination: photoViewer(for: image)) {
                        image.enhancedImage
                            .scaledToFill()
                            .frame(width: imageWidth, height: imageWidth)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                } else {
                    noEventCell
                }
            }
        }
    }
    
    @ViewBuilder
    private var noEventCell: some View {
        Rectangle()
            .fill(.clear)
            .padding()
            .frame(width: imageWidth, height: imageWidth)
    }
}

#Preview {
    @Environment(\.modelContext) var modelContext
    return GalleryView(viewModel: GalleryView.ViewModel(modelContext: modelContext))
}
