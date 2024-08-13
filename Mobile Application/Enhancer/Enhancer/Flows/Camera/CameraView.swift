//
//  CameraView.swift
//  Enhancer
//
//  Created by Turca Vasile  on 23.05.2024.
//

import SwiftUI
import PhotosUI
import SwiftData

@MainActor
struct CameraView: View {
    var viewModel: ViewModel
    @State private var showCamera = false

    var body: some View {
        VStack(spacing: 10) {
            highlightedPhoto
            photosHStack
            Spacer()
            openCameraButton
        }
        .toolbar {
            photosPicker
        }
        .padding(.horizontal, 16)
        .fullScreenCover(isPresented: self.$showCamera) {
            accessCamera
        }
        .onAppear {
            viewModel.refreshImages()
        }
    }
    
    private func photoViewer(for image: ProcessedImage) -> PhotoViewer {
        PhotoViewer(viewModel: viewModel.photoViewerViewModel(image: image) {
            viewModel.delete(image: image)
        })
    }
    
    @ViewBuilder
    var openCameraButton: some View {
        SolidColorButton(text: Strings.openCamera) {
            self.showCamera.toggle()
        }
    }
    
    @ViewBuilder
    var photosPicker: some View {
        UploadPhoto() { image in
            viewModel.save(image: image)
        }
    }
    
    @ViewBuilder
    var accessCamera: some View {
        AccessCameraView() { image in
            viewModel.save(image: image)
        }
    }

    @ViewBuilder
    var highlightedPhoto: some View {
        if viewModel.images.count > 0 {
            NavigationLink(destination: photoViewer(for: viewModel.images[0])) {
                viewModel.images[0].enhancedImage
                    .resizable()
                    .scaledToFill()
                    .frame(width: UIScreen.main.bounds.size.width, height: UIScreen.main.bounds.size.width)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .padding(16)
            }
        } else {
            Text(Strings.noImage)
                .font(.title)
                .fontWeight(.semibold)
        }
    }
    
    @ViewBuilder
    var photosHStack: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                if viewModel.images.count > 1 {
                    ForEach(1..<viewModel.images.count, id: \.self) { index in
                        NavigationLink(destination: photoViewer(for: viewModel.images[index])) {
                            viewModel.images[index].enhancedImage
                                .resizable()
                                .frame(width: 100, height: 100)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                }
            }
        }
    }
}

#Preview {
    @Environment(\.modelContext) var modelContext
    return CameraView(viewModel: CameraView.ViewModel(modelContext: modelContext))
}
