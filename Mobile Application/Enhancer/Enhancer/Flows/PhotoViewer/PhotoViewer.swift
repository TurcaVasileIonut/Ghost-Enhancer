//
//  PhotoViewer.swift
//  Enhancer
//
//  Created by Turca Vasile  on 24.05.2024.
//

import SwiftUI

struct PhotoViewer: View {
    @Environment(\.dismiss) private var dismiss
    
    @ObservedObject var viewModel: PhotoViewerViewModel
    
    private let supportedAccidentalyScrollForSingleTap = 10.0
    @State private var isPhotoViewerActive: Bool = false
    @State private var scale: CGFloat = 1.0
    
    var body: some View {
        VStack(alignment: .leading) {
            Text(viewModel.currentLabel)
                .foregroundStyle(.white)
            if let image = viewModel.currentImage {
                displayedImage(image: image)
                Spacer()
                tabView(image: image)
            }

        }
        .padding(.horizontal, 20)
        .gesture(MagnificationGesture()
            .onChanged { scale in
                self.scale = min(max(scale.magnitude, 0.5), 2.0)
            })
        .toolbar {
            ToolbarItem(placement: .principal) {
                Text(Date().stripTime)
            }
        }
    }
    
    @ViewBuilder
    private func displayedImage(image: Image) -> some View {
        image
            .resizable()
            .scaleEffect(scale, anchor: .center)
            .gesture(DragGesture(minimumDistance: 0, coordinateSpace: .local)
                .onEnded({ value in
                    if value.translation.width < -supportedAccidentalyScrollForSingleTap, viewModel.imageIndex < viewModel.items.count - 1 {
                        withAnimation {
                            viewModel.imageIndex += 1
                        }
                    }
                    
                    if value.translation.width > supportedAccidentalyScrollForSingleTap, viewModel.imageIndex > 0 {
                        withAnimation {
                            viewModel.imageIndex -= 1
                        }
                    }
                }))
            .frame(width: UIScreen.main.bounds.size.width,
                   height: UIScreen.main.bounds.size.width)
    }
    
    @ViewBuilder
    private func tabView(image: Image) -> some View {
        HStack {
            ShareLink(item: image, preview: SharePreview(viewModel.currentLabel, image: image))
            Spacer()
            Button(action: {
                DispatchQueue.main.async {
                    viewModel.deleteImage()
                    dismiss()
                }
            }, label: {
                HStack {
                    Text(Strings.delete)
                    Image(uiImage: UIImage(imageLiteralResourceName: "trash.fill"))
                }
            })
        }
        .frame(width: UIScreen.main.bounds.size.width)
        .padding(.vertical)
    }
}


#Preview {
    PhotoViewer(viewModel: PhotoViewerViewModel(items: [
        TabItem(label: "Original", image: Image(uiImage: UIImage(imageLiteralResourceName: "Mock"))),
        TabItem(label: "Enhanced", image: Image(uiImage: UIImage(imageLiteralResourceName: "Mock")))
    ]){}) 
}
