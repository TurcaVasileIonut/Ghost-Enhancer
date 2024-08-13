//
//  AlbumCard.swift
//  Enhancer
//
//  Created by Turca Vasile  on 28.05.2024.
//

import SwiftUI

@MainActor
struct HomeCard: View {
    var viewModel: ViewModel
    
    var body: some View {
        NavigationLink(destination: CameraView(viewModel: viewModel.cameraViewModel)) {
            ZStack {
                Rectangle()
                    .fill(.gray)
                viewModel.lastEnhancedImage
                    .resizable()
                    .scaledToFit()
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }
}

