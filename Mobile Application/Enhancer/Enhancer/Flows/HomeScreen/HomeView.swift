//
//  HomeView.swift
//  Enhancer
//
//  Created by Turca Vasile  on 25.05.2024.
//

import SwiftUI

@MainActor
struct HomeView: View {
    var viewModel: ViewModel
    
    var body: some View {
        VStack(alignment: .leading) {
            header
            albums
        }
        .padding(16)
    }
    
    @ViewBuilder
    var header: some View {
        Text("Ghost Enhancer")
            .font(.title)
    }
    
    @ViewBuilder
    var albums: some View {
        ScrollView(.vertical, showsIndicators: false) {
            principalAlbum
            Spacer()
                .frame(height: 16)
            secondayAlbumsGrid
        }
    }
    
    @ViewBuilder
    var principalAlbum: some View {
        HomeCard(viewModel: viewModel.primaryCardViewModel)
            .frame(maxHeight: 300)
    }
    
    @ViewBuilder
    var secondayAlbumsGrid: some View {
        GridView(rows: viewModel.secondaryAlbumsRowsCount, columns: viewModel.secondaryAlbumsColumsCount, spacing: 16) { index in
            if let viewModel = viewModel.viewModelFor(albumAt: index) {
                HomeCard(viewModel: viewModel)
                    .frame(maxHeight: 150)
            } else {
                Rectangle()
                    .fill(.clear)
                    .frame(maxHeight: 150)
            }
        }
    }
}
