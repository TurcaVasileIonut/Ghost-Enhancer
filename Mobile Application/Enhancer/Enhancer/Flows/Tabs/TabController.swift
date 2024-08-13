//
//  HomeScreen.swift
//  Enhancer
//
//  Created by Turca Vasile  on 24.05.2024.
//

import SwiftUI

struct TabController: View {
    
    @State private var selectedTab: Tab = .home
    var viewModel: ViewModel

    enum Tab: Equatable {
        case home
        case camera
        case gallery
    }
    
    var body: some View {
        NavigationStack {
            TabView(selection: $selectedTab) {
                HomeView(viewModel: viewModel.homeViewModel)
                    .tabItem {
                        tabIcon(iconSystemName: "photo.stack", tabName: Strings.albums)
                    }
                    .tag(Tab.home)
                
                CameraView(viewModel: viewModel.cameraViewModel)
                    .tabItem {
                        tabIcon(iconSystemName: "plus", tabName: "")
                    }
                    .tag(Tab.camera)
                
                GalleryView(viewModel: viewModel.galleryViewModel)
                    .tabItem {
                        tabIcon(iconSystemName: "photo", tabName: Strings.gallery)
                    }
                    .tag(Tab.gallery)
            }
            .onChange(of: selectedTab) {
                viewModel.tabChanged(tab: selectedTab)
            }
        }
    }
    
    @ViewBuilder
    func tabIcon(iconSystemName: String, tabName: String) -> some View {
        VStack {
            Image(systemName: iconSystemName)
            Text(tabName)
        }
    }
}

#Preview {
    @Environment(\.modelContext) var modelContext
    return TabController(viewModel: TabController.ViewModel(modelContext: modelContext))
}
