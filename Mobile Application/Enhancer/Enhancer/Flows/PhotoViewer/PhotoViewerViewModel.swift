//
//  PhotoViewerViewModel.swift
//  Enhancer
//
//  Created by Turca Vasile  on 24.05.2024.
//

import Foundation
import SwiftUI

@MainActor
class PhotoViewerViewModel: ObservableObject {
    @Published var imageIndex: Int = 0
    @Published private var fullScreenTabView = false
    
    var items: [TabItem]
    var fullScreenImage: Image?
    var deleteImage: () -> ()
    
    init(items: [TabItem], deleteImage: @escaping () -> ()) {
        self.items = items
        self.deleteImage = deleteImage
    }
    
    var currentLabel: String {
        items[imageIndex].label
    }
    
    var currentImage: Image? {
        items[imageIndex].image
    }
    
    func tabViewTap() {
        fullScreenTabView = !fullScreenTabView
    }
    
    func setFullScreenImage(fullScreenImage: Image) {
        self.fullScreenImage = fullScreenImage
    }
}
