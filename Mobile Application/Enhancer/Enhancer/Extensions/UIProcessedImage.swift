//
//  ProcessedImage.swift
//  Enhancer
//
//  Created by Turca Vasile  on 29.05.2024.
//

import SwiftUI

extension ProcessedImage {
    var originalImage: Image {
        guard let uiImage = UIImage(data: originalImageData) else {
            return Image(systemName: "photo.circle.fill")
        }
        return Image(uiImage: uiImage)
    }
    
    var enhancedImage: Image {
        guard let uiImage = UIImage(data: enhancedImageData) else {
            return Image(systemName: "photo.circle.fill")
        }
        return Image(uiImage: uiImage)
    }
}
