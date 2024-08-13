//
//  EnhancedImage.swift
//  Enhancer
//
//  Created by Turca Vasile  on 23.05.2024.
//

import SwiftUI
import SwiftData

@Model
class ProcessedImage: Identifiable {

    @Attribute(.unique) 
    var id: String = UUID().uuidString
    
    @Attribute
    var date: Date
    
    @Attribute(.externalStorage)
    var enhancedImageData: Data
    
    @Attribute(.externalStorage)
    var originalImageData: Data
    
    @Relationship(inverse: \Album.enhancedImages)
    var album: Album

    init(id: String = UUID().uuidString, date: Date = Date(), originalImage: Data, enhancedImage: Data, album: Album) {
        self.id = id
        self.date = date
        self.originalImageData = originalImage
        self.enhancedImageData = enhancedImage
        self.album = album
    }
}
