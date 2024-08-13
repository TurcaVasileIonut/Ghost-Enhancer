//
//  Albums.swift
//  Enhancer
//
//  Created by Turca Vasile  on 25.05.2024.
//

import Foundation
import SwiftData

@Model
class Album: Identifiable {

    @Attribute(.unique) 
    var id: String = UUID().uuidString
    
    @Relationship(deleteRule: .cascade, inverse: \ProcessedImage.id)
    var enhancedImages: [ProcessedImage]

    init(id: String = UUID().uuidString) {
        self.id = id
        self.enhancedImages = [] 
    }
}

extension Album {
    var lastEnhancedImage: ProcessedImage? {
        enhancedImages.max(by: { $0.date < $1.date })
    }
}
