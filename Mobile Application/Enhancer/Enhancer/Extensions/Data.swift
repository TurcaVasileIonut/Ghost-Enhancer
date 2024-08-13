//
//  Data.swift
//  Enhancer
//
//  Created by Turca Vasile  on 24.05.2024.
//

import Foundation
import SwiftUI

extension Data {
    var image: Image? {
        guard let uiImage = UIImage(data: self) else {
            return nil
        }
        return Image(uiImage: uiImage)
    }
}
