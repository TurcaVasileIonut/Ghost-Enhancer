//
//  UIImage.swift
//  Enhancer
//
//  Created by Turca Vasile  on 23.05.2024.
//

import SwiftUI

enum ImageCreationError: Error {
    case unableToCreateCGImage
}

extension UIImage {
    
    convenience init(from pixelBuffer: CVPixelBuffer) throws {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext(options: nil)

        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            throw ImageCreationError.unableToCreateCGImage
        }

        self.init(cgImage: cgImage)
    }
    
    func resize(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContext(size)
        draw(in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return image
    }

    var cvPixelBuffer: CVPixelBuffer? {
        guard let cgImage = self.cgImage else { return nil }

        let options: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]

        let width = cgImage.width
        let height = cgImage.height

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32ARGB, options as CFDictionary, &pixelBuffer)

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])

        let pixelData = CVPixelBufferGetBaseAddress(buffer)

        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()

        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

        if let context = CGContext(data: pixelData, width: width, height: height,
                                   bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                   space: rgbColorSpace,
                                   bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) {
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        }

        CVPixelBufferUnlockBaseAddress(buffer, [])

        return buffer
    }
}

