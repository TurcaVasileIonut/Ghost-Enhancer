//
//  AccessCameraView.swift
//  Enhancer
//
//  Created by Turca Vasile  on 23.05.2024.
//

import SwiftUI
import PhotosUI

struct AccessCameraView: UIViewControllerRepresentable {
    @State var selectedImage: UIImage? {
        didSet {
            guard let selectedImage = selectedImage else {
                return
            }
            onSelected(selectedImage)
        }
    }
    
    var onSelected: (UIImage) -> Void
    @Environment(\.presentationMode) var isPresented
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let imagePicker = UIImagePickerController()
        imagePicker.sourceType = .camera
        imagePicker.delegate = context.coordinator
        return imagePicker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {
        
    }

    func makeCoordinator() -> Coordinator {
        return Coordinator(picker: self)
    }
}

class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    var picker: AccessCameraView
    
    init(picker: AccessCameraView) {
        self.picker = picker
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        guard let selectedImage = info[.originalImage] as? UIImage else { return }
        self.picker.selectedImage =  resizeImage(image: selectedImage, targetSize: CGSize(width: 256, height: 256))
        self.picker.isPresented.wrappedValue.dismiss()
    }
    
    private func resizeImage(image: UIImage, targetSize: CGSize) -> UIImage {
        let size = image.size

        let widthRatio  = targetSize.width  / size.width
        let heightRatio = targetSize.height / size.height
        let scaleFactor = min(widthRatio, heightRatio)

        let newWidth = size.width * scaleFactor
        let newHeight = size.height * scaleFactor
        let newSize = CGSize(width: newWidth, height: newHeight)

        UIGraphicsBeginImageContextWithOptions(newSize, false, 0)
        image.draw(in: CGRect(origin: .zero, size: newSize))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        return newImage ?? image
    }
}
