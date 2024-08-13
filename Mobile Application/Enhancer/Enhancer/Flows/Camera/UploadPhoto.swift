//
//  PhotosPicker.swift
//  Enhancer
//
//  Created by Turca Vasile  on 24.05.2024.
//

import SwiftUI
import PhotosUI

struct UploadPhoto: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage? {
        didSet {
            guard let selectedImage = selectedImage else {
                return
            }
            onSelected(selectedImage)
            selectedItem = nil
        }
    }
    var onSelected: (UIImage) -> Void
    
    var body: some View {
        PhotosPicker(selection: $selectedItem, matching: .images) {
            Text(Strings.upload)
                .foregroundColor(Color("RoyalBlue"))
        }
        .onChange(of: selectedItem) {
            Task {
                await loadImage(item: selectedItem)
            }
        }
    }
    
    private func loadImage(item: PhotosPickerItem?) async {
        guard let item = item else {
            return
        }

        do {
            if let data = try await item.loadTransferable(type: Data.self),
               let image = UIImage(data: data) {
                selectedImage = image
            } else {
                print("Could not load image data.")
            }
        } catch {
            print("Error loading image: \(error)")
        }
    }
}


