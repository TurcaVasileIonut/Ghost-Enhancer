//
//  SuperResolutionService.swift
//  Enhancer
//
//  Created by Turca Vasile  on 29.05.2024.
//

import Foundation
import UIKit
import CoreML
import Accelerate

class SuperResolutionService {
    
    private var model: IE1?
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        do {
            let config = MLModelConfiguration()
            model = try IE1(configuration: config)
        } catch {
            
        }
    }
    
    func analyzeImage(image: UIImage) async -> UIImage? {
        guard let model = self.model,
              let pixelBuffer = image.resize(to: CGSize(width: 16, height: 16))?.cvPixelBuffer else {
            return nil
        }
        let modelInput = IE1Input(x_1: pixelBuffer)
        do {
            let prediction = try await model.prediction(input: modelInput).var_1350
            return try UIImage(from: prediction)
        } catch {
            print("Enhancing failure")
            return nil
        }
    }
}
