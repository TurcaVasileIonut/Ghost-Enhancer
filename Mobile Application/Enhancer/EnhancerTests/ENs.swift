//
//  ENs.swift
//  EnhancerTests
//
//  Created by Turca Vasile  on 01.06.2024.
//
import XCTest
import SwiftUI
@testable import Enhancer

final class ENs: XCTestCase {

    override func setUpWithError() throws {
    }

    override func tearDownWithError() throws {
    }

    @MainActor
    func testExample() throws {
        let tabItems = [
            TabItem(id: UUID(), label: "CircleImage", image: Image(systemName: "circle")),
            TabItem(id: UUID(), label: "Rectangle", image: Image(systemName: "rectangle"))
        ]
        let viewModel = PhotoViewerViewModel(items: tabItems) {}
        
        XCTAssert(viewModel.currentLabel == "CircleImage")
        viewModel.imageIndex += 1
        XCTAssert(viewModel.currentLabel == "Rectangle")
    }

}