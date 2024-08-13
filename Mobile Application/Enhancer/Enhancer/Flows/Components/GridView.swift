//
//  GridView.swift
//  Enhancer
//
//  Created by Turca Vasile  on 24.05.2024.
//

import SwiftUI

struct GridIndex {
    var row: Int
    var column: Int
}

struct GridView<T: View>: View {
    let rows: Int
    let columns: Int
    var spacing: CGFloat = 0
    
    @ViewBuilder var views: (_ index: GridIndex) -> T
    
    var body: some View {
        HStack(spacing: spacing) {
            ForEach(0..<columns, id: \.self) { column in
                VStack(spacing: spacing) {
                    ForEach(0..<rows, id: \.self) { row in
                        views(GridIndex(row: row, column: column))
                    }
                }
            }
        }
    }
}
