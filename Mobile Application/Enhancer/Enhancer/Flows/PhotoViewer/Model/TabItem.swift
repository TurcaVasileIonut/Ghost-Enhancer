//
//  TabItem.swift
//  Enhancer
//
//  Created by Turca Vasile  on 24.05.2024.
//

import SwiftUI

struct TabItem: Identifiable, Hashable {
    var id = UUID()
    var label: String 
    var image: Image? 
    
    static func == (lhs: TabItem, rhs: TabItem) -> Bool {
        lhs.id == rhs.id
    }
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}
