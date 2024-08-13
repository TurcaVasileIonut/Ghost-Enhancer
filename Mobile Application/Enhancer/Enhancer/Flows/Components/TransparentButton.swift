//
//  TransparentButton.swift
//  Enhancer
//
//  Created by Turca Vasile  on 23.05.2024.
//

import SwiftUI

struct TransparentButton: View {
    
    var text: String
    var action: () -> ()
    
    var body: some View {
        Button(
            action: action,
            label: {
                HStack {
                    Spacer()
                    Text(text)
                        .foregroundColor(Color("RoyalBlue"))
                    Spacer()
                }
                .padding(.vertical)
                .overlay {
                    RoundedRectangle(cornerRadius: CGFloat(10))
                        .stroke(Color("RoyalBlue"), lineWidth: CGFloat(2))
                }
            }
        )
        .listRowBackground(Color.clear)
    }
}

struct TransparentButton_Preview {
    static var previews: some View {
        TransparentButton(text: "Transparent button") {}
    }
}
