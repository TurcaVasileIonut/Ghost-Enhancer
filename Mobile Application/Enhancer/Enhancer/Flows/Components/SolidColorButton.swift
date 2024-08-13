//
//  SolidColorButton.swift
//  Enhancer
//
//  Created by Turca Vasile  on 23.05.2024.
//

import SwiftUI

struct SolidColorButton: View {
    var text: String
    var background: Color = Color("RoyalBlue")
    var foreground: Color = .black
    var action: () -> ()
    
    var body: some View {
        Button(text) {
            action()
        }
        .padding()
        .frame(maxWidth:.infinity)
        .background(background)
        .foregroundColor(foreground)
        .cornerRadius(10)
    }
}

struct SolidColorButton_Preview: PreviewProvider {
    static var previews: some View {
        SolidColorButton(text: "Button", action: {} )
    }
}
