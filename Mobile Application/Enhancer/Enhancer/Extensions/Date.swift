//
//  Date.swift
//  Enhancer
//
//  Created by Turca Vasile  on 24.05.2024.
//

import Foundation

extension Date {

    var stripTime: String {
        let dateFormatter = DateFormatter()
        dateFormatter.timeStyle = DateFormatter.Style.none
        dateFormatter.dateStyle = DateFormatter.Style.short

        dateFormatter.dateFormat = "dd MM yyyy"
        return dateFormatter.string(from: self)
    }

}
