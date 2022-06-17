//
//  MLMultiArrayExtensions.swift
//  AboutTime
//
//  Created by Eugene Dorfman on 17/6/22.
//

import Foundation
import CoreML

public extension MLMultiArray {
    func argmax() -> Int {
        var m = self[0].floatValue
        var mi = 0
        for i in 0..<self.count {
            if self[i].floatValue > m {
                m = self[i].floatValue
                mi = i
            }
        }
        return mi
    }
}
