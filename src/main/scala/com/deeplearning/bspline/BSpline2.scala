package com.deeplearning.bspline

import scala.collection.mutable.ArrayBuffer

class BSpline2(
                val degree: Int,
                val knots: Array[Double],
                val controlPoints: Array[Double]
              ) {
  private val n = controlPoints.length - 1
  private val m = knots.length - 1

  def evaluate(x: Double): Double = {
    if (x < knots.head || x > knots.last) {
      throw new IllegalArgumentException("Input value out of range")
    }

    val span = findSpan(x)
    val basisFunctions = computeBasisFunctions(span, x)

    var result = 0.0
    for (i <- 0 to degree) {
      result += controlPoints(span - degree + i) * basisFunctions(i)
    }
    result
  }

  private def findSpan(x: Double): Int = {
   // if (x == knots(n + 1)) return n

    var low = degree
    var high = m + 1
    var mid = (low + high) / 2

    while (x < knots(mid) || x >= knots(mid + 1)) {
      if (x < knots(mid)) high = mid
      else low = mid
      mid = (low + high) / 2
    }

    mid
  }

  private def computeBasisFunctions(span: Int, x: Double): Array[Double] = {
    val left = new ArrayBuffer[Double](degree + 1)
    val right = new ArrayBuffer[Double](degree + 1)
    val basisFunctions = Array.fill(degree + 1)(0.0)

    basisFunctions(0) = 1.0
    for (j <- 1 to degree) {
      left += x - knots(span + 1 - j)
      right += knots(span + j) - x
      var saved = 0.0
      for (r <- 0 until j) {
        val temp = basisFunctions(r) / (right(r) + left(j - r - 1))
        basisFunctions(r) = saved + right(r) * temp
        saved = left(j - r - 1) * temp
      }
      basisFunctions(j) = saved
    }
    basisFunctions
  }
}

// Example usage
object BSplineExample {
  def main(args: Array[String]): Unit = {
    val degree = 3
    val knots = Array(0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0)
    val controlPoints = Array(0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0)

    val bspline2 = new BSpline2(degree, knots, controlPoints)

    val inputValue = 2.5
    val outputValue = bspline2.evaluate(inputValue)
    println(s"Input: $inputValue, Output: $outputValue")
  }
}
