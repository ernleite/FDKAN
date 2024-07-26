package com.deeplearning.bspline

import breeze.linalg._

object BSpline {
  def bsplineBasis(i: Int, k: Int, t: Double, knots: DenseVector[Double]): Double = {
    if (k == 1) {
      if (knots(i) <= t && t < knots(i + 1)) 1.0 else 0.0
    } else {
      val d1 = if (knots(i + k - 1) - knots(i) == 0) 0.0
      else (t - knots(i)) / (knots(i + k - 1) - knots(i))
      val d2 = if (knots(i + k) - knots(i + 1) == 0) 0.0
      else (knots(i + k) - t) / (knots(i + k) - knots(i + 1))
      d1 * bsplineBasis(i, k - 1, t, knots) + d2 * bsplineBasis(i + 1, k - 1, t, knots)
    }
  }
  def findAndUpdateControlPoints(x: Double, knots: DenseVector[Double], controlPoints: DenseVector[Double], k: Int, delta: Double): DenseVector[Double] = {
    //require(knots.length == controlPoints.length + k, "Number of knots must be equal to number of control points plus k")

    // Find the knot span index
    val n = controlPoints.length - 1
    val knotSpanIndex = (0 until n + 1).find(i => knots(i) <= x && x < knots(i + 1)).getOrElse {
      if (x == knots(knots.length - 1)) n else throw new IllegalArgumentException(s"x value $x is outside the knot vector range")
    }

    // Calculate the start index for relevant control points
    val startIndex = math.max(0, knotSpanIndex - k + 1)

    // Extract relevant control points
    val endIndex = math.min(startIndex + k, controlPoints.length)
    val relevantControlPoints = controlPoints(startIndex until endIndex)

    // Subtract delta from relevant control points
    val updatedControlPoints = relevantControlPoints - delta

    // Update the control points matrix
    val updatedControlPointsMatrix = controlPoints.copy
    for (i <- startIndex until endIndex) {
      updatedControlPointsMatrix(i) = updatedControlPoints(i - startIndex)
    }

    updatedControlPointsMatrix
  }

  def findRelevantKnots(x: Double, knots: DenseVector[Double], k: Int): (Int, DenseVector[Double]) = {
    // Find the interval where x falls
    val interval = knots.toArray.zipWithIndex.find { case (knot, _) => knot > x }
      .map(_._2).getOrElse(knots.length) - 1

    // Ensure the interval is within the valid range
    val validInterval = math.max(k - 1, math.min(interval, knots.length - k))

    // Calculate the start index for relevant knots
    val startIndex = math.max(0, validInterval - k + 1)

    // Extract relevant knots
    val relevantKnots = knots(startIndex until startIndex + k + 1)

    (startIndex, relevantKnots)
  }


  def findSpan(x: Double,degree:Int,c:Int,knots:Array[Double]): Int = {
    // if (x == knots(n + 1)) return n

    var low = degree
    var high = c + 1
    var mid = (low + high) / 2

    while (x < knots(mid) || x >= knots(mid + 1)) {
      if (x < knots(mid)) high = mid
      else low = mid
      mid = (low + high) / 2
    }

    mid
  }

  def bspline(t: Double, controlPoints: DenseMatrix[Double], knots: DenseVector[Double], k: Int): DenseVector[Double] = {
    val n = controlPoints.rows - 1
    val d = controlPoints.cols
    val result = DenseVector.zeros[Double](d)
    for (i <- 0 to n) {
      val basis = bsplineBasis(i, k, t, knots)
      result += controlPoints(i, ::).t * basis
    }
    result
  }

  // Implement the spline function
  def compute(controlPoints: DenseMatrix[Double], k: Int, x:Double, knots: DenseVector[Double]): Double = {
    bspline(x, controlPoints, knots, k).toArray.sum
  }
}