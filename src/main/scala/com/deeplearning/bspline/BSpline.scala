package com.deeplearning.bspline

import breeze.linalg._
object BSpline {
    def bsplineBasis(i: Int, k: Int, t: Double, knots: DenseVector[Double]): Double = {
      if (k == 1) {
        if (knots(i) <= t && t < knots(i + 1)) 1.0
        else if (t == knots(knots.length - 1) && i == knots.length - 2) 1.0
        else 0.0
      } else {
        val d1 = if (knots(i + k - 1) - knots(i) == 0) 0.0
        else (t - knots(i)) / (knots(i + k - 1) - knots(i))
        val d2 = if (knots(i + k) - knots(i + 1) == 0) 0.0
        else (knots(i + k) - t) / (knots(i + k) - knots(i + 1))
        d1 * bsplineBasis(i, k - 1, t, knots) + d2 * bsplineBasis(i + 1, k - 1, t, knots)
      }
    }

    def bsplineBasisDerivative(i: Int, k: Int, t: Double, knots: DenseVector[Double]): Double = {
      if (k == 1) {
        0.0
      } else {
        val d1 = if (knots(i + k - 1) - knots(i) == 0) 0.0
        else k.toDouble / (knots(i + k - 1) - knots(i))
        val d2 = if (knots(i + k) - knots(i + 1) == 0) 0.0
        else k.toDouble / (knots(i + k) - knots(i + 1))
        d1 * bsplineBasis(i, k - 1, t, knots) - d2 * bsplineBasis(i + 1, k - 1, t, knots)
      }
    }

    def bsplineDerivative(t: Double, controlPoints: DenseVector[Double], knots: DenseVector[Double], k: Int): Double = {
      val n = controlPoints.length/2
      var result = 0.0
      for (i <- 0 until n) {
        val basisDerivative = bsplineBasisDerivative(i, k, t, knots)
        result += controlPoints(i) * basisDerivative
      }
      result
    }

    def bspline(t: Double, controlPoints: DenseMatrix[Double], knots: DenseVector[Double], k: Int): DenseVector[Double] = {
      val n = controlPoints.rows
      val d = controlPoints.cols
      val result = DenseVector.zeros[Double](d)
      for (i <- 0 until n) {
        val basis = bsplineBasis(i, k, t, knots)
        result += controlPoints(i, ::).t * basis
      }
      result
    }

    def compute(controlPoints: DenseMatrix[Double], k: Int, x: Double, knots: DenseVector[Double]): Double = {
      val t = bspline(x, controlPoints, knots, k)(1)
      t
    }

  def findSpan(x: Double, degree: Int, c: Int, knots: Array[Double]): Int = {
    if (x >= knots(c)) return c - 1
    var low = degree
    var high = c
    var mid = (low + high) / 2
    while (x < knots(mid) || x >= knots(mid + 1)) {
      if (x < knots(mid)) high = mid
      else low = mid
      mid = (low + high) / 2
    }
    mid
  }

  def leastSquares(A: DenseMatrix[Double], B: DenseVector[Double]): DenseVector[Double] = {
    // Compute the QR decomposition of A
    val qrResult = qr(A)

    // Extract Q and R matrices
    val Q = qrResult.q
    val R = qrResult.r

    // Compute Q^T * B
    val QtB = Q.t * B

    // Solve R * X = Q^T * B for X
    val X = R \ QtB

    X
  }
}
