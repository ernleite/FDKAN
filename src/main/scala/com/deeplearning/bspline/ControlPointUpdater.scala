package com.deeplearning.bspline

import breeze.linalg._

object ControlPointUpdater {
  def updateControlPoints(
                           x: Double,
                           knots: DenseVector[Double],
                           controlPoints: DenseVector[Double],
                           degree: Int,
                           delta: Double
                         ): DenseVector[Double] = {
    val numPoints = controlPoints.length / 2
    require(knots.length >= numPoints + degree + 1,
      "Number of knots must be at least number of control points plus degree plus 1")

    // Find the knot span index
    val knotSpanIndex = knots.toArray.zipWithIndex.find(_._1 > x) match {
      case Some((_, index)) => index - 1
      case None => knots.length - degree - 2
    }

    // Find the closest control point within the relevant span
    val startIndex = math.max(0, knotSpanIndex - degree)
    val endIndex = math.min(startIndex + degree + 1, numPoints)

    val closestPointIndex = (startIndex until endIndex).minBy(i =>
      math.abs(controlPoints(i) - x)
    )

    // Update only the closest control point's y-coordinate
    val updatedControlPoints = controlPoints.copy
    updatedControlPoints(numPoints + closestPointIndex) -= delta

    updatedControlPoints
  }

  def printUpdatedControlPoints(x: Double, knots: DenseVector[Double], controlPoints: DenseVector[Double], degree: Int, delta: Double): Unit = {
    val updatedPoints = updateControlPoints(x, knots, controlPoints, degree, delta)
    val numPoints = updatedPoints.length / 2
    println(s"For x = $x:")
    for (i <- 0 until numPoints) {
      println(s"Control point $i: (${updatedPoints(i)}, ${updatedPoints(numPoints + i)})")
    }
    println()
  }
}

