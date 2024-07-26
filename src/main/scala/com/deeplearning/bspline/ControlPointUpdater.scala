package com.deeplearning.bspline

import breeze.linalg._

case class Point(x: Double, y: Double)

object ControlPointUpdater {
  def updateControlPoints(
                           x: Double,
                           knots: DenseVector[Double],
                           controlPoints: DenseVector[Point],
                           degree: Int,
                           delta: Double
                         ): DenseVector[Point] = {
    require(knots.length == controlPoints.length + degree + 1,
      "Number of knots must be equal to number of control points plus degree plus 1")

    // Find the knot span index
    val knotSpanIndex = knots.toArray.zipWithIndex.find(_._1 > x) match {
      case Some((_, index)) => index - 1
      case None => knots.length - degree - 2
    }

    // Find the closest control point within the relevant span
    val startIndex = math.max(0, knotSpanIndex - degree)
    val endIndex = math.min(startIndex + degree + 1, controlPoints.length)
    val relevantPoints = controlPoints(startIndex until endIndex)

    val closestPointIndex = (startIndex until endIndex).minBy(i =>
      math.abs(controlPoints(i).x - x)
    )

    // Update only the closest control point
    val updatedControlPoints = controlPoints.copy
    updatedControlPoints(closestPointIndex) = Point(
      controlPoints(closestPointIndex).x,
      controlPoints(closestPointIndex).y - delta
    )

    updatedControlPoints
  }

  def printUpdatedControlPoints(x: Double, knots: DenseVector[Double], controlPoints: DenseVector[Point], degree: Int, delta: Double): Unit = {
    val updatedPoints = updateControlPoints(x, knots, controlPoints, degree, delta)
    println(s"For x = $x:")
    updatedPoints.toArray.zipWithIndex.foreach { case (point, index) =>
      println(s"Control point $index: (${point.x}, ${point.y})")
    }
    println()
  }
}
