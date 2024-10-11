package com.deeplearning.bspline

import breeze.linalg._

object ControlPointUpdater {
  def updateControlPoints(
                           x: Double,
                           knots: DenseVector[Double],
                           controlPoints: DenseMatrix[Double],
                           degree: Int,
                           delta: Double
                         ): DenseMatrix[Double] = {
    val numPoints = controlPoints.rows
    require(knots.length >= numPoints + degree + 1,
      "Number of knots must be at least number of control points plus degree plus 1")

    // Find the knot span index
    val knotSpanIndex = knots.toArray.zipWithIndex.find(_._1 > x) match {
      case Some((_, index)) => index - 1
      case None => knots.length - degree - 2
    }

    val knotSpanValue = knots(knotSpanIndex)

    // Update only the closest control point's y-coordinate
    val updatedControlPoints = controlPoints.copy
    val numrows = updatedControlPoints.rows

    var verif = false
    for (i <-0 until numrows) {
      if (controlPoints(i,0)>=knotSpanValue && verif == false)  {
        updatedControlPoints(i,1) = delta
        verif = true
      }
    }

    updatedControlPoints
  }
}

