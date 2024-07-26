package com.deeplearning.bspline

import breeze.linalg.{DenseMatrix, DenseVector, linspace}
import breeze.plot.{Figure, plot, scatter}
import com.deeplearning.{ActivationManager, Network}

object BSplineFigure {
  val length = 100
  val startIndex = 0.0
  val endIndex = 0.999999999999


  def draw(controlPoints:DenseMatrix[Double], knots:DenseVector[Double], degree:Int, epoch:Int , layer:Int, internalSubLayer:Int ): String = {
    val ts = linspace(startIndex,endIndex, length)
    val points = ts.map(t =>  BSpline.bspline(t, controlPoints, knots, degree))

    // Extract x and y coordinates
    val xCoords = points.map(_(0))
    val yCoords = points.map(_(1))

    // Plot the result
    val f = Figure()
    val p = f.subplot(0)
    p += plot(xCoords, yCoords, name = "B-Spline")
    p += scatter(controlPoints(::, 0), controlPoints(::, 1), _ => 0.1, name = "Control Points")
    p.xlabel = "X"
    p.ylabel = "Y"
    p.title = "B-Spline Curve"
    p.legend = true
    val fileid = "bspline-" + epoch + "_"+ layer + "_" + internalSubLayer
    f.saveas("c://src/tmp/"+fileid+".png")
    fileid
  }
}
