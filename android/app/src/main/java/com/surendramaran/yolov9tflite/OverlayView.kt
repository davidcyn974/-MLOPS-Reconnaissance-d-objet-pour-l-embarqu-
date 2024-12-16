package com.surendramaran.yolov9tflite

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()

    private var bounds = Rect()

    init {
        initPaints()
    }

    fun clear() {
        results = listOf()
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.strokeWidth = 12F  // Thicker lines
        boxPaint.style = Paint.Style.STROKE
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        results.forEach {
            // Set color based on class
            boxPaint.color = when(it.clsName.lowercase()) {
                "with_mask" -> Color.GREEN
                "without_mask" -> Color.RED
                "mask_weared_incorrect" -> Color.YELLOW
                else -> Color.WHITE
            }

            val left = it.x1 * width
            val top = it.y1 * height
            val right = it.x2 * width
            val bottom = it.y2 * height

            // Draw box with thicker stroke
            canvas.drawRect(left, top, right, bottom, boxPaint)

            // Draw label with confidence
            val drawableText = "${it.clsName} (${(it.cnf * 100).toInt()}%)"

            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()

            // Draw background for text
            canvas.drawRect(
                left,
                top - textHeight - BOUNDING_RECT_TEXT_PADDING,  // Move text above box
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top,
                textBackgroundPaint
            )

            // Draw text
            canvas.drawText(
                drawableText,
                left + BOUNDING_RECT_TEXT_PADDING/2,
                top - BOUNDING_RECT_TEXT_PADDING/2,
                textPaint
            )
        }
    }

    fun setResults(boundingBoxes: List<BoundingBox>) {
        results = boundingBoxes
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}