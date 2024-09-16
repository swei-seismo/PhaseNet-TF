package main

import "math"

// pad zero to input if length<windowLength. Also make the final length divisible by hopLength
func padZeroTransform(input [][]float32, windowLength int, hopLength int) [][]float32 {
	length := len(input[0])
	if length < windowLength {
		length = windowLength
	}
	if length%hopLength != 0 {
		length = length + hopLength - length%hopLength
	}

	output := make([][]float32, len(input))
	for i := range input {
		output[i] = make([]float32, length)
		copy(output[i], input[i])
	}
	return output
}

// input has size (3,total_length), output should has size (batch_size,3,windowLength), with sliding step of hopLength
func waveformToBatchTransform(input [][]float32, windowLength int, hopLength int) [][][]float32 {
	batchSize := (len(input[0])-windowLength)/hopLength + 1
	output := make([][][]float32, batchSize)
	for i := range output {
		output[i] = make([][]float32, len(input))
	}

	for i := 0; i < batchSize; i++ {
		for j := range input {
			output[i][j] = input[j][i*hopLength : i*hopLength+windowLength]
		}
	}

	return output
}

// reverse of waveformToBatchTransform, input has size (batch_size,nclasses,windowLength), output has size (nclasses,total_length)
func batchToWaveformTransform(input [][][]float32, windowLength int, hopLength int) [][]float32 {
	batchSize, channel, _ := len(input), len(input[0]), len(input[0][0])
	totalLength := batchSize*hopLength + windowLength - hopLength
	output := make([][]float32, channel)
	for i := range output {
		output[i] = make([]float32, totalLength)
	}

	// for overlapping part, use max value
	for i := 0; i < batchSize; i++ {
		for j := 0; j < channel; j++ {
			for k := 0; k < windowLength; k++ {
				if output[j][i*hopLength+k] < input[i][j][k] {
					output[j][i*hopLength+k] = input[i][j][k]
				}
			}
		}
	}
	return output
}

// batch normalizer
func batchNormalizeTransform(input [][][]float32) [][][]float32 {
	batchSize := len(input)
	output := make([][][]float32, batchSize)

	for i := 0; i < batchSize; i++ {
		maxStdDev := float32(0)
		meanVals := make([]float32, len(input[i]))
		output[i] = make([][]float32, len(input[i]))
		for j := range input[i] {
			data := make([]float32, len(input[i][j]))
			copy(data, input[i][j])

			sum := float32(0)
			for _, val := range data {
				sum += val
			}
			meanVal := sum / float32(len(data))
			meanVals[j] = meanVal

			for k := range data {
				data[k] -= meanVal
			}

			squaresSum := float32(0)
			for _, val := range data {
				squaresSum += (val * val)
			}
			variance := squaresSum / float32(len(data))
			stdDev := float32(math.Sqrt(float64(variance)))

			if stdDev > maxStdDev {
				maxStdDev = stdDev
			}

			output[i][j] = data
		}

		if maxStdDev == 0 {
			maxStdDev = 1
		}

		for j := range output[i] {
			for k := range output[i][j] {
				output[i][j][k] -= meanVals[j]
				output[i][j][k] /= maxStdDev
			}
		}
	}
	return output
}

// find peaks along 1d array
func findPeaks(data []float32, minHeight float32, minDistance int) ([]int, []float32) {
	peaks := make([]int, 0)

	var potentialPeakIndex int
	var potentialPeakHeight float32

	for i := 1; i < len(data)-1; i++ {
		if data[i] > minHeight && data[i] > data[i-1] && data[i] > data[i+1] {
			if i-potentialPeakIndex >= minDistance || data[i] > potentialPeakHeight {
				potentialPeakIndex = i
				potentialPeakHeight = data[i]
			}
		}
		if i-potentialPeakIndex >= minDistance && potentialPeakHeight > minHeight {
			peaks = append(peaks, potentialPeakIndex)
			potentialPeakHeight = minHeight
		}
	}

	// Add the last potential peak
	if potentialPeakHeight > minHeight {
		peaks = append(peaks, potentialPeakIndex)
	}

	// fetch the height of each peak
	heights := make([]float32, len(peaks))
	for i := range peaks {
		heights[i] = data[peaks[i]]
	}

	return peaks, heights
}

// Softmax function
func softmax(matrix [][][]float32) [][][]float32 {
	// Copy the input matrix to store the result
	result := make([][][]float32, len(matrix))
	for i := range matrix {
		result[i] = make([][]float32, len(matrix[i]))
		for j := range matrix[i] {
			result[i][j] = make([]float32, len(matrix[i][j]))
			copy(result[i][j], matrix[i][j])
		}
	}

	// Apply the softmax
	for i := range matrix {
		for k := range matrix[i][0] {
			sum := float32(0)
			for j := range matrix[i] {
				result[i][j][k] = float32(math.Exp(float64(result[i][j][k])))
				sum += result[i][j][k]
			}
			if sum != 0 {
				for j := range matrix[i] {
					result[i][j][k] /= sum
				}
			}
		}
	}
	return result
}
